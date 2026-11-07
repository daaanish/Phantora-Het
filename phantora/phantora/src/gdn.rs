//! Versioned semantic keys and measured-only fused timing table. No GPU calls.
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::error::Error;
use std::path::Path;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorMeta {
    pub shape: Vec<i64>,
    pub strides: Vec<i64>,
    pub dtype: String,
    pub requires_grad: bool,
}

impl TensorMeta {
    fn validate(&self) -> Result<(), String> {
        if self.shape.is_empty() || self.shape.len() > 8 || self.shape.len() != self.strides.len()
            || self.shape.iter().any(|&d| d <= 0) || self.strides.iter().any(|&s| s < 0)
            || !matches!(self.dtype.as_str(), "float32" | "float16" | "bfloat16") {
            return Err("invalid fused tensor metadata".into());
        }
        self.shape.iter().try_fold(1i64, |n, &d| n.checked_mul(d))
            .ok_or("fused tensor size overflow")?;
        self.shape.iter().zip(&self.strides).try_fold(1i64, |n, (&d, &s)| {
            n.checked_add((d - 1).checked_mul(s)?)
        }).ok_or("fused tensor storage span overflow")?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GdnKey {
    pub version: u32,
    pub op: String,
    pub direction: String,
    pub inputs: Vec<Option<TensorMeta>>,
    pub grad_enabled: bool,
    pub grad_output: Option<TensorMeta>,
    pub options: BTreeMap<String, String>,
}

impl GdnKey {
    pub fn parse(raw: &str) -> Result<Self, String> {
        if raw.len() > 32768 { return Err("fused key exceeds transport limit".into()); }
        let mut key: Self = serde_json::from_str(raw).map_err(|e| e.to_string())?;
        if key.version != 1 || !matches!(key.direction.as_str(), "forward" | "backward") {
            return Err("unsupported fused key version/direction".into());
        }
        if (key.direction == "backward") != key.grad_output.is_some() {
            return Err("backward requires incoming gradient metadata".into());
        }
        if key.direction == "backward" && !key.inputs.iter().flatten().any(|t| t.requires_grad) {
            return Err("backward requires a differentiable input".into());
        }
        for tensor in key.inputs.iter().flatten().chain(key.grad_output.iter()) { tensor.validate()?; }
        let required: &[&str] = match key.op.as_str() {
            "l2norm" => &["eps", "output_dtype"],
            "causal_conv1d" => &["activation"],
            "chunk_gated_delta_rule" => &["scale", "use_qk_l2norm_in_kernel"],
            _ => return Err("unknown fused boundary".into()),
        };
        if key.options.len() != required.len() || required.iter().any(|name| !key.options.contains_key(*name)) {
            return Err("unsupported fused options".into());
        }
        // Normalize numeric spellings identically for transported and CSV keys.
        for name in ["eps", "scale"] {
            if let Some(text) = key.options.get_mut(name) {
                let number = if let Some(bits) = text.strip_prefix("f64:") {
                    f64::from_bits(u64::from_str_radix(bits, 16).map_err(|e| e.to_string())?)
                } else { text.parse::<f64>().map_err(|e| e.to_string())? };
                if !number.is_finite() || number <= 0.0 { return Err("invalid fused numeric option".into()); }
                *text = format!("f64:{:016x}", number.to_bits());
            }
        }
        let count = match key.op.as_str() { "l2norm" => 1, "causal_conv1d" => 3, _ => 5 };
        if key.inputs.len() != count || key.inputs.iter().enumerate().any(|(i, t)| t.is_none() && !(key.op == "causal_conv1d" && i == 2)) {
            return Err("invalid fused input arity".into());
        }
        let x = key.inputs[0].as_ref().ok_or("missing fused input")?;
        let output_shape = match key.op.as_str() {
            "l2norm" => {
                if !matches!(key.options["output_dtype"].as_str(), "float32" | "float16" | "bfloat16") {
                    return Err("invalid l2norm output dtype".into());
                }
                x.shape.clone()
            }
            "causal_conv1d" => {
                let w = key.inputs[1].as_ref().ok_or("missing convolution weight")?;
                if x.shape.len() != 3 || w.shape.len() != 2 || w.shape[0] != x.shape[1]
                    || !matches!(w.shape[1], 2 | 3 | 4)
                    || !matches!(key.options["activation"].as_str(), "none" | "silu") {
                    return Err("invalid convolution shape/options".into());
                }
                if let Some(b) = &key.inputs[2] {
                    if b.shape != vec![x.shape[1]] { return Err("invalid convolution bias".into()); }
                }
                x.shape.clone()
            }
            _ => {
                let k = key.inputs[1].as_ref().ok_or("missing k")?;
                let v = key.inputs[2].as_ref().ok_or("missing v")?;
                let g = key.inputs[3].as_ref().ok_or("missing g")?;
                let beta = key.inputs[4].as_ref().ok_or("missing beta")?;
                if x.shape.len() != 4 || v.shape.len() != 4 || x.shape != k.shape
                    || x.shape[..2] != v.shape[..2] || v.shape[2] % x.shape[2] != 0
                    || g.shape != v.shape[..3] || beta.shape != g.shape
                    || x.dtype != k.dtype || x.dtype != v.dtype
                    || !matches!(key.options["use_qk_l2norm_in_kernel"].as_str(), "true" | "false") {
                    return Err("invalid chunk shapes/options".into());
                }
                v.shape.clone()
            }
        };
        if let Some(grad) = &key.grad_output {
            let dtype = if key.op == "l2norm" { &key.options["output_dtype"] } else { &x.dtype };
            if grad.shape != output_shape || &grad.dtype != dtype || !key.grad_enabled || grad.requires_grad {
                return Err("incoming gradient does not match fused output".into());
            }
        }
        Ok(key)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub gpu: String,
    pub torch: String,
    pub torch_revision: String,
    pub cuda: String,
    pub triton: String,
    pub fla: String,
    pub causal_conv1d: String,
    pub measurement: String,
}

impl Provenance {
    fn validate(&self) -> Result<(), String> {
        if self.gpu.trim().is_empty() || self.torch.split('+').next() != Some("2.9.1")
            || self.torch_revision != "bbb19f8c856fcf8d7c60267fecb976b4749e2cff"
            || self.cuda != "12.8"
            || self.triton != "3.7.0" || self.fla != "0.4.0" || self.causal_conv1d != "1.5.3.post1"
            || !matches!(self.measurement.as_str(), "cupti_kernel_sum" | "cuda_event_wall") {
            return Err("incompatible fused timing provenance".into());
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct Timing { pub nanos: u64, pub provenance: Provenance }

#[derive(Default, Clone)]
pub struct GdnDb { pub entries: HashMap<GdnKey, Timing> }

impl GdnDb {
    pub fn load(dir: &Path, gpu: &str) -> Result<Self, Box<dyn Error>> {
        let mut db = Self::default();
        let path = dir.join("gdn.csv");
        if !path.exists() { return Ok(db); }
        let mut reader = csv::Reader::from_path(path)?;
        if reader.headers()? != &csv::StringRecord::from(vec!["key", "nanos", "provenance"]) {
            return Err("invalid gdn.csv header".into());
        }
        let mut provenance = None;
        for row in reader.records() {
            let row = row?;
            if row.len() != 3 { return Err("invalid gdn.csv row".into()); }
            let key = GdnKey::parse(&row[0])?;
            let nanos: u64 = row[1].parse()?;
            let source: Provenance = serde_json::from_str(&row[2])?;
            source.validate()?;
            if nanos == 0 || gpu.is_empty() || source.gpu != gpu {
                return Err("zero/discovery timing or mismatched GPU in gdn.csv".into());
            }
            if let Some(previous) = &provenance {
                if previous != &source { return Err("mixed fused timing provenance".into()); }
            }
            provenance = Some(source.clone());
            if db.entries.insert(key, Timing { nanos, provenance: source }).is_some() {
                return Err("duplicate canonical fused timing key".into());
            }
        }
        Ok(db)
    }

    pub fn save(&self, dir: &Path) -> Result<(), Box<dyn Error>> {
        let path = dir.join("gdn.csv");
        if self.entries.is_empty() {
            if path.exists() { std::fs::remove_file(path)?; }
            return Ok(());
        }
        let mut rows = Vec::new();
        for (key, timing) in &self.entries {
            timing.provenance.validate()?;
            if timing.nanos == 0 { return Err("cannot save zero as measured fused timing".into()); }
            rows.push((serde_json::to_string(key)?, timing.nanos.to_string(), serde_json::to_string(&timing.provenance)?));
        }
        rows.sort();
        let mut writer = csv::Writer::from_path(path)?;
        writer.write_record(["key", "nanos", "provenance"])?;
        for (key, nanos, provenance) in rows { writer.write_record([key, nanos, provenance])?; }
        writer.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn examples() -> Vec<Value> {
        serde_json::from_str(include_str!("../../../tests/assets/qwen_hybrid_keys.json")).unwrap()
    }

    fn provenance() -> Provenance {
        Provenance {
            gpu: "test GPU".into(), torch: "2.9.1".into(),
            torch_revision: "bbb19f8c856fcf8d7c60267fecb976b4749e2cff".into(),
            cuda: "12.8".into(), triton: "3.7.0".into(), fla: "0.4.0".into(), causal_conv1d: "1.5.3.post1".into(),
            measurement: "cupti_kernel_sum".into(),
        }
    }

    #[test]
    fn python_keys_round_trip_and_numeric_spellings_match() {
        let values = examples();
        for value in &values {
            let key = GdnKey::parse(&value.to_string()).unwrap();
            assert_eq!(GdnKey::parse(&serde_json::to_string(&key).unwrap()).unwrap(), key);
        }
        let mut alternative = values[0].clone();
        alternative["options"]["eps"] = Value::String("0.000001".into());
        assert_eq!(GdnKey::parse(&values[0].to_string()).unwrap(), GdnKey::parse(&alternative.to_string()).unwrap());
        assert_ne!(GdnKey::parse(&values[0].to_string()).unwrap(), GdnKey::parse(&values[1].to_string()).unwrap());
    }

    #[test]
    fn malformed_keys_are_invalid_and_missing_is_deduplicated() {
        let values = examples();
        let mut estimator = GdnEstimator::default();
        assert_eq!(estimator.estimate(&values[0].to_string()), Duration::ZERO);
        estimator.estimate(&values[0].to_string());
        assert_eq!(estimator.missing.len(), 1);
        for path in ["version", "options", "inputs", "grad_output"] {
            let mut invalid = values[0].clone();
            invalid[path] = Value::Bool(true);
            assert!(GdnKey::parse(&invalid.to_string()).is_err());
        }
        assert_eq!(estimator.estimate("not JSON"), Duration::ZERO);
        assert_eq!(estimator.invalid.len(), 1);
        assert_eq!(estimator.missing.len(), 1);
        let mut invalid = values[1].clone();
        invalid["grad_output"]["shape"][0] = Value::from(99);
        assert!(GdnKey::parse(&invalid.to_string()).is_err());
        let mut invalid = values[0].clone();
        invalid["options"]["eps"] = Value::String("NaN".into());
        assert!(GdnKey::parse(&invalid.to_string()).is_err());
        invalid["unknown"] = Value::Bool(true);
        assert!(GdnKey::parse(&invalid.to_string()).is_err());
    }

    #[test]
    fn measured_database_round_trip_rejects_zero_and_mismatched_provenance() {
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("phantora-gdn-{}-{stamp}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(GdnDb::load(&dir, "test GPU").unwrap().entries.is_empty());
        let key = GdnKey::parse(&examples()[0].to_string()).unwrap();
        let mut db = GdnDb::default();
        db.entries.insert(key.clone(), Timing { nanos: 2500, provenance: provenance() });
        db.save(&dir).unwrap();
        assert_eq!(GdnDb::load(&dir, "test GPU").unwrap().entries[&key].nanos, 2500);
        assert!(GdnDb::load(&dir, "other GPU").is_err());
        let mut wrong = provenance();
        wrong.fla = "other".into();
        assert!(wrong.validate().is_err());
        let mut wrong_cuda = provenance();
        wrong_cuda.cuda = "12.7".into();
        assert!(wrong_cuda.validate().is_err());
        db.entries.get_mut(&key).unwrap().nanos = 0;
        assert!(db.save(&dir).is_err());
        let mut estimator = GdnEstimator::default();
        estimator.estimate(&examples()[0].to_string());
        estimator.write_discovery(&dir).unwrap();
        assert!(dir.join("gdn.missing.csv").is_file());
        let status: Value = serde_json::from_slice(&std::fs::read(dir.join("gdn.status.json")).unwrap()).unwrap();
        assert_eq!(status["valid"], false);
        // Discovery has its own file and cannot overwrite the measured table.
        assert_eq!(GdnDb::load(&dir, "test GPU").unwrap().entries[&key].nanos, 2500);
        let mut writer = csv::Writer::from_path(dir.join("gdn.csv")).unwrap();
        writer.write_record(["key", "nanos", "provenance"]).unwrap();
        writer.write_record([serde_json::to_string(&key).unwrap(), "0".into(), serde_json::to_string(&provenance()).unwrap()]).unwrap();
        writer.flush().unwrap();
        assert!(GdnDb::load(&dir, "test GPU").is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[derive(Default)]
pub struct GdnEstimator {
    pub db: GdnDb,
    pub missing: HashSet<GdnKey>,
    pub invalid: HashSet<String>,
}

impl GdnEstimator {
    pub fn estimate(&mut self, raw: &str) -> Duration {
        let key = match GdnKey::parse(raw) {
            Ok(key) => key,
            Err(error) => {
                self.invalid.insert(error.clone());
                eprintln!("INVALID fused event: {error}");
                return Duration::ZERO;
            }
        };
        if let Some(timing) = self.db.entries.get(&key) { return Duration::from_nanos(timing.nanos); }
        if self.missing.insert(key) {
            eprintln!("INVALID fused timing coverage: missing measured key; discovery only (run offline fused profiler)");
        }
        Duration::ZERO
    }

    pub fn write_discovery(&self, dir: &Path) -> Result<(), Box<dyn Error>> {
        std::fs::create_dir_all(dir)?;
        let mut writer = csv::Writer::from_path(dir.join("gdn.missing.csv"))?;
        writer.write_record(["key", "status"])?;
        let mut keys = self.missing.iter().map(serde_json::to_string).collect::<Result<Vec<_>, _>>()?;
        keys.sort();
        for key in keys { writer.write_record([key.as_str(), "missing"])?; }
        writer.flush()?;
        let mut errors: Vec<_> = self.invalid.iter().cloned().collect();
        errors.sort();
        std::fs::write(dir.join("gdn.status.json"), serde_json::to_vec_pretty(&serde_json::json!({
            "valid": self.missing.is_empty() && self.invalid.is_empty(),
            "missing_keys": self.missing.len(), "invalid_events": errors,
            "profiling": "external fused profiler required; no synthetic timing fallback"
        }))?)?;
        Ok(())
    }
}
