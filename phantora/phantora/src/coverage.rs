//! Opt-in run/region accounting; never exits while workers may await replies.
use crate::gdn::GdnKey;
use cuda_call::HostId;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub run_id: String,
    pub expected_world: u32,
    pub expected: BTreeMap<u32, BTreeMap<String, u64>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Control {
    version: u32, run_id: String, phase: String, rank: u32, world_size: u32,
    expected: BTreeMap<String, u64>,
    #[serde(default)]
    observation: Option<Observation>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Observation { op: String, kind: String, signature: String }

#[derive(Serialize)]
struct Rank {
    host: HostId,
    started: bool,
    finished: bool,
    exited: bool,
    observer_started: bool,
    observer_finished: bool,
    actual: BTreeMap<String, u64>,
    expected: BTreeMap<String, u64>,
}

#[derive(Default, Serialize)]
struct OpCount { count: u64, example: String }

pub struct Coverage {
    plan: Plan,
    path: PathBuf,
    mode: String,
    ranks: BTreeMap<u32, Rank>,
    hosts: HashMap<HostId, u32>,
    errors: Vec<String>,
    categories: BTreeMap<String, BTreeMap<String, OpCount>>,
    outside_region: u64,
    malformed_messages: u64,
}

impl Coverage {
    pub fn new(plan: Plan, path: PathBuf, mode: String) -> Result<Self, String> {
        if !matches!(mode.as_str(), "strict" | "discovery") || plan.run_id.is_empty() || plan.expected_world == 0
            || plan.expected.keys().copied().collect::<Vec<_>>() != (0..plan.expected_world).collect::<Vec<_>>() {
            return Err("invalid coverage rank manifest".into());
        }
        let names: Vec<_> = ["causal_conv1d", "l2norm", "chunk_gated_delta_rule"].iter()
            .flat_map(|op| [format!("{op}/forward"), format!("{op}/backward")]).collect();
        for counts in plan.expected.values() {
            if counts.len() != 6 || names.iter().any(|name| !counts.contains_key(name)) {
                return Err("incomplete expected boundary count manifest".into());
            }
        }
        Ok(Self { plan, path, mode, ranks: BTreeMap::new(), hosts: HashMap::new(), errors: vec![],
            categories: BTreeMap::new(), outside_region: 0, malformed_messages: 0 })
    }

    pub fn from_args(args: &crate::args::Args) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        if args.coverage_mode.is_none() && args.run_id.is_none() && args.expected_ranks.is_none()
            && args.coverage_plan.is_none() && args.coverage_status.is_none() { return Ok(None); }
        let plan: Plan = serde_json::from_slice(&std::fs::read(args.coverage_plan.as_ref().ok_or("missing coverage plan")?)?)?;
        if args.run_id.as_ref() != Some(&plan.run_id) || args.expected_ranks != Some(plan.expected_world) {
            return Err("coverage CLI identity does not match manifest".into());
        }
        Ok(Some(Self::new(plan, args.coverage_status.clone().ok_or("missing coverage status path")?,
            args.coverage_mode.clone().ok_or("missing coverage mode")?)?))
    }

    pub fn error(&mut self, message: String) {
        if !self.errors.contains(&message) { self.errors.push(message); }
    }

    pub fn malformed(&mut self, message: String) {
        self.malformed_messages += 1;
        self.error(message);
    }

    pub fn control(&mut self, host: &HostId, raw: &str) {
        let message: Control = match serde_json::from_str(raw) {
            Ok(message) => message,
            Err(error) => { self.malformed(format!("malformed region control: {error}")); return; }
        };
        if message.version != 1 || message.run_id != self.plan.run_id || message.world_size != self.plan.expected_world
            || message.rank >= self.plan.expected_world {
            self.error("stale or incompatible rank registration/control".into()); return;
        }
        if (message.phase == "device_only") != message.observation.is_some() {
            self.malformed("invalid device-only observation payload".into()); return;
        }
        if message.phase == "register" {
            if self.ranks.contains_key(&message.rank) || self.hosts.contains_key(host)
                || message.expected != self.plan.expected[&message.rank] {
                self.error("duplicate rank/host or incorrect expected event counts".into()); return;
            }
            self.hosts.insert(host.clone(), message.rank);
            self.ranks.insert(message.rank, Rank { host: host.clone(), started: false, finished: false,
                exited: false, observer_started: false, observer_finished: false, actual: BTreeMap::new(), expected: message.expected });
            return;
        }
        if self.hosts.get(host) != Some(&message.rank) || !message.expected.is_empty() {
            self.error("unregistered or mismatched region host/rank".into()); return;
        }
        if message.phase == "device_only" {
            if self.active(host).is_none() { return; }
            let observation = message.observation.unwrap();
            let category = match (observation.kind.as_str(), observation.op.as_str()) {
                ("allocation_only", "aten::empty" | "aten::empty_strided" | "aten::empty_like") => "device_only_allocation",
                ("native_transfer", "aten::_to_copy") => "native_transfer_observation",
                ("unsupported", _) => "unmapped_device_only",
                _ => { self.malformed("invalid device-only classification".into()); return; }
            };
            let entry = self.categories.entry(category.into()).or_default().entry(observation.op.clone()).or_default();
            entry.count += 1;
            if entry.example.is_empty() { entry.example = observation.signature.chars().take(1024).collect(); }
            if category == "unmapped_device_only" { self.error(format!("unmapped CUDA device-only operation: {}", observation.op)); }
            return;
        }
        let rank = self.ranks.get_mut(&message.rank).unwrap();
        match message.phase.as_str() {
            "observer_start" if !rank.observer_started && !rank.started => rank.observer_started = true,
            "observer_finish" if rank.observer_started && rank.started && !rank.finished && !rank.observer_finished => rank.observer_finished = true,
            "start" if rank.observer_started && !rank.started && !rank.finished => rank.started = true,
            "finish" if rank.started && rank.observer_finished && !rank.finished => {
                rank.finished = true;
                if rank.expected.iter().any(|(name, count)| rank.actual.get(name).copied().unwrap_or(0) != *count) {
                    self.error(format!("rank {} dropped or extra fused events", message.rank));
                }
            }
            _ => self.error("duplicate/out-of-order training region transition".into()),
        }
    }

    pub fn exit(&mut self, host: &HostId) {
        if let Some(id) = self.hosts.get(host) {
            self.ranks.get_mut(id).unwrap().exited = true;
        } else { self.error("unregistered rank exited".into()); }
    }

    fn active(&mut self, host: &HostId) -> Option<u32> {
        let rank = self.hosts.get(host).copied();
        if rank.and_then(|id| self.ranks.get(&id)).is_some_and(|r| r.started && !r.finished) { rank }
        else { self.outside_region += 1; None }
    }

    pub fn semantic(&mut self, host: &HostId, raw: &str) {
        let Some(rank) = self.active(host) else { return; };
        match GdnKey::parse(raw) {
            Ok(key) => { *self.ranks.get_mut(&rank).unwrap().actual.entry(format!("{}/{}", key.op, key.direction)).or_default() += 1; }
            Err(error) => self.malformed(format!("malformed fused event: {error}")),
        }
    }

    pub fn torch(&mut self, host: &HostId, name: &str, signature: &str, modeled: bool, recognized: bool) {
        if self.active(host).is_none() { return; }
        let category = if modeled { "modeled_compute" } else if metadata_only(name) { "metadata_view" }
            else if matches!(name, "aten::empty" | "aten::empty_strided" | "aten::empty_like") { "allocation_only" }
            else if recognized { "malformed_compute" } else { "unmapped_compute" };
        let entry = self.categories.entry(category.into()).or_default().entry(name.into()).or_default();
        entry.count += 1;
        if entry.example.is_empty() { entry.example = signature.chars().take(1024).collect(); }
        if category == "unmapped_compute" { self.error(format!("unmapped CUDA operation: {name}")); }
        if category == "malformed_compute" { self.malformed(format!("malformed CUDA operation: {name}")); }
    }

    pub fn write(&self, misses: &BTreeMap<String, usize>) -> std::io::Result<()> {
        let complete = self.ranks.len() == self.plan.expected_world as usize && self.ranks.values().all(|r| r.started && r.finished && r.exited && r.observer_started && r.observer_finished);
        let valid = complete && self.errors.is_empty() && misses.values().all(|&n| n == 0);
        let value = serde_json::json!({ "version": 1, "run_id": self.plan.run_id,
            "phase": if complete { "complete" } else if self.ranks.is_empty() { "ready" } else { "running" },
            "valid": valid, "mode": self.mode, "expected_world": self.plan.expected_world,
            "scope": "training-region event counts, outermost CUDA operators and device-only observer; timing misses across entire run",
            "ranks": self.ranks, "errors": self.errors, "timing_misses": misses,
            "categories": self.categories, "outside_region_events": self.outside_region,
            "malformed_messages": self.malformed_messages });
        if let Some(parent) = self.path.parent() { std::fs::create_dir_all(parent)?; }
        let temporary = self.path.with_extension("tmp");
        std::fs::write(&temporary, serde_json::to_vec_pretty(&value)?)?;
        std::fs::rename(temporary, &self.path)
    }
}

fn metadata_only(name: &str) -> bool {
    matches!(name, "aten::view" | "aten::_unsafe_view" | "aten::_reshape_alias" | "aten::transpose"
        | "aten::t" | "aten::permute" | "aten::squeeze" | "aten::unsqueeze" | "aten::detach"
        | "aten::alias" | "aten::as_strided" | "aten::expand" | "aten::expand_as"
        | "aten::slice" | "aten::select" | "aten::narrow" | "aten::split" | "aten::split_with_sizes" | "aten::unbind")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn coverage(world: u32) -> Coverage {
        let counts: BTreeMap<_, _> = ["causal_conv1d", "l2norm", "chunk_gated_delta_rule"].iter()
            .flat_map(|op| [format!("{op}/forward"), format!("{op}/backward")]).map(|name| (name, 0)).collect();
        Coverage::new(Plan { run_id: "test".into(), expected_world: world,
            expected: (0..world).map(|rank| (rank, counts.clone())).collect() }, PathBuf::new(), "strict".into()).unwrap()
    }
    fn host(pid: u32) -> HostId { HostId { hostname: "host-1".into(), pid } }
    fn send(run: &mut Coverage, host: &HostId, rank: u32, phase: &str) {
        if phase == "start" { send(run, host, rank, "observer_start"); }
        if phase == "finish" { send(run, host, rank, "observer_finish"); }
        let expected = if phase == "register" { run.plan.expected[&rank].clone() } else { BTreeMap::new() };
        let raw = serde_json::json!({"version": 1, "run_id": "test", "phase": phase,
            "rank": rank, "world_size": run.plan.expected_world, "expected": expected}).to_string();
        run.control(host, &raw);
    }

    #[test]
    fn all_ranks_finish_and_exit_before_status_is_complete() {
        let mut run = coverage(2);
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!("phantora-coverage-{}-{stamp}.json", std::process::id()));
        run.path = path.clone();
        for rank in 0..2 {
            send(&mut run, &host(rank + 1), rank, "register");
            send(&mut run, &host(rank + 1), rank, "start");
            send(&mut run, &host(rank + 1), rank, "finish");
        }
        run.exit(&host(1));
        run.write(&BTreeMap::new()).unwrap();
        let status: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(status["phase"], "running");
        run.exit(&host(2));
        run.write(&BTreeMap::new()).unwrap();
        let status: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(status["valid"], true);
        run.write(&BTreeMap::from([("compute".into(), 1)])).unwrap();
        let status: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(status["valid"], false);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn stale_duplicate_rank_and_dropped_backward_are_invalid() {
        let mut run = coverage(1);
        run.control(&host(1), r#"{"version":1,"run_id":"stale","phase":"register","rank":0,"world_size":1,"expected":{}}"#);
        assert!(!run.errors.is_empty());
        send(&mut run, &host(1), 0, "register");
        send(&mut run, &host(2), 0, "register");
        assert!(run.errors.iter().any(|e| e.contains("duplicate")));
        let mut run = coverage(1);
        run.plan.expected.get_mut(&0).unwrap().insert("l2norm/backward".into(), 1);
        send(&mut run, &host(1), 0, "register");
        send(&mut run, &host(1), 0, "start");
        send(&mut run, &host(1), 0, "finish");
        assert!(run.errors.iter().any(|e| e.contains("dropped")));
    }

    #[test]
    fn observer_activation_and_classifications_are_required() {
        let mut run = coverage(1);
        send(&mut run, &host(1), 0, "register");
        // Missing Python dispatch-mode activation cannot create a valid region.
        let raw = serde_json::json!({"version": 1, "run_id": "test", "phase": "start",
            "rank": 0, "world_size": 1, "expected": {}}).to_string();
        run.control(&host(1), &raw);
        assert!(!run.ranks[&0].started);
        assert!(!run.errors.is_empty());
        let mut run = coverage(1);
        send(&mut run, &host(1), 0, "register");
        send(&mut run, &host(1), 0, "start");
        for (op, kind) in [("aten::empty", "allocation_only"), ("aten::_to_copy", "native_transfer"),
                           ("aten::ones_like", "unsupported")] {
            let raw = serde_json::json!({"version": 1, "run_id": "test", "phase": "device_only",
                "rank": 0, "world_size": 1, "expected": {},
                "observation": {"op": op, "kind": kind, "signature": "CPU->CUDA device argument"}}).to_string();
            run.control(&host(1), &raw);
        }
        assert_eq!(run.categories["device_only_allocation"]["aten::empty"].count, 1);
        assert_eq!(run.categories["native_transfer_observation"]["aten::_to_copy"].count, 1);
        assert_eq!(run.categories["unmapped_device_only"]["aten::ones_like"].count, 1);
        assert_eq!(run.errors.len(), 1);
        let raw = serde_json::json!({"version": 1, "run_id": "test", "phase": "device_only",
            "rank": 0, "world_size": 1, "expected": {},
            "observation": {"op": "aten::zeros", "kind": "allocation_only", "signature": "forged exemption"}}).to_string();
        run.control(&host(1), &raw);
        assert_eq!(run.malformed_messages, 1);
    }

    #[test]
    fn metadata_does_not_hide_copy_compute_or_initialization_scope() {
        let mut run = coverage(1);
        run.torch(&host(1), "aten::exp", "init", false, false);
        assert!(run.errors.is_empty());
        send(&mut run, &host(1), 0, "register");
        send(&mut run, &host(1), 0, "start");
        run.torch(&host(1), "aten::view", "shape", false, false);
        run.torch(&host(1), "aten::empty_strided", "device-only allocation", false, false);
        run.torch(&host(1), "aten::exp", "float tensor", true, true);
        assert!(run.errors.is_empty());
        run.torch(&host(1), "aten::contiguous", "copy", false, false);
        run.torch(&host(1), "aten::zeros", "device-only compute", false, false);
        assert_eq!(run.errors.len(), 2);
        assert_eq!(run.outside_region, 1);
        run.torch(&host(1), "aten::exp", "bad arguments", false, true);
        assert_eq!(run.categories["malformed_compute"]["aten::exp"].count, 1);
        assert_eq!(run.malformed_messages, 1);
        run.semantic(&host(1), "malformed");
        assert_eq!(run.errors.len(), 4);
        assert_eq!(run.malformed_messages, 2);
    }
}
