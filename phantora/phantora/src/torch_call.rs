use cuda_call::{CudaStream, HostId, ResponseId};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct TorchCallJson {
    pid: u32,
    tid: i32,
    hostname: String,
    stream: Option<(i32, i32)>,
    cur: i64,
    name: String,
    args: Vec<TorchValueJson>,
}

impl TorchCallJson {
    pub fn into_msg(self) -> TorchCallMsg {
        let args = self
            .args
            .into_iter()
            .map(TorchValueJson::into_value)
            .collect();
        TorchCallMsg {
            pid: self.pid,
            tid: self.tid,
            hostname: self.hostname,
            stream: self.stream,
            curr_time: self.cur,
            name: self.name,
            args,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TorchValueJson {
    Tensor {
        shape: Vec<i64>,
        dtype: i32,
        device: String,
    },
    Tuple {
        elements: Vec<TorchValueJson>,
    },
    List {
        elements: Vec<TorchValueJson>,
    },
    Double {
        value: f64,
    },
    Int {
        value: i64,
    },
    Bool {
        value: bool,
    },
    String {
        value: String,
    },
    Device {
        value: String,
    },
}

impl TorchValueJson {
    fn into_value(self) -> TorchValue {
        match self {
            TorchValueJson::Tensor {
                shape,
                dtype,
                device,
            } => {
                let dtype = tch::Kind::from_c_int(dtype).unwrap();
                let device = parse_device(device);
                TorchValue::Tensor(shape, dtype, device)
            }
            TorchValueJson::Tuple { elements } => {
                let elements = elements
                    .into_iter()
                    .map(TorchValueJson::into_value)
                    .collect();
                TorchValue::Tuple(elements)
            }
            TorchValueJson::List { elements } => {
                let elements = elements
                    .into_iter()
                    .map(TorchValueJson::into_value)
                    .collect();
                TorchValue::List(elements)
            }
            TorchValueJson::Double { value } => TorchValue::Double(value),
            TorchValueJson::Int { value } => TorchValue::Int(value),
            TorchValueJson::Bool { value } => TorchValue::Bool(value),
            TorchValueJson::String { value } => TorchValue::String(value),
            TorchValueJson::Device { value } => {
                if value == "cuda" { TorchValue::CurrentCudaDevice }
                else { TorchValue::Device(parse_device(value)) }
            }
        }
    }
}

fn parse_device<T: AsRef<str>>(s: T) -> tch::Device {
    let s = s.as_ref();
    if let Some(id) = s.strip_prefix("cuda:") {
        tch::Device::Cuda(id.parse().unwrap())
    } else if s == "cpu" {
        tch::Device::Cpu
    } else if s == "meta" {
        tch::Device::Cpu
    } else {
        log::warn!("Unknown device: {}", s);
        tch::Device::Cpu
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TorchCallMsg {
    pub pid: u32,
    pub tid: i32,
    pub hostname: String,
    pub stream: Option<(i32, i32)>,
    pub curr_time: i64,
    pub name: String,
    pub args: Vec<TorchValue>,
}

impl TorchCallMsg {
    pub fn is_aten(&self) -> bool {
        // TODO: support convolution_backward in bindings
        self.name.starts_with("aten::") && self.name != "aten::convolution_backward"
    }

    pub fn gpu_index(&self) -> Option<usize> {
        self.args.iter().find_map(TorchValue::gpu_index)
    }

    pub fn cuda_argument_index(&self) -> Option<usize> {
        fn find(value: &TorchValue, current: usize) -> Option<usize> {
            match value {
                TorchValue::CurrentCudaDevice => Some(current),
                TorchValue::Device(tch::Device::Cuda(device)) => Some(*device),
                TorchValue::List(values) | TorchValue::Tuple(values) => values.iter().find_map(|value| find(value, current)),
                _ => value.gpu_index(),
            }
        }
        let current = self.stream.map(|(device, _)| device.max(0) as usize).unwrap_or(0);
        self.args.iter().find_map(|value| find(value, current))
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum TorchValue {
    Tensor(Vec<i64>, tch::Kind, tch::Device),
    Tuple(Vec<TorchValue>),
    List(Vec<TorchValue>),
    Double(f64),
    Int(i64),
    Bool(bool),
    String(String),
    Device(tch::Device),
    CurrentCudaDevice,
}

impl TorchValue {
    pub fn gpu_index(&self) -> Option<usize> {
        match self {
            TorchValue::Tensor(_, _, device) => match device {
                tch::Device::Cuda(i) => Some(*i as usize),
                _ => None,
            },
            TorchValue::Tuple(elements) | TorchValue::List(elements) => {
                elements.iter().find_map(TorchValue::gpu_index)
            }
            _ => None,
        }
    }
}

fn maybe_tensor_list(value: &[TorchValue]) -> Option<Vec<TensorInfo>> {
    value
        .iter()
        .map(|v| match v {
            TorchValue::Tensor(shape, dtype, _) => Some(TensorInfo {
                shape: shape.clone(),
                dtype: *dtype,
            }),
            _ => None,
        })
        .collect()
}

fn qwen_call(op: &str, args: &[TorchValue]) -> Option<TorchCallInfo> {
    use TorchValue::*;
    fn scalar(value: &TorchValue) -> Option<u64> {
        let number = match value { Int(v) => *v as f64, Double(v) => *v, _ => return None };
        number.is_finite().then_some(number.to_bits())
    }
    let mut inputs = Vec::new();
    let mut scalars = Vec::new();
    let mut dims = Vec::new();
    let mut keepdim = false;
    let mut dtype = None;
    let tensor = |value: &TorchValue| match value {
        Tensor(shape, kind, _) => Some(TensorInfo { shape: shape.clone(), dtype: *kind }), _ => None,
    };
    match (op, args) {
        ("exp" | "neg" | "sigmoid" | "silu" | "rsqrt", [x]) => inputs.push(tensor(x)?),
        ("sigmoid_backward" | "silu_backward", [grad, x]) => {
            inputs.extend([tensor(grad)?, tensor(x)?]);
        }
        ("pow", [x @ Tensor(..), exponent]) => {
            let input = tensor(x)?;
            let exponent = scalar(exponent)?;
            if f64::from_bits(exponent) == 2.0 { return Some(TorchCallInfo::Pow(input)); }
            inputs.push(input);
            scalars.push(exponent);
        }
        ("div_", [x, other @ Tensor(..)]) => inputs.extend([tensor(x)?, tensor(other)?]),
        ("div_", [x, other]) => { inputs.push(tensor(x)?); scalars.push(scalar(other)?); }
        ("to", [x, Int(kind), Bool(_), Bool(copy)]) => {
            if *copy { return None; }
            return Some(TorchCallInfo::ConvDType(tensor(x)?, tch::Kind::from_c_int(*kind as i32).ok()?));
        }
        ("softplus", [x]) => { inputs.push(tensor(x)?); scalars.extend([1f64.to_bits(), 20f64.to_bits()]); }
        ("softplus", [x, beta, threshold]) => {
            inputs.push(tensor(x)?); scalars.extend([scalar(beta)?, scalar(threshold)?]);
        }
        ("softplus_backward", [grad, x, beta, threshold]) => {
            inputs.extend([tensor(grad)?, tensor(x)?]); scalars.extend([scalar(beta)?, scalar(threshold)?]);
        }
        ("mean" | "sum", [x, tail @ ..]) => {
            let input = tensor(x)?;
            match tail {
                [] => { if op == "sum" { return Some(TorchCallInfo::Sum(input)); } }
                [List(values), Bool(keep), rest @ ..] => {
                    for value in values { if let Int(dim) = value { dims.push(*dim); } else { return None; } }
                    keepdim = *keep;
                    match rest { [] => (), [Int(kind)] => dtype = Some(tch::Kind::from_c_int(*kind as i32).ok()?), _ => return None }
                }
                [Int(dim), Bool(keep)] => { dims.push(*dim); keepdim = *keep; }
                [Int(kind)] => dtype = Some(tch::Kind::from_c_int(*kind as i32).ok()?),
                _ => return None,
            }
            let rank = input.shape.len() as i64;
            for dim in &mut dims {
                if *dim < 0 { *dim += rank; }
                if *dim < 0 || *dim >= rank { return None; }
            }
            dims.sort_unstable();
            if dims.windows(2).any(|pair| pair[0] == pair[1]) { return None; }
            inputs.push(input);
        }
        _ => return None,
    }
    Some(TorchCallInfo::QwenOp { op: op.into(), inputs, scalars, dims, keepdim, dtype })
}

impl TorchCallMsg {
    pub fn into_call(self, arg_device: i32) -> Option<TorchCall> {
        self.into_call_mode(arg_device, false).0
    }

    pub fn into_hybrid_call(self, arg_device: i32) -> Option<TorchCall> {
        self.into_call_mode(arg_device, true).0
    }

    pub fn into_hybrid_classified_call(self, arg_device: i32) -> (Option<TorchCall>, bool) {
        self.into_call_mode(arg_device, true)
    }

    fn into_call_mode(self, arg_device: i32, hybrid: bool) -> (Option<TorchCall>, bool) {
        let stream = match self.stream {
            Some((device, id)) => CudaStream { device, id },
            None => CudaStream {
                device: arg_device,
                id: 0,
            },
        };

        use TorchValue::*;
        let mut recognized = true;
        let call_info = match self.name.as_ref() {
            "aten::mm" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _)] => Some(TorchCallInfo::MM(
                    TensorInfo {
                        shape: shape1.clone(),
                        dtype: *kind1,
                    },
                    TensorInfo {
                        shape: shape2.clone(),
                        dtype: *kind2,
                    },
                )),
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::matmul" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _)] => {
                    Some(TorchCallInfo::MatMul(
                        TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                    ))
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::linear" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _)] => {
                    Some(TorchCallInfo::Linear(
                        TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                        None,
                    ))
                }
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _), Tensor(shape3, kind3, _)] => {
                    Some(TorchCallInfo::Linear(
                        TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                        Some(TensorInfo {
                            shape: shape3.clone(),
                            dtype: *kind3,
                        }),
                    ))
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::bmm" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _)] => Some(TorchCallInfo::BMM(
                    TensorInfo {
                        shape: shape1.clone(),
                        dtype: *kind1,
                    },
                    TensorInfo {
                        shape: shape2.clone(),
                        dtype: *kind2,
                    },
                )),
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::addmm" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _), Tensor(shape3, kind3, _), ..] => {
                    Some(TorchCallInfo::AddMM(
                        TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                        TensorInfo {
                            shape: shape3.clone(),
                            dtype: *kind3,
                        },
                    ))
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::baddbmm" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _), Tensor(shape3, kind3, _), ..] => {
                    Some(TorchCallInfo::BAddBMM(
                        TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                        TensorInfo {
                            shape: shape3.clone(),
                            dtype: *kind3,
                        },
                    ))
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::mul" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _)] => Some(TorchCallInfo::Mul(
                    TensorInfo {
                        shape: shape1.clone(),
                        dtype: *kind1,
                    },
                    TensorInfo {
                        shape: shape2.clone(),
                        dtype: *kind2,
                    },
                )),
                [Tensor(shape, kind, _), _] => Some(TorchCallInfo::MulScalar(TensorInfo {
                    shape: shape.clone(),
                    dtype: *kind,
                })),
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::mul_" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _)] => Some(TorchCallInfo::Mul_(
                    TensorInfo {
                        shape: shape1.clone(),
                        dtype: *kind1,
                    },
                    TensorInfo {
                        shape: shape2.clone(),
                        dtype: *kind2,
                    },
                )),
                [Tensor(shape, kind, _), _] => Some(TorchCallInfo::MulScalar_(TensorInfo {
                    shape: shape.clone(),
                    dtype: *kind,
                })),
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::_foreach_mul_" => match self.args.as_slice() {
                [List(tensor_list1), List(tensor_list2), ..] => maybe_tensor_list(tensor_list1)
                    .and_then(|tensor_list1| {
                        maybe_tensor_list(tensor_list2).and_then(|tensor_list2| {
                            Some(TorchCallInfo::ForeachMul_(tensor_list1, tensor_list2))
                        })
                    })
                    .or_else(|| {
                        log::warn!("{} args not match: {:?}", self.name, self.args);
                        None
                    }),
                [List(tensor_list1), ..] => match maybe_tensor_list(tensor_list1) {
                    Some(tensor_list1) => Some(TorchCallInfo::ForeachMulScalar_(tensor_list1)),
                    None => {
                        log::warn!("{} args not match: {:?}", self.name, self.args);
                        None
                    }
                },
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::add" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _), ..] => {
                    Some(TorchCallInfo::Add(
                        TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                    ))
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::add_" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _), ..] => {
                    Some(TorchCallInfo::Add_(
                        TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                    ))
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::div" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _)] => Some(TorchCallInfo::Div(
                    TensorInfo {
                        shape: shape1.clone(),
                        dtype: *kind1,
                    },
                    TensorInfo {
                        shape: shape2.clone(),
                        dtype: *kind2,
                    },
                )),
                [Tensor(shape, kind, _), _] => Some(TorchCallInfo::DivScalar(TensorInfo {
                    shape: shape.clone(),
                    dtype: *kind,
                })),
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::pow" if hybrid => qwen_call("pow", &self.args),
            "aten::pow" => match self.args.as_slice() {
                [Tensor(shape, kind, _), _] | [_, Tensor(shape, kind, _)] => Some(TorchCallInfo::Pow(TensorInfo {
                    shape: shape.clone(), dtype: *kind,
                })),
                _ => None,
            },
            "aten::exp" | "aten::neg" | "aten::div_" | "aten::sigmoid" | "aten::silu" | "aten::rsqrt"
            | "aten::softplus" | "aten::mean" | "aten::sigmoid_backward"
            | "aten::silu_backward" | "aten::softplus_backward" if hybrid => {
                qwen_call(self.name.trim_start_matches("aten::"), &self.args)
            }
            "aten::addcmul_" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _), Tensor(shape3, kind3, _), ..] => {
                    Some(TorchCallInfo::AddCMul_(
                        TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                        TensorInfo {
                            shape: shape3.clone(),
                            dtype: *kind3,
                        },
                    ))
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::addcdiv_" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _), Tensor(shape3, kind3, _), ..] => {
                    Some(TorchCallInfo::AddCDiv_(
                        TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                        TensorInfo {
                            shape: shape3.clone(),
                            dtype: *kind3,
                        },
                    ))
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::_foreach_addcmul_" => match self.args.as_slice() {
                [List(tensor_list1), List(tensor_list2), List(tensor_list3), ..] => {
                    maybe_tensor_list(tensor_list1)
                        .and_then(|tensor_list1| {
                            maybe_tensor_list(tensor_list2).and_then(|tensor_list2| {
                                maybe_tensor_list(tensor_list3).and_then(|tensor_list3| {
                                    Some(TorchCallInfo::ForeachAddCMul_(
                                        tensor_list1,
                                        tensor_list2,
                                        tensor_list3,
                                    ))
                                })
                            })
                        })
                        .or_else(|| {
                            log::warn!("{} args not match: {:?}", self.name, self.args);
                            None
                        })
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::_foreach_addcdiv_" => match self.args.as_slice() {
                [List(tensor_list1), List(tensor_list2), List(tensor_list3), ..] => {
                    maybe_tensor_list(tensor_list1)
                        .and_then(|tensor_list1| {
                            maybe_tensor_list(tensor_list2).and_then(|tensor_list2| {
                                maybe_tensor_list(tensor_list3).and_then(|tensor_list3| {
                                    Some(TorchCallInfo::ForeachAddCDiv_(
                                        tensor_list1,
                                        tensor_list2,
                                        tensor_list3,
                                    ))
                                })
                            })
                        })
                        .or_else(|| {
                            log::warn!("{} args not match: {:?}", self.name, self.args);
                            None
                        })
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::masked_fill" => match self.args.as_slice() {
                [Tensor(shape, kind, _), Tensor(mask_shape, mask_kind, _), ..] => {
                    Some(TorchCallInfo::MaskedFill(
                        TensorInfo {
                            shape: shape.clone(),
                            dtype: *kind,
                        },
                        TensorInfo {
                            shape: mask_shape.clone(),
                            dtype: *mask_kind,
                        },
                    ))
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::masked_fill_" => match self.args.as_slice() {
                [Tensor(shape, kind, _), Tensor(mask_shape, mask_kind, _), ..] => {
                    Some(TorchCallInfo::MaskedFill_(
                        TensorInfo {
                            shape: shape.clone(),
                            dtype: *kind,
                        },
                        TensorInfo {
                            shape: mask_shape.clone(),
                            dtype: *mask_kind,
                        },
                    ))
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::dropout" => match self.args.as_slice() {
                [Tensor(shape, kind, _), Double(p), Bool(train)] => Some(TorchCallInfo::Dropout(
                    TensorInfo {
                        shape: shape.clone(),
                        dtype: *kind,
                    },
                    (*p * 1_000_000.0).round() as i64,
                    *train,
                )),
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::native_dropout" => match self.args.as_slice() {
                [Tensor(shape, kind, _), Double(p), Bool(train)] => {
                    Some(TorchCallInfo::NativeDropout(
                        TensorInfo {
                            shape: shape.clone(),
                            dtype: *kind,
                        },
                        (*p * 1_000_000.0).round() as i64,
                        *train,
                    ))
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::native_dropout_backward" => match self.args.as_slice() {
                [Tensor(grad_shape, grad_kind, _), Tensor(mask_shape, mask_kind, _), Double(scale)] => {
                    Some(TorchCallInfo::NativeDropoutBackward(
                        TensorInfo {
                            shape: grad_shape.clone(),
                            dtype: *grad_kind,
                        },
                        TensorInfo {
                            shape: mask_shape.clone(),
                            dtype: *mask_kind,
                        },
                        (*scale * 1_000_000.0).round() as i64,
                    ))
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::_fused_dropout" => match self.args.as_slice() {
                [Tensor(shape, kind, _), Double(p)] => Some(TorchCallInfo::FusedDropout(
                    TensorInfo {
                        shape: shape.clone(),
                        dtype: *kind,
                    },
                    (*p * 1_000_000.0).round() as i64,
                )),
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::where" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _), Tensor(shape3, kind3, _)] => {
                    Some(TorchCallInfo::Where(
                        TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                        TensorInfo {
                            shape: shape3.clone(),
                            dtype: *kind3,
                        },
                    ))
                }
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _), _] => {
                    Some(TorchCallInfo::WhereScalar(
                        TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                    ))
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::sqrt" => match self.args.as_slice() {
                [Tensor(shape, kind, _)] => Some(TorchCallInfo::Sqrt(TensorInfo {
                    shape: shape.clone(),
                    dtype: *kind,
                })),
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::gelu" => match self.args.as_slice() {
                [Tensor(shape, kind, _), ..] => Some(TorchCallInfo::Gelu(TensorInfo {
                    shape: shape.clone(),
                    dtype: *kind,
                })),
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::gelu_backward" => match self.args.as_slice() {
                [Tensor(grad_shape, grad_kind, _), Tensor(self_shape, self_kind, _), ..] => {
                    Some(TorchCallInfo::GeluBackward(
                        TensorInfo {
                            shape: grad_shape.clone(),
                            dtype: *grad_kind,
                        },
                        TensorInfo {
                            shape: self_shape.clone(),
                            dtype: *self_kind,
                        },
                    ))
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::sum" if hybrid => qwen_call("sum", &self.args),
            "aten::sum" => match self.args.as_slice() {
                [Tensor(shape, kind, _), ..] => Some(TorchCallInfo::Sum(TensorInfo {
                    shape: shape.clone(), dtype: *kind,
                })),
                _ => None,
            },
            "aten::softmax" => match self.args.as_slice() {
                [Tensor(shape, kind, _), Int(dim), ..] => Some(TorchCallInfo::Softmax(
                    TensorInfo {
                        shape: shape.clone(),
                        dtype: *kind,
                    },
                    *dim,
                )),
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::_softmax_backward_data" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _), Int(dim), ..] => {
                    Some(TorchCallInfo::SoftmaxBackward(
                        TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                        *dim,
                    ))
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::zeros_like" => match self.args.as_slice() {
                [Tensor(shape, kind, _), ..] => Some(TorchCallInfo::ZerosLike(TensorInfo {
                    shape: shape.clone(),
                    dtype: *kind,
                })),
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::to" if hybrid => qwen_call("to", &self.args),
            "aten::to" => match self.args.as_slice() {
                [Tensor(shape, kind, _), Int(dtype_idx), Bool(_), Bool(_)] => {
                    Some(TorchCallInfo::ConvDType(
                        TensorInfo {
                            shape: shape.clone(),
                            dtype: *kind,
                        },
                        tch::Kind::from_c_int(*dtype_idx as _).unwrap(),
                    ))
                }
                _ => {
                    // this is ok, only consider conv-dtype-only calls
                    log::debug!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::scaled_dot_product_attention" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _), Tensor(shape3, kind3, _), Tensor(mask_shape, mask_kind, _), Double(_), Bool(causal), Double(_), Bool(gqa)]
                | [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _), Tensor(shape3, kind3, _), Tensor(mask_shape, mask_kind, _), Double(_), Bool(causal), Bool(gqa)] => {
                    Some(TorchCallInfo::SDPA {
                        q: TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        k: TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                        v: TensorInfo {
                            shape: shape3.clone(),
                            dtype: *kind3,
                        },
                        mask: Some(TensorInfo {
                            shape: mask_shape.clone(),
                            dtype: *mask_kind,
                        }),
                        causal: *causal,
                        gqa: *gqa,
                    })
                }
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _), Tensor(shape3, kind3, _), Double(_), Bool(causal), Double(_), Bool(gqa)]
                | [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _), Tensor(shape3, kind3, _), Double(_), Bool(causal), Bool(gqa)] => {
                    Some(TorchCallInfo::SDPA {
                        q: TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        k: TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                        v: TensorInfo {
                            shape: shape3.clone(),
                            dtype: *kind3,
                        },
                        mask: None,
                        causal: *causal,
                        gqa: *gqa,
                    })
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::_scaled_dot_product_flash_attention_backward" => match self.args.as_slice() {
                [Tensor(shape1, kind1, _), Tensor(shape2, kind2, _), Tensor(shape3, kind3, _), Tensor(shape4, kind4, _), Tensor(shape5, kind5, _), Tensor(shape6, kind6, _), Int(max_q), Int(max_k), _, Bool(causal), _, _, _] => {
                    Some(TorchCallInfo::SDPABackward {
                        grad: TensorInfo {
                            shape: shape1.clone(),
                            dtype: *kind1,
                        },
                        q: TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                        k: TensorInfo {
                            shape: shape3.clone(),
                            dtype: *kind3,
                        },
                        v: TensorInfo {
                            shape: shape4.clone(),
                            dtype: *kind4,
                        },
                        out: TensorInfo {
                            shape: shape5.clone(),
                            dtype: *kind5,
                        },
                        logsumexp: TensorInfo {
                            shape: shape6.clone(),
                            dtype: *kind6,
                        },
                        max_q: *max_q,
                        max_k: *max_k,
                        causal: *causal,
                    })
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::_scaled_dot_product_efficient_attention_backward" => match self.args.as_slice() {
                [Tensor(_, _, _), Tensor(shape2, kind2, _), Tensor(shape3, kind3, _), Tensor(shape4, kind4, _), bias_arg, Tensor(_, _, _), Tensor(_, _, _), _, _, Double(_), List(_), Bool(causal), ..] => {
                    Some(TorchCallInfo::SDPAEfficientBackward {
                        q: TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                        k: TensorInfo {
                            shape: shape3.clone(),
                            dtype: *kind3,
                        },
                        v: TensorInfo {
                            shape: shape4.clone(),
                            dtype: *kind4,
                        },
                        bias: match bias_arg {
                            Tensor(bias_shape, bias_kind, _) => Some(TensorInfo {
                                shape: bias_shape.clone(),
                                dtype: *bias_kind,
                            }),
                            _ => None,
                        },
                        causal: *causal,
                    })
                }
                [Tensor(_, _, _), Tensor(shape2, kind2, _), Tensor(shape3, kind3, _), Tensor(shape4, kind4, _), Tensor(_, _, _), Tensor(_, _, _), _, _, Double(_), List(_), Bool(causal), ..] => {
                    Some(TorchCallInfo::SDPAEfficientBackward {
                        q: TensorInfo {
                            shape: shape2.clone(),
                            dtype: *kind2,
                        },
                        k: TensorInfo {
                            shape: shape3.clone(),
                            dtype: *kind3,
                        },
                        v: TensorInfo {
                            shape: shape4.clone(),
                            dtype: *kind4,
                        },
                        bias: None,
                        causal: *causal,
                    })
                }
                _ => {
                    log::warn!("{} args not match: {:?}", self.name, self.args);
                    None
                }
            },
            "aten::conv2d" => match self.args.as_slice() {
                [Tensor(input_shape, input_kind, _), Tensor(weight_shape, weight_kind, _), List(stride), List(padding), List(dilation), Int(groups)] => {
                    match stride.as_slice() {
                        [Int(s1), Int(s2)] => match padding.as_slice() {
                            [Int(p1), Int(p2)] => match dilation.as_slice() {
                                [Int(d1), Int(d2)] => Some(TorchCallInfo::Conv2d {
                                    input: TensorInfo {
                                        shape: input_shape.clone(),
                                        dtype: *input_kind,
                                    },
                                    weight: TensorInfo {
                                        shape: weight_shape.clone(),
                                        dtype: *weight_kind,
                                    },
                                    bias: None,
                                    stride: [*s1, *s2],
                                    padding: [*p1, *p2],
                                    dilation: [*d1, *d2],
                                    groups: *groups,
                                }),
                                _ => None,
                            },
                            _ => None,
                        },
                        _ => None,
                    }
                }
                [Tensor(input_shape, input_kind, _), Tensor(weight_shape, weight_kind, _), Tensor(bias_shape, bias_kind, _), List(stride), List(padding), List(dilation), Int(groups)] => {
                    match stride.as_slice() {
                        [Int(s1), Int(s2)] => match padding.as_slice() {
                            [Int(p1), Int(p2)] => match dilation.as_slice() {
                                [Int(d1), Int(d2)] => Some(TorchCallInfo::Conv2d {
                                    input: TensorInfo {
                                        shape: input_shape.clone(),
                                        dtype: *input_kind,
                                    },
                                    weight: TensorInfo {
                                        shape: weight_shape.clone(),
                                        dtype: *weight_kind,
                                    },
                                    bias: Some(TensorInfo {
                                        shape: bias_shape.clone(),
                                        dtype: *bias_kind,
                                    }),
                                    stride: [*s1, *s2],
                                    padding: [*p1, *p2],
                                    dilation: [*d1, *d2],
                                    groups: *groups,
                                }),
                                _ => None,
                            },
                            _ => None,
                        },
                        _ => None,
                    }
                }
                _ => None,
            },
            "aten::_slow_conv2d_backward" => match self.args.as_slice() {
                [Tensor(grad_output_shape, grad_output_kind, _), Tensor(input_shape, input_kind, _), Tensor(weight_shape, weight_kind, _), List(kernel), List(stride), List(padding), List(_)] => {
                    match kernel.as_slice() {
                        [Int(k1), Int(k2)] => match stride.as_slice() {
                            [Int(s1), Int(s2)] => match padding.as_slice() {
                                [Int(p1), Int(p2)] => Some(TorchCallInfo::Conv2dBackward {
                                    grad_output: TensorInfo {
                                        shape: grad_output_shape.clone(),
                                        dtype: *grad_output_kind,
                                    },
                                    input: TensorInfo {
                                        shape: input_shape.clone(),
                                        dtype: *input_kind,
                                    },
                                    weight: TensorInfo {
                                        shape: weight_shape.clone(),
                                        dtype: *weight_kind,
                                    },
                                    kernel: [*k1, *k2],
                                    stride: [*s1, *s2],
                                    padding: [*p1, *p2],
                                }),
                                _ => None,
                            },
                            _ => None,
                        },
                        _ => None,
                    }
                }
                _ => None,
            },
            _ => { recognized = false; None },
        };
        (call_info.map(|info| TorchCall {
            time: self.curr_time,
            id: ResponseId {
                host: HostId {
                    hostname: self.hostname,
                    pid: self.pid,
                },
                tid: self.tid,
            },
            stream,
            info,
        }), recognized)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorInfo {
    pub shape: Vec<i64>,
    pub dtype: tch::Kind,
}

#[derive(strum::Display, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TorchCallInfo {
    MM(TensorInfo, TensorInfo),
    MatMul(TensorInfo, TensorInfo),
    Linear(TensorInfo, TensorInfo, Option<TensorInfo>),
    BMM(TensorInfo, TensorInfo),
    AddMM(TensorInfo, TensorInfo, TensorInfo),
    BAddBMM(TensorInfo, TensorInfo, TensorInfo),
    Mul(TensorInfo, TensorInfo),
    MulScalar(TensorInfo),
    Mul_(TensorInfo, TensorInfo),
    MulScalar_(TensorInfo),
    ForeachMul_(Vec<TensorInfo>, Vec<TensorInfo>),
    ForeachMulScalar_(Vec<TensorInfo>),
    Add(TensorInfo, TensorInfo),
    Add_(TensorInfo, TensorInfo),
    Div(TensorInfo, TensorInfo),
    DivScalar(TensorInfo),
    Pow(TensorInfo),
    AddCMul_(TensorInfo, TensorInfo, TensorInfo),
    AddCDiv_(TensorInfo, TensorInfo, TensorInfo),
    ForeachAddCMul_(Vec<TensorInfo>, Vec<TensorInfo>, Vec<TensorInfo>),
    ForeachAddCDiv_(Vec<TensorInfo>, Vec<TensorInfo>, Vec<TensorInfo>),
    MaskedFill(TensorInfo, TensorInfo),
    MaskedFill_(TensorInfo, TensorInfo),
    Dropout(TensorInfo, i64, bool),
    NativeDropout(TensorInfo, i64, bool),
    NativeDropoutBackward(TensorInfo, TensorInfo, i64),
    FusedDropout(TensorInfo, i64),
    Where(TensorInfo, TensorInfo, TensorInfo),
    WhereScalar(TensorInfo, TensorInfo),
    Gelu(TensorInfo),
    GeluBackward(TensorInfo, TensorInfo),
    Sum(TensorInfo),
    Sqrt(TensorInfo),
    Softmax(TensorInfo, i64),
    SoftmaxBackward(TensorInfo, TensorInfo, i64),
    ZerosLike(TensorInfo),
    ConvDType(TensorInfo, tch::Kind),
    SDPA {
        q: TensorInfo,
        k: TensorInfo,
        v: TensorInfo,
        // Omit when None so the serialized perf-db key matches the pre-mask
        // spelling and re-records don't churn (perf_db.rs is the only serializer).
        #[serde(skip_serializing_if = "Option::is_none")]
        mask: Option<TensorInfo>,
        causal: bool,
        gqa: bool,
    },
    SDPAEfficientBackward {
        q: TensorInfo,
        k: TensorInfo,
        v: TensorInfo,
        #[serde(skip_serializing_if = "Option::is_none")]
        bias: Option<TensorInfo>,
        causal: bool,
    },
    SDPABackward {
        grad: TensorInfo,
        q: TensorInfo,
        k: TensorInfo,
        v: TensorInfo,
        out: TensorInfo,
        logsumexp: TensorInfo,
        max_q: i64,
        max_k: i64,
        causal: bool,
    },
    Conv2d {
        input: TensorInfo,
        weight: TensorInfo,
        #[serde(skip_serializing_if = "Option::is_none")]
        bias: Option<TensorInfo>,
        stride: [i64; 2],
        padding: [i64; 2],
        dilation: [i64; 2],
        groups: i64,
    },
    Conv2dBackward {
        grad_output: TensorInfo,
        input: TensorInfo,
        weight: TensorInfo,
        kernel: [i64; 2],
        stride: [i64; 2],
        padding: [i64; 2],
    },
    QwenOp {
        op: String, inputs: Vec<TensorInfo>, scalars: Vec<u64>, dims: Vec<i64>,
        keepdim: bool, dtype: Option<tch::Kind>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TorchCall {
    pub time: i64,
    pub id: ResponseId,
    pub stream: CudaStream,
    pub info: TorchCallInfo,
}

#[cfg(test)]
mod hybrid_tests {
    use super::*;

    fn tensor() -> TorchValue { TorchValue::Tensor(vec![2, 3], tch::Kind::Float, tch::Device::Cuda(0)) }
    fn message(name: &str, args: Vec<TorchValue>) -> TorchCallMsg {
        TorchCallMsg { pid: 1, tid: 1, hostname: "host".into(), stream: Some((0, 0)), curr_time: 0, name: name.into(), args }
    }

    #[test]
    fn qwen_mappings_are_opt_in_and_power_two_retains_legacy_key() {
        assert!(message("aten::silu", vec![tensor()]).into_call(0).is_none());
        assert!(matches!(message("aten::silu", vec![tensor()]).into_hybrid_call(0).unwrap().info,
            TorchCallInfo::QwenOp { op, .. } if op == "silu"));
        for exponent in [2, 3] {
            assert!(matches!(message("aten::pow", vec![tensor(), TorchValue::Int(exponent)]).into_call(0).unwrap().info,
                TorchCallInfo::Pow(_)));
        }
        assert!(matches!(message("aten::pow", vec![tensor(), TorchValue::Int(2)]).into_hybrid_call(0).unwrap().info,
            TorchCallInfo::Pow(_)));
        assert!(matches!(message("aten::pow", vec![tensor(), TorchValue::Int(3)]).into_hybrid_call(0).unwrap().info,
            TorchCallInfo::QwenOp { scalars, .. } if scalars == vec![3f64.to_bits()]));
    }

    #[test]
    fn hybrid_reductions_keep_dimensions_and_device_only_calls_are_visible() {
        let args = || vec![tensor(), TorchValue::List(vec![TorchValue::Int(-1)]), TorchValue::Bool(true)];
        assert!(matches!(message("aten::sum", args()).into_call(0).unwrap().info, TorchCallInfo::Sum(_)));
        assert!(matches!(message("aten::sum", args()).into_hybrid_call(0).unwrap().info,
            TorchCallInfo::QwenOp { dims, keepdim: true, .. } if dims == vec![1]));
        let creation = message("aten::zeros", vec![TorchValue::Device(tch::Device::Cuda(2))]);
        assert_eq!(creation.gpu_index(), None);
        assert_eq!(creation.cuda_argument_index(), Some(2));
        let mut current = message("aten::zeros", vec![TorchValue::CurrentCudaDevice]);
        current.stream = Some((3, 7));
        assert_eq!(current.gpu_index(), None);
        assert_eq!(current.cuda_argument_index(), Some(3));
        assert!(message("aten::silu", vec![TorchValue::Int(1)]).into_hybrid_call(0).is_none());
    }
}
