use phantora::args::get_args;
use phantora::simulator;
use phantora::torch_call::{TorchCallJson, TorchCallMsg};

use cuda_call::{capi, CudaCallMsg, HostId, ResponseId};
use serde::{Deserialize, Serialize};
use std::fs;
use std::os::unix::net::UnixDatagram;

fn main() {
    let env = env_logger::Env::new()
        .filter("PHANTORA_LOG")
        .write_style("PHANTORA_LOG_STYLE");
    env_logger::init_from_env(env);

    let _args = get_args();
    main_loop();
}

#[derive(Serialize, Deserialize)]
enum Message {
    Cuda(CudaCallMsg),
    Torch(TorchCallMsg, i32 /* device index */),
    Exit(ResponseId, i64),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExitJson {
    pub pid: u32,
    pub tid: i32,
    pub hostname: String,
    pub cur: i64,
}

fn main_loop() {
    // Create domain socket for Phantora simulator
    let socket_path = capi::simulator_socket_path();
    let _ = fs::remove_file(&socket_path);
    let recv_socket = UnixDatagram::bind(socket_path).unwrap();

    let mut buf = [0u8; 1024 * 1024];
    let netconfig: netsim::config::Config = netsim::config::read_config(&get_args().net_config);
    println!("netconfig: {:#?}", netconfig);
    let cluster = netsim::config::build_cloud(&netconfig);
    let netsim = netsim::simulator::SimulatorBuilder::new()
        .with_setting(netconfig.simulator)
        .cluster(cluster)
        .host_mapping(netconfig.host_mapping)
        .build()
        .expect("Fail to create network simulator");
    let mut simulator = simulator::Simulator::new(netsim);
    let mut coverage = phantora::coverage::Coverage::from_args(get_args()).expect("invalid coverage configuration");
    if let Some(run) = &coverage { run.write(&simulator.coverage_misses()).expect("cannot write ready status"); }

    loop {
        let sz = recv_socket.recv(&mut buf).unwrap();
        if sz == 0 || sz >= buf.len() {
            if let Some(run) = &mut coverage { run.malformed("invalid/truncated datagram".into()); continue; }
            panic!("invalid datagram");
        }
        let tag = buf[sz - 1];
        let bytes = &buf[..sz - 1];
        let parsed: Result<Option<Message>, String> = match tag {
            1 => bincode::deserialize::<CudaCallMsg>(bytes).map(|msg| Some(Message::Cuda(msg))).map_err(|e| e.to_string()),
            2 => serde_json::from_slice::<TorchCallJson>(bytes).map_err(|e| e.to_string())
                .and_then(|raw| std::panic::catch_unwind(|| raw.into_msg()).map_err(|_| "invalid torch value".into()))
                .map(|call| {
                    let device = call.gpu_index().or_else(|| if coverage.is_some() { call.cuda_argument_index() } else { None });
                    device.map(|index| Message::Torch(call, index as i32))
                }),
            3 => serde_json::from_slice::<ExitJson>(bytes).map_err(|e| e.to_string()).map(|exit| Some(Message::Exit(
                ResponseId { host: HostId { hostname: exit.hostname, pid: exit.pid }, tid: exit.tid }, exit.cur))),
            _ => Err(format!("unknown datagram tag {tag}")),
        };
        let message = match parsed {
            Ok(Some(message)) => message,
            Ok(None) => continue,
            Err(error) => {
                if let Some(run) = &mut coverage {
                    run.malformed(format!("malformed trace message: {error}"));
                    if let Err(error) = run.write(&simulator.coverage_misses()) { eprintln!("coverage status failure: {error}"); }
                    continue;
                }
                panic!("{error}");
            }
        };
        let mut report = false;
        match message {
            Message::Cuda(msg) => {
                log::debug!("{:?}", msg);
                if let Some(run) = &mut coverage {
                    match &msg.call {
                        cuda_call::CudaCall::CoverageControl { payload } => { run.control(&msg.id.host, payload); report = true; }
                        cuda_call::CudaCall::GdnCall { key_json, .. } => run.semantic(&msg.id.host, key_json),
                        _ => (),
                    }
                }
                simulator.handle_cuda_call(msg);
            }
            Message::Torch(callmsg, arg_device) => {
                log::debug!("{:?}", callmsg);
                let identity = coverage.as_ref().map(|_| (
                    HostId { hostname: callmsg.hostname.clone(), pid: callmsg.pid },
                    callmsg.name.clone(), format!("{:?}", callmsg.args),
                ));
                let parsed = std::panic::catch_unwind(|| if coverage.is_some() {
                    callmsg.into_hybrid_classified_call(arg_device)
                } else { (callmsg.into_call(arg_device), false) });
                let (call, recognized) = match parsed {
                    Ok(parsed) => parsed,
                    Err(_) => {
                        if let Some(run) = &mut coverage { run.malformed("panic parsing CUDA operator arguments".into()); continue; }
                        panic!("invalid legacy CUDA operator arguments");
                    }
                };
                if let (Some(run), Some((host, name, signature))) = (&mut coverage, identity) {
                    run.torch(&host, &name, &signature, call.is_some(), recognized);
                }
                if let Some(call) = call { simulator.handle_torch_call(call); }
            }
            Message::Exit(host, cur) => {
                log::debug!("{:?} exited at {}", host, cur);
                if let Some(run) = &mut coverage { run.exit(&host.host); report = true; }
                simulator.handle_exit(host, cur);
            }
        }
        simulator.pump_parked_syncs();
        if report {
            if let Some(run) = &coverage {
                if let Err(error) = run.write(&simulator.coverage_misses()) { eprintln!("coverage status failure: {error}"); }
            }
        }
    }
}
