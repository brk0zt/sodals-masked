//! SODALS RT OPC UA İstemcisi — Binary Giriş Noktası
//!
//! Kullanım:
//!   sudo ./sodals_rt_opcua --endpoint opc.tcp://192.168.1.10:4840 --cores 0,1
//!
//! RT İzolasyon:
//!   `isolcpus=0,1` ile bu çekirdekler scheduler'dan çıkarılmış olmalı.
//!   `taskset -c 0,1` ile manuel CPU affinite ayarı yapılır.

use sodals_industrial::{
    fail_safe::{transition_state, SystemState},
    opcua_rt_client::{OpcUaNodeIds, OpcUaRtClient},
    zmq_bridge::{IpcEndpoints, SensorPublisher},
};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tracing::{info, warn};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    tracing_subscriber::fmt::init();

    let endpoint = std::env::var("OPCUA_ENDPOINT")
        .unwrap_or_else(|_| "opc.tcp://192.168.1.10:4840".to_string());

    info!("SODALS RT OPC UA başlatılıyor: endpoint={}", endpoint);

    // --- 1. CPU Affinite (RT Çekirdek 0-1) ---
    set_cpu_affinity(&[0, 1]);

    // --- 2. ZeroMQ Publisher Başlat ---
    let endpoints = IpcEndpoints::default();
    let publisher = Arc::new(
        SensorPublisher::new(&endpoints.sensor_pub, 100)
            .expect("ZeroMQ PUB soketi oluşturulamadı"),
    );

    // --- 3. OPC UA Client Başlat ---
    let nodes = OpcUaNodeIds::default_siemens();
    let mut client = OpcUaRtClient::new(&endpoint, nodes)
        .expect("OPC UA client oluşturulamadı");

    if let Err(e) = client.connect(&endpoint).await {
        warn!("OPC UA ilk bağlantı başarısız: {:?}. Fail-Safe tetiklendi.", e);
        transition_state(SystemState::FailSafe);
        // Yeniden bağlanma döngüsü
        loop {
            tokio::time::sleep(Duration::from_secs(5)).await;
            if client.connect(&endpoint).await.is_ok() {
                break;
            }
        }
    }

    info!("OPC UA bağlandı. RT 10ms döngü başlıyor.");

    // --- 4. RT Döngüyü Çalıştır ---
    client.run_rt_loop().await;
}

/// `sched_setaffinity` ile süreci RT çekirdeklerine kilitle.
/// Linux-only. Başarısız olursa loglar ama durmaz.
#[cfg(target_os = "linux")]
fn set_cpu_affinity(cores: &[usize]) {
    use libc::{cpu_set_t, sched_setaffinity, CPU_SET, CPU_ZERO};
    unsafe {
        let mut set: cpu_set_t = std::mem::zeroed();
        CPU_ZERO(&mut set);
        for &c in cores {
            CPU_SET(c, &mut set);
        }
        let pid = libc::getpid();
        let ret = sched_setaffinity(pid, std::mem::size_of::<cpu_set_t>(), &set);
        if ret != 0 {
            warn!("sched_setaffinity başarısız (errno={})", *libc::__errno_location());
        } else {
            info!("CPU affinite ayarlandı: {:?}", cores);
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn set_cpu_affinity(_cores: &[usize]) {
    warn!("CPU affinite bu platformda desteklenmiyor.");
}
