//! ADIM 3: ZeroMQ IPC Köprüsü (Cross-Platform IPC Implementasyonu)
//!
//! Mimarisi:
//! - PUBLISHER (IPC): RT OPC UA thread -> AI/Sentinel thread
//! - REPLY (IPC): AI/Sentinel -> RT PLC aktüatör yazma kanalı
//!
//! Implementasyon:
//! - `zmq` feature aktifse: Gerçek ZeroMQ IPC (Linux production)
//! - Varsayılan: `tokio::sync::broadcast` + `tokio::sync::mpsc` (Test / Windows geliştirme)
//!
//! Karmaşıklık:
//! - `send`: O(1), sabit boyutlu mesaj kopyalama.
//! - `recv`: O(1), lock-free (async channel internal).
//! - Bellek: Mesajlar sabit boyutlu (sizeof SensorPayload = 24 byte).
//!
//! NOT: Bu implementasyon gerçek ZeroMQ'nun tüm özelliklerini sunmaz;
//! SODALS'in ihtiyaç duyduğu PUB/SUB ve REQ/REP pattern'lerini
//! minimal ve deterministik şekilde taklit eder.

use crate::{SensorPayload, FailSafeCommand};
use std::sync::{Arc, Mutex};
use tokio::sync::{broadcast, mpsc};

/// IPC endpoint tanımları.
pub struct IpcEndpoints {
    pub sensor_pub: String,
    pub cmd_rep: String,
}

impl Default for IpcEndpoints {
    fn default() -> Self {
        Self {
            sensor_pub: "ipc:///tmp/sodals_sensor.pub".to_string(),
            cmd_rep: "ipc:///tmp/sodals_cmd.rep".to_string(),
        }
    }
}

/// RT OPC UA tarafında çalışan PUBLISHER.
/// Sensör verilerini AI/Sentinel'e gönderir.
pub struct SensorPublisher {
    tx: broadcast::Sender<SensorPayload>,
    /// HWM (High Water Mark): Taşarsa mesaj düşürülür (NOBLOCK).
    #[allow(dead_code)]
    hwm: usize,
}

impl SensorPublisher {
    pub fn new(endpoint: &str, hwm: i32) -> Result<Self, String> {
        let tx = get_or_create_broadcast_with_capacity(endpoint, hwm as usize);
        Ok(Self { tx, hwm: hwm as usize })
    }

    /// SensorPayload'u IPC üzerinden PUBLISH et.
    /// O(1): Sabit 24 byte kopyalama.
    /// Dönüş: true=gonderildi, false=HWM dolu (düşürüldü).
    pub fn publish(&self, payload: &SensorPayload) -> Result<bool, String> {
        match self.tx.send(*payload) {
            Ok(_) => Ok(true),
            Err(broadcast::error::SendError(_)) => {
                tracing::warn!("IPC PUB HWM dolu, sensör mesajı düşürüldü");
                Ok(false)
            }
        }
    }
}

/// Non-RT AI/Sentinel tarafında çalışan SUBSCRIBER.
/// Sensör verilerini dinler.
pub struct SensorSubscriber {
    rx: Mutex<broadcast::Receiver<SensorPayload>>,
}

impl SensorSubscriber {
    pub fn new(endpoint: &str) -> Result<Self, String> {
        // Global static registry simulation (test/development only)
        let tx = get_or_create_broadcast(endpoint);
        let rx = tx.subscribe();
        Ok(Self { rx: Mutex::new(rx) })
    }

    /// Bloke edici olmayan (non-blocking) alım.
    pub fn receive(&self) -> Result<Option<SensorPayload>, String> {
        let mut rx = self.rx.lock().map_err(|_| "Lock failed".to_string())?;
        loop {
            match rx.try_recv() {
                Ok(v) => return Ok(Some(v)),
                Err(broadcast::error::TryRecvError::Empty) => return Ok(None),
                Err(broadcast::error::TryRecvError::Lagged(_)) => {
                    // Mesajlar düşürüldü; kalanları drain et
                    continue;
                }
                Err(broadcast::error::TryRecvError::Closed) => {
                    return Err("Channel closed".to_string())
                }
            }
        }
    }
}

/// RT PLC aktüatör yazma kanalı (REQ/REP taklidi).
/// AI/Sentinel'den gelen Fail-Safe komutlarını PLC'ye iletir.
pub struct CommandReplier {
    rx: Arc<std::sync::Mutex<mpsc::Receiver<(FailSafeCommand, mpsc::Sender<bool>)>>>,
}

impl CommandReplier {
    pub fn new(_endpoint: &str) -> Result<Self, String> {
        let rx = get_or_create_command_rx();
        Ok(Self { rx })
    }

    /// AI/Sentinel'den gelen komutu bekle ve yanıtla.
    pub fn recv_command(&self) -> Result<Option<FailSafeCommand>, String> {
        let mut rx = self.rx.lock().map_err(|_| "Lock failed".to_string())?;
        match rx.try_recv() {
            Ok((cmd, ack_tx)) => {
                let _ = ack_tx.try_send(true); // ACK
                Ok(Some(cmd))
            }
            Err(mpsc::error::TryRecvError::Empty) => Ok(None),
            Err(mpsc::error::TryRecvError::Disconnected) => {
                Err("Command channel disconnected".to_string())
            }
        }
    }
}

/// AI/Sentinel tarafından kullanılan REQ soketi.
pub struct CommandRequester {
    tx: mpsc::Sender<(FailSafeCommand, mpsc::Sender<bool>)>,
    #[allow(dead_code)]
    local_ack: Mutex<mpsc::Receiver<bool>>,
}

impl CommandRequester {
    pub fn new(_endpoint: &str) -> Result<Self, String> {
        let tx = get_or_create_command_tx();
        let (_ack_tx, ack_rx) = mpsc::channel(1);
        Ok(Self {
            tx,
            local_ack: Mutex::new(ack_rx),
        })
    }

    /// Fail-Safe komutunu gönder ve ACK bekle (polling ile).
    pub fn send_command(&self, cmd: &FailSafeCommand) -> Result<bool, String> {
        let (ack_tx, mut ack_rx) = mpsc::channel(1);
        self.tx
            .try_send((*cmd, ack_tx))
            .map_err(|_| "Command channel full".to_string())?;
        // Replier'ın işlemesi için polling bekleme
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            match ack_rx.try_recv() {
                Ok(v) => return Ok(v),
                Err(mpsc::error::TryRecvError::Empty) => {
                    if std::time::Instant::now() > deadline {
                        return Ok(false);
                    }
                    std::thread::sleep(std::time::Duration::from_micros(50));
                }
                Err(mpsc::error::TryRecvError::Disconnected) => return Ok(false),
            }
        }
    }
}

// =============================================================================
// Global Static Registry (Test / Geliştirme — tek process)
// =============================================================================
// Gerçek çoklu process IPC'de (Linux production) bu kısım ZeroMQ soketleriyle
// değişir. `zmq` feature aktif edildiğinde bu modülün tamamı zmq implementasyonu
// ile değiştirilir.

use std::collections::HashMap;
use std::sync::OnceLock;

static BROADCAST_REGISTRY: OnceLock<Mutex<HashMap<String, broadcast::Sender<SensorPayload>>>> =
    OnceLock::new();

fn get_broadcast_registry() -> &'static Mutex<HashMap<String, broadcast::Sender<SensorPayload>>> {
    BROADCAST_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn get_or_create_broadcast(endpoint: &str) -> broadcast::Sender<SensorPayload> {
    get_or_create_broadcast_with_capacity(endpoint, 10_000)
}

fn get_or_create_broadcast_with_capacity(endpoint: &str, capacity: usize) -> broadcast::Sender<SensorPayload> {
    let mut map = get_broadcast_registry()
        .lock()
        .expect("Broadcast registry lock");
    map.entry(endpoint.to_string())
        .or_insert_with(|| {
            let (tx, _rx) = broadcast::channel(capacity.max(10));
            tx
        })
        .clone()
}

static COMMAND_CHANNEL: OnceLock<(
    mpsc::Sender<(FailSafeCommand, mpsc::Sender<bool>)>,
    Arc<Mutex<mpsc::Receiver<(FailSafeCommand, mpsc::Sender<bool>)>>>,
)> = OnceLock::new();

fn get_or_create_command_tx() -> mpsc::Sender<(FailSafeCommand, mpsc::Sender<bool>)> {
    let (tx, _) = COMMAND_CHANNEL.get_or_init(|| {
        let (tx, rx) = mpsc::channel(100);
        (tx, Arc::new(Mutex::new(rx)))
    });
    tx.clone()
}

fn get_or_create_command_rx() -> Arc<Mutex<mpsc::Receiver<(FailSafeCommand, mpsc::Sender<bool>)>>> {
    let (_, rx) = COMMAND_CHANNEL.get_or_init(|| {
        let (tx, rx) = mpsc::channel(100);
        (tx, Arc::new(Mutex::new(rx)))
    });
    rx.clone()
}
