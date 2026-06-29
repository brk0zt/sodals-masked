//! ADIM 2: Rust OPC UA İstemcisi (Gerçek Zamanlı Çekirdek)
//!
//! Karmaşıklık analizi:
//! - Ana döngü: O(1), her iterasyon sabit iş yükü.
//! - Bellek: Init-time tek alloc; hot-path'te alloc YASAK.
//! - Zamanlama: `tokio::time::interval` ile 10ms deterministik tick.
//!
//! NOT: `opcua` feature aktif değilse bu modül stub implementasyon sunar.
//! Production Linux'ta: `cargo build --features opcua`

use crate::{fail_safe, SensorPayload, IndustrialError, RtRingBuffer};
use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::{interval, MissedTickBehavior};

/// OPC UA NodeId tanımları (Siemens S7-1500 / Allen-Bradley uyumlu).
pub struct OpcUaNodeIds {
    pub ns: u16,
    pub temperature: String,
    pub pressure: String,
    pub motor_rpm: String,
    pub status_word: String,
}

impl OpcUaNodeIds {
    pub fn default_siemens() -> Self {
        Self {
            ns: 3,
            temperature: "::\"Program_blocks\".\"Data\".\"Temperature\".value".to_string(),
            pressure: "::\"Program_blocks\".\"Data\".\"Pressure\".value".to_string(),
            motor_rpm: "::\"Program_blocks\".\"Data\".\"MotorSpeed\".value".to_string(),
            status_word: "::\"Program_blocks\".\"Data\".\"StatusWord\".value".to_string(),
        }
    }

    pub fn default_ab() -> Self {
        Self {
            ns: 2,
            temperature: "ns=2;s=[PLC]Temperature".to_string(),
            pressure: "ns=2;s=[PLC]Pressure".to_string(),
            motor_rpm: "ns=2;s=[PLC]MotorSpeed".to_string(),
            status_word: "ns=2;s=[PLC]StatusWord".to_string(),
        }
    }
}

// =============================================================================
// STUB Implementasyon (varsayılan — Windows geliştirme / test)
// =============================================================================
#[cfg(not(feature = "opcua"))]
pub struct OpcUaRtClient {
    last_success_ns: AtomicU64,
    tx_ring: Arc<std::sync::Mutex<RtRingBuffer<SensorPayload, 1024>>>,
}

#[cfg(not(feature = "opcua"))]
impl OpcUaRtClient {
    pub fn new(
        _node_ids: OpcUaNodeIds,
        tx_ring: Arc<std::sync::Mutex<RtRingBuffer<SensorPayload, 1024>>>,
    ) -> Self {
        Self {
            last_success_ns: AtomicU64::new(0),
            tx_ring,
        }
    }

    pub async fn connect(&mut self, _endpoint_url: &str) -> Result<(), IndustrialError> {
        tracing::warn!("OPC UA stub: connect çağrıldı (opcua feature aktif değil)");
        fail_safe::transition_state(fail_safe::SystemState::Running);
        Ok(())
    }

    pub async fn run_deterministic_loop(&mut self, tick_us: u64) -> Result<(), IndustrialError> {
        let mut ticker = interval(Duration::from_micros(tick_us));
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
        loop {
            let _ = ticker.tick().await;
            let now = Instant::now().elapsed().as_nanos() as u64;
            let payload = SensorPayload {
                timestamp_ns: now,
                temperature: 25.0,
                pressure: 1.0,
                motor_rpm: 1500.0,
                status_mask: 0,
                fault_code: 0,
                _pad: [0; 2],
            };
            if let Ok(ring) = self.tx_ring.lock() {
                let _ = ring.push(payload);
            }
            self.last_success_ns.store(now, Ordering::Relaxed);
        }
    }
}

// =============================================================================
// GERÇEK Implementasyon (opcua feature aktifse — Linux production)
// =============================================================================
#[cfg(feature = "opcua")]
use opcua::client::prelude::*;

#[cfg(feature = "opcua")]
pub struct OpcUaRtClient {
    client: Client,
    session: Option<Session>,
    nodes: OpcUaNodeIds,
    last_success_ns: AtomicU64,
    read_request: ReadRequest,
    tx_ring: Arc<std::sync::Mutex<RtRingBuffer<SensorPayload, 1024>>>,
}

#[cfg(feature = "opcua")]
impl OpcUaRtClient {
    pub fn new(
        node_ids: OpcUaNodeIds,
        tx_ring: Arc<std::sync::Mutex<RtRingBuffer<SensorPayload, 1024>>>,
    ) -> Self {
        let client = ClientBuilder::new()
            .application_name("SODALS_RT_OPC")
            .application_uri("urn:sodals:rt:opcua")
            .create_client()
            .expect("ClientBuilder hatası");

        let read_request = ReadRequest {
            nodes_to_read: Some(vec![
                ReadValueId {
                    node_id: NodeId::new(node_ids.ns, node_ids.temperature.clone()),
                    attribute_id: AttributeId::Value as u32,
                    index_range: UAString::null(),
                    data_encoding: QualifiedName::null(),
                },
                ReadValueId {
                    node_id: NodeId::new(node_ids.ns, node_ids.pressure.clone()),
                    attribute_id: AttributeId::Value as u32,
                    index_range: UAString::null(),
                    data_encoding: QualifiedName::null(),
                },
                ReadValueId {
                    node_id: NodeId::new(node_ids.ns, node_ids.motor_rpm.clone()),
                    attribute_id: AttributeId::Value as u32,
                    index_range: UAString::null(),
                    data_encoding: QualifiedName::null(),
                },
                ReadValueId {
                    node_id: NodeId::new(node_ids.ns, node_ids.status_word.clone()),
                    attribute_id: AttributeId::Value as u32,
                    index_range: UAString::null(),
                    data_encoding: QualifiedName::null(),
                },
            ]),
            timestamps_to_return: TimestampsToReturn::Both,
            max_age: 0.0,
        };

        Self {
            client,
            session: None,
            nodes: node_ids,
            last_success_ns: AtomicU64::new(0),
            read_request,
            tx_ring,
        }
    }

    pub async fn connect(&mut self, endpoint_url: &str) -> Result<(), IndustrialError> {
        let session = self
            .client
            .connect_to_endpoint(
                endpoint_url,
                IdentityToken::Anonymous,
            )
            .await
            .map_err(|e| {
                tracing::error!("OPC UA bağlantı hatası: {:?}", e);
                fail_safe::transition_state(fail_safe::SystemState::FailSafe);
                IndustrialError::OpcUaDisconnect
            })?;
        self.session = Some(session);
        fail_safe::transition_state(fail_safe::SystemState::Running);
        Ok(())
    }

    async fn poll_once(&mut self) -> Result<(), IndustrialError> {
        let session = self
            .session
            .as_mut()
            .ok_or(IndustrialError::OpcUaDisconnect)?;

        let response = session
            .read(
                &self.read_request,
            )
            .map_err(|_| IndustrialError::OpcUaDisconnect)?;

        // Yanıtı SensorPayload'a dönüştür
        let mut payload = SensorPayload::default();
        if let Some(results) = response.results {
            if let Some(dv) = results.get(0) {
                if let Some(v) = &dv.value {
                    if let Ok(t) = v.clone().try_into() {
                        payload.temperature = t;
                    }
                }
            }
            if let Some(dv) = results.get(1) {
                if let Some(v) = &dv.value {
                    if let Ok(p) = v.clone().try_into() {
                        payload.pressure = p;
                    }
                }
            }
            if let Some(dv) = results.get(2) {
                if let Some(v) = &dv.value {
                    if let Ok(r) = v.clone().try_into() {
                        payload.motor_rpm = r;
                    }
                }
            }
            if let Some(dv) = results.get(3) {
                if let Some(v) = &dv.value {
                    if let Ok(s) = v.clone().try_into() {
                        payload.status_mask = s;
                    }
                }
            }
        }

        payload.timestamp_ns = Instant::now().elapsed().as_nanos() as u64;

        if let Ok(mut ring) = self.tx_ring.lock() {
            ring.push(payload)?;
        }

        self.last_success_ns.store(payload.timestamp_ns, Ordering::Relaxed);
        Ok(())
    }

    pub async fn run_deterministic_loop(&mut self, tick_us: u64) -> Result<(), IndustrialError> {
        let mut ticker = interval(Duration::from_micros(tick_us));
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

        loop {
            let start = Instant::now();
            let _ = ticker.tick().await;

            if let Err(e) = self.poll_once().await {
                tracing::error!("OPC UA polling hatası: {:?}", e);
                fail_safe::transition_state(fail_safe::SystemState::Degraded);
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }

            let elapsed = start.elapsed();
            if elapsed > Duration::from_millis(12) {
                tracing::warn!("RT clock drift: {:?} (limit 12ms)", elapsed);
                fail_safe::transition_state(fail_safe::SystemState::Degraded);
            }
        }
    }
}
