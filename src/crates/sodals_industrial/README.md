# SODALS Endüstriyel Zırh ve Haberleşme Köprüsü

`sodals_industrial` crate'i, SODALS B2B yapay zeka sisteminin **karma kritiklikli (mixed-criticality)** endüstriyel katmanını sunar.

## Mimari Özeti

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  RT Çekirdek (Core 0-1)          │  Non-RT Çekirdek (Core 2+) + GPU        │
│  ─────────────────────────────── │  ─────────────────────────────────────  │
│  ┌─────────────┐    ┌──────────┐ │  ┌──────────────┐    ┌──────────────┐  │
│  │ OPC UA      │───▶│ ZeroMQ   │ │  │ ZeroMQ SUB   │───▶│ Qwen 8B      │  │
│  │ Client      │    │ PUB/REP  │ │  │              │    │ (Python)     │  │
│  │ (10ms tick) │    │ IPC      │ │  │              │    │              │  │
│  └─────────────┘    └──────────┘ │  └──────────────┘    └──────────────┘  │
│         ▲                          │           │                │          │
│         │                          │           ▼                ▼          │
│  ┌─────────────┐                   │  ┌──────────────────────────────────┐  │
│  │ PLC         │◀──────────────────│  │ SentinelGuard (Rust + PyO3)      │  │
│  │ Siemens/AB  │  Fail-Safe Komut  │  │ • Frekans imza kontrolü        │  │
│  └─────────────┘                   │  │ • Fiziksel limit validasyonu   │  │
│                                    │  └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Modüller

| Dosya | Görev | Karmaşıklık |
|-------|-------|-------------|
| `lib.rs` | Sabit boyutlu veri yapıları (`SensorPayload`, `RtRingBuffer`) | O(1) push/pop |
| `opcua_rt_client.rs` | OPC UA deterministik polling (10ms) | O(1) per tick |
| `zmq_bridge.rs` | ZeroMQ IPC PUB/SUB + REQ/REP | O(1) send/recv |
| `sentinel_guard.rs` | Euler/frekans imza + PyO3 köprüsü | O(128)=O(1) |
| `fail_safe.rs` | Güvenli durum makinesi + PLC komutları | O(1) state machine |
| `qmem_format.rs` | `.qmem` binary format parser/serializer | O(1) header validate |

## Derleme

```bash
# RT OPC UA istemcisi (release, LTO, panic=abort)
cargo build --release -p sodals_industrial --bin sodals_rt_opcua

# AI Orchestrator
cargo build --release -p sodals_industrial --bin sodals_ai_orchestrator

# Sentinel Python modülü (PyO3)
cargo build --release -p sodals_industrial --features python
maturin develop --features python  # Python sanal ortamında
```

## Çalıştırma

```bash
# 1. OS izolasyonu (bir kez)
sudo bash scripts/rt_isolation_setup.sh
sudo reboot

# 2. RT servisi (systemd)
sudo cp scripts/sodals_rt.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sodals_rt

# 3. AI servisi
sudo cp scripts/sodals_ai.service /etc/systemd/system/
sudo systemctl enable --now sodals_ai
```

## Fail-Safe Davranışları

| Senaryo | Eylem | PLC Komutu |
|---------|-------|------------|
| OPC UA kopması | `SystemState::FailSafe` | `STOP_MOTOR` (0x01) |
| ZeroMQ HWM dolması | Mesaj düşürülür, loglanır | — |
| Sentinel reddi | `EMERGENCY_STOP` (0xFF) | Tüm aktüatörler OFF |
| RT saat kayması (>12ms) | `ClockDrift` logu | `STOP_MOTOR` |
| Kernel panic | **YASAK** — `panic=abort` ile proses ölür, systemd yeniden başlatır |

## Bellek Güvenliği

- **Hot-path alloc yok**: Tüm sensör mesajları `SensorPayload::SERIALIZED_SIZE` (24 byte) sabit boyutludur.
- **Ring buffer**: `RtRingBuffer<T, N>` lock-free SPSC; init-time tek alloc.
- **ZeroMQ**: `SNDHWM/RCVHWM` bounded; `NOBLOCK` ile RT thread asla bloklanmaz.
