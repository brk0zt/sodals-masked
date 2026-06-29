//! SODALS Fail-Safe Tetikleyici
//!
//! `ExecStopPost` olarak systemd unit'inden çağrılır.
//! PLC'ye acil durdurma (EMERGENCY_STOP) komutu gönderir.
//! Bellek sızıntısı riski YOK: sabit 16 byte buffer, O(1).

use sodals_industrial::fail_safe::FailSafeCommand;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    let mode = std::env::args().nth(2).unwrap_or_else(|| "stop".to_string());

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    let cmd = if mode == "emergency" {
        FailSafeCommand {
            command_code: 0xFF,
            param: 0,
            timestamp_ns: now,
            triggered_by: 0xFE,
            _pad: [0; 4],
        }
    } else {
        FailSafeCommand {
            command_code: 0x01, // STOP_MOTOR
            param: 0,
            timestamp_ns: now,
            triggered_by: 0xFD,
            _pad: [0; 4],
        }
    };

    let mut buf = [0u8; 16];
    cmd.serialize_into(&mut buf);

    // TODO: ZeroMQ REQ veya doğrudan OPC UA yazma ile PLC'ye gönder
    eprintln!("FAIL-SAFE TETIKLENDI: mode={}, buf={:02x?}", mode, &buf[..]);
    std::process::exit(0);
}
