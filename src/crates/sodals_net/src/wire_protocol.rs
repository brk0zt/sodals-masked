// src/wire_protocol.rs
use anyhow::Result;

pub const START_BYTE: u8 = 0xAA;
pub const END_BYTE: u8 = 0x55;

// Mesaj Türleri
pub const MSG_TELEMETRY: u8 = 0x01; // IoT Verisi
pub const MSG_TEXT: u8 = 0x02;      // Düz Metin

// UYARI DÜZELTMESİ: İleride IoT cihazlarına emir göndermek için kullanılacak.
#[allow(dead_code)] 
pub const MSG_COMMAND: u8 = 0x03;   // Emirler

#[derive(Debug)]
pub enum WireMessage {
    Telemetry { 
        temp: f32, 
        // UYARI DÜZELTMESİ: Şu an sadece sıcaklık verisini işliyoruz, 
        // nem verisi protokolde var ama kernel tarafında okunmuyor.
        #[allow(dead_code)] 
        humidity: f32 
    },
    Text(String),
    Unknown,
}

/// Gelen ham byte verisini anlamlı mesaja çevirir (Deserialization)
/// Heap tahsisi yapmaz (Zero-Copy mantığına yakın).
pub fn decode(buffer: &[u8]) -> Result<WireMessage> {
    if buffer.len() < 5 { return Ok(WireMessage::Unknown); } // En az Header + Footer lazım
    if buffer[0] != START_BYTE || buffer[buffer.len()-1] != END_BYTE {
        return Err(anyhow::anyhow!("Bozuk Paket"));
    }

    let msg_type = buffer[1];
    // let len = u16::from_be_bytes([buffer[2], buffer[3]]); // Uzunluk kontrolü eklenebilir

    match msg_type {
        MSG_TELEMETRY => {
            // Payload: [Temp(4 byte)][Hum(4 byte)]
            if buffer.len() < 4 + 4 + 4 + 1 { return Err(anyhow::anyhow!("Eksik Telemetri Verisi")); }
            
            // Rust'ta byte array'den float okuma
            let temp_bytes: [u8; 4] = buffer[4..8].try_into()?;
            let hum_bytes: [u8; 4] = buffer[8..12].try_into()?;
            
            let temp = f32::from_le_bytes(temp_bytes); // Little Endian (Arduino standartı)
            let humidity = f32::from_le_bytes(hum_bytes);
            
            Ok(WireMessage::Telemetry { temp, humidity })
        }
        MSG_TEXT => {
            // Payload: UTF-8 String
            let text_bytes = &buffer[4..buffer.len()-1];
            let text = String::from_utf8(text_bytes.to_vec())?;
            Ok(WireMessage::Text(text))
        }
        _ => Ok(WireMessage::Unknown),
    }
}