use serde::{Serialize, Deserialize};
use num_complex::Complex64;
use std::fs::File;
use std::io::{Write, Read};
use chrono::Utc;
use sha2::{Sha256, Digest};

/// .qmem Dosya FormatII YapIIsII
/// Bu yapII diske bayt-bayt (binary) olarak yazIIlIIr.
#[derive(Serialize, Deserialize, Debug)]
pub struct QMemFile {
    pub magic: String,       // Dosya imzasII: "SODALS_MEM"
    pub version: u8,         // Format versiyonu: 1
    pub timestamp: i64,      // OluIIturulma zamanII (Unix)
    pub tags: Vec<String>,   // IIrn: ["sohbet", "IInemli"]
    pub data: Vec<Complex64> // Epizodik HafIIza (Spektrum)
}

impl QMemFile {
    /// Yeni bir hafIIza dosyasII oluIIturur
    pub fn new(spectrum: Vec<Complex64>, tags: Vec<String>) -> Self {
        QMemFile {
            magic: "SODALS_MEM".to_string(),
            version: 1,
            timestamp: Utc::now().timestamp(),
            tags,
            data: spectrum,
        }
    }

    /// Diske Asenkron Kaydet (.qmem olarak)
    /// SHA256 ve bloklu Disk IO iIIlemlerini tokio blocking pool'a atarak
    /// ana tokio (main) event loop'unu kilitlemez (Hot-Path izolasyonu).
    pub async fn save_async(self, filename: String) -> std::io::Result<()> {
        tokio::task::spawn_blocking(move || {
            let mut file = File::create(&filename)?;
            
            // Veriyi sIIkIIIItIIrIIlmIIII binary formata IIevir
            let encoded: Vec<u8> = bincode::serialize(&self)
                .map_err(std::io::Error::other)?;
            
            // Checksum Hesapla (SHA-256)
            let mut hasher = Sha256::new();
            hasher.update(&encoded);
            let hash = hasher.finalize();

            // Veri + Checksum yaz
            file.write_all(&encoded)?;
            file.write_all(&hash)?;
            println!("IIII HafIIza mIIhIIrlendi: {} ({} bayt + checksum)", filename, encoded.len());
            Ok(())
        })
        .await
        .map_err(std::io::Error::other)?
    }

    /// Diskten Oku
    pub fn load(filename: &str) -> std::io::Result<Self> {
        let mut file = File::open(filename)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        if buffer.len() < 32 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Dosya IIok kIIsa (Checksum yok)"));
        }

        let split_idx = buffer.len() - 32;
        let (data, stored_hash) = buffer.split_at(split_idx);

        // Checksum DoIIrulama
        let mut hasher = Sha256::new();
        hasher.update(data);
        if hasher.finalize().as_slice() != stored_hash {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "QMem file corrupted"));
        }

        let decoded: QMemFile = bincode::deserialize(data)
            .map_err(std::io::Error::other)?;

        if decoded.magic != "SODALS_MEM" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "GeIIersiz SODALS dosyasII"));
        }

        Ok(decoded)
    }
}













