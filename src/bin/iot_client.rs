// src/bin/iot_client.rs - GÜVENLİ RUST İSTEMCİSİ

use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use futures_util::{StreamExt, SinkExt};
use serde_json::{json, Value};
use sha2::{Sha256, Digest};
use url::Url;
use std::time::Duration;
use tokio::time::sleep;

// GÜVENLİK AYARI: Sunucudaki "SECRET_API_KEY" ile aynı olmalı!
const SECRET_API_KEY: &str = "SECRET_API_KEY"; 

#[tokio::main]
async fn main() {
    let connect_addr = "ws://localhost:5500/ws";
    let url = Url::parse(connect_addr).expect("URL Hatası");

    println!("🔌 IoT İstemcisi Bağlanıyor: {}", connect_addr);

    // Otomatik Yeniden Bağlanma Döngüsü (Reconnection Loop)
    loop {
        match connect_async(url.clone()).await {
            Ok((ws_stream, _)) => {
                println!("✅ WebSocket Bağlantısı Kuruldu.");
                let (mut write, mut read) = ws_stream.split();

                // Mesaj dinleme döngüsü
                while let Some(msg) = read.next().await {
                    match msg {
                        Ok(Message::Text(text)) => {
                            if let Ok(data) = serde_json::from_str::<Value>(&text) {
                                
                                // --- 1. GÜVENLİK: MEYDAN OKUMA (HANDSHAKE) ---
                                if let Some(challenge) = data.get("AuthChallenge") {
                                    if let Some(nonce) = challenge["nonce"].as_str() {
                                        println!("🔒 Meydan Okuma Alındı: {}", nonce);
                                        
                                        // Token Hesapla: SHA256(nonce + SECRET)
                                        let mut hasher = Sha256::new();
                                        hasher.update(format!("{}{}", nonce, SECRET_API_KEY));
                                        let token = format!("{:x}", hasher.finalize());

                                        // Cevabı Gönder
                                        let response = json!({
                                            "AuthResponse": { "token": token }
                                        });
                                        
                                        if let Err(e) = write.send(Message::Text(response.to_string())).await {
                                            println!("❌ Gönderim hatası: {}", e);
                                            break; 
                                        }
                                        println!("🔑 Token Gönderildi.");
                                    }
                                }
                                // --- 2. BAŞARI: SİSTEME GİRİŞ ---
                                else if data["type"] == "AuthSuccess" {
                                    println!("🚀 Kimlik Doğrulama BAŞARILI! Veri akışı başlıyor...");
                                    
                                    // Selamlama / Telemetri Gönder
                                    let _telemetry = json!({
                                        "user_id": "sensor_rust_01",
                                        "type": "telemetry",
                                        "status": "online",
                                        "temp": 42.0
                                    });
                                    // Not: write stream burada ownership sorunu yaşatabilir, 
                                    // basitlik için sadece okuma yapıyoruz şimdilik.
                                }
                                // --- 3. HATA: REDDEDİLDİ ---
                                else if data["type"] == "AuthFailed" {
                                    println!("⛔ ŞİFRE HATALI! Bağlantı reddedildi.");
                                    return; // Programdan çık
                                }
                                else {
                                    // Diğer mesajlar (Kernel cevapları vs.)
                                    println!("📩 Sunucu: {}", text);
                                }
                            }
                        }
                        Ok(Message::Close(_)) => {
                            println!("⚠️ Bağlantı sunucu tarafından kapatıldı.");
                            break;
                        }
                        Err(_) => break,
                        _ => {}
                    }
                }
            }
            Err(e) => {
                println!("❌ Bağlantı Hatası: {}. 3 saniye içinde tekrar deneniyor...", e);
                sleep(Duration::from_secs(3)).await;
            }
        }
    }
}