use tokio::sync::mpsc;
use sodals_net::protocol::{SystemMsg, MemoryRecord};
use crate::vault::Vault;
use sodals_common::log_manager::{LogManager, LogLevel};
use rand;

pub struct VectorStoreActor {
    rx: mpsc::Receiver<SystemMsg>,
    kernel_tx: mpsc::Sender<SystemMsg>,
    vault: Vault,
}

impl VectorStoreActor {
    pub fn new(rx: mpsc::Receiver<SystemMsg>, kernel_tx: mpsc::Sender<SystemMsg>) -> Self {
        let logger = LogManager::new(LogLevel::INFO);
        let path = std::path::Path::new("sodals.vault");
        
        println!("[STORE] Vault yükleniyor...");
        let vault = Vault::unlock(path, "dev-password", &logger)
            .expect("Vault açılamadı! (Yetki veya disk hatası)");
        println!("[STORE] Vault başarıyla yüklendi.");

        Self { rx, kernel_tx, vault }
    }

    pub async fn run(mut self) {
        println!("[STORE] Hafıza Motoru Döngüye Girdi.");

        while let Some(msg) = self.rx.recv().await {
            match msg {
                SystemMsg::MemorizeRequest { user_id, text, vector } => {
                    let preview = if text.chars().count() > 30 { 
                        format!("{}...", text.chars().take(30).collect::<String>()) 
                    } else { 
                        text.clone() 
                    };
                    println!("[STORE] Kayıt İsteği Alındı ({}) -> {}", user_id, preview);
                    
                    let id = rand::random::<u64>();
                    if let Err(e) = self.vault.add_memory(id, &user_id, &text, vector) {
                        eprintln!("[STORE] ❌ Kayıt Hatası: {}", e);
                    } else {
                        println!("[STORE] ✅ Başarıyla Kaydedildi (ID: {})", id);
                    }
                }

                SystemMsg::VectorSearchRequest { user_id, query_vector, top_k, request_id } => {
                    // 1. Log: İstek geldi
                    println!("[STORE] 🔍 Arama Başlıyor (ReqID: {} - User: {})", request_id, user_id);
                    
                    // 2. İşlem: Vault Araması (Burada takılıyor olabilir)
                    let matches = self.vault.search(&query_vector, top_k);
                    
                    // 3. Log: Arama bitti
                    println!("[STORE] 🔍 Arama Tamamlandı. {} eşleşme bulundu.", matches.len());

                    let mut results = Vec::new();
                    for (id, _score) in matches {
                        if let Ok(text) = self.vault.retrieve_content(id) {
                            // Skor bilgisini de loglayalım (Debug için)
                            // println!("[STORE] Bulunan: {} (Skor: {:.4})", id, score);
                            results.push(text);
                        }
                    }

                    // 4. Log: Kernel'a gönderiliyor
                    println!("[STORE] Sonuçlar Kernel'a gönderiliyor... ({} adet)", results.len());

                    if let Err(e) = self.kernel_tx.send(SystemMsg::VectorSearchResult { request_id, results }).await {
                         eprintln!("[STORE] ❌ KANAL HATASI: Kernel mesajı alamadı! Hata: {}", e);
                    } else {
                         println!("[STORE] ✅ Sonuçlar Kernel kanalına bırakıldı.");
                    }
                }

                SystemMsg::GetAllMemoriesRequest { req_id, user_id: _ } => {
                    let _ = self.kernel_tx.send(SystemMsg::GetAllMemoriesResponse { req_id, memories: Vec::<MemoryRecord>::new() }).await;
                }
                
                SystemMsg::PruneMemoriesRequest { req_id, .. } => {
                    let _ = self.kernel_tx.send(SystemMsg::PruneMemoriesResult {
                        req_id, deleted_count: 0, details: "Otomatik".to_string(),
                    }).await;
                }
                _ => {} 
            }
        }
        println!("[STORE] Kanal kapandı, Store Actor kapanıyor.");
    }
}