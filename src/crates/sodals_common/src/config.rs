use std::env;
use std::error::Error;


#[derive(Clone, Debug)]
pub struct SystemConfig {
    pub secret_api_key: String,
    pub model_path: String,
    pub tokenizer_path: String,
    pub host: String,
    pub port: u16,
    pub network_capacity: usize,
    pub euler_resonance: bool,
    pub num_universes: usize,  // Configurable parallelism for Ouroboros
}

impl SystemConfig {
    pub fn load() -> Result<Self, Box<dyn Error>> {
        // En sağlam yöntem: Çalışma dizinini ekrana basalım
        let cwd = env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        println!("[Config] Mevcut çalışma dizini: {}", cwd.display());
        
        // Çalışma dizinindeki .env dosyasının tam yolunu bulalım
        let env_path = cwd.join(".env");
        println!("[Config] .env aranıyor: {}", env_path.display());

        // .env dosyasını açıkça yükle
        if let Err(e) = dotenvy::from_path(&env_path) {
            println!("[Config] Belirtilen yolda .env dosyası bulunamadı, varsayılana dönülüyor: {}", e);
            let _ = dotenvy::dotenv();
        } else {
            println!("[Config] .env dosyası başarıyla yüklendi.");
        }

        println!("Loading configuration from environment");

        let secret_api_key = env::var("SODALS_SECRET_KEY")
            .map_err(|_| "CRITICAL: 'SODALS_SECRET_KEY' not found.")?;

        let home_dir = env::var("USERPROFILE")
            .or_else(|_| env::var("HOME"))
            .unwrap_or_else(|_| ".".to_string());
        
        let model_path = env::var("SODALS_MODEL_PATH")
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| format!("{}/models/Qwen3-8B-Q4_K_M.gguf", home_dir));

        let tokenizer_path = env::var("SODALS_TOKENIZER_PATH")
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| format!("{}/models/tokenizer.json", home_dir));

        let host = env::var("SODALS_HOST")
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| "127.0.0.1".to_string());
        
        // --- DEBUG & FIX BLOKU ---
        let port_raw = env::var("SODALS_PORT").unwrap_or_else(|_| "3000".to_string());
        
        // Konsola ne okuduğumuzu basalım:
        println!("[DEBUG] SODALS_PORT Ham Değer: '{}'", port_raw);

        // Temizle ve dönüştür
        let port = port_raw
            .trim() // Boşlukları temizle
            .replace("\"", "") // Tırnak işaretlerini temizle (varsa)
            .replace("'", "")  // Tek tırnakları temizle (varsa)
            .parse::<u16>()
            .map_err(|e| format!("PORT Hatası! Okunan: '{}' -> Hata: {}", port_raw, e))?;
        // -------------------------

        let network_capacity = env::var("SODALS_NETWORK_CAPACITY")
            .unwrap_or_else(|_| "1000".to_string())
            .trim()
            .parse()
            .unwrap_or(1000);

        let euler_resonance = env::var("SODALS_PHYSICS_ENABLED")
            .unwrap_or_else(|_| "true".to_string())
            .parse()
            .unwrap_or(true);

        let num_universes = env::var("SODALS_NUM_UNIVERSES")
            .unwrap_or_else(|_| "2".to_string())
            .trim()
            .parse()
            .unwrap_or(2)
            .max(1);  // At least 1 universe
        Ok(Self { secret_api_key, model_path, tokenizer_path, host, port, network_capacity, euler_resonance, num_universes })

    }
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            secret_api_key: "dev".to_string(),
            model_path: "./models/model.gguf".to_string(),
            tokenizer_path: "./models/tokenizer.json".to_string(),
            host: "127.0.0.1".to_string(),
            port: 3000,
            network_capacity: 1000,
            euler_resonance: true,
            num_universes: 2,  // Local execution default
        }
    }
}
