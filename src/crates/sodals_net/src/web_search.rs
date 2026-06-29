// src/crates/sodals_net/src/web_search.rs - ETİK HİBRİT ARAMA MOTORU

use reqwest::header;
use scraper::{Html, Selector};
// use anyhow::Result; // Gerekirse açılabilir

pub struct WebSearchEngine {
    client: reqwest::Client,
}

impl Default for WebSearchEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl WebSearchEngine {
    pub fn new() -> Self {
        // [ETİK] Dürüst User-Agent
        let mut headers = header::HeaderMap::new();
        headers.insert("User-Agent", header::HeaderValue::from_static("SODALS/1.0 (Research AI; Open Source)"));

        let client = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(std::time::Duration::from_secs(10)) // 10s Timeout
            .build()
            .unwrap_or_default();

        Self { client }
    }

    pub async fn search(&self, query: &str) -> String {
        // LogManager yerine standart çıktı kullanıyoruz (Circular Dependency'yi önlemek için)
        println!("[WebSearch] DuckDuckGo üzerinden aranıyor: {}", query);
        
        // 1. DuckDuckGo HTML sürümü
        let url = format!("https://html.duckduckgo.com/html/?q={}", query);
        
        match self.client.get(&url).send().await {
            Ok(resp) => {
                // metni alırken hata olursa handle ediyoruz
                match resp.text().await {
                    Ok(html_text) => return self.parse_duckduckgo(&html_text),
                    Err(e) => eprintln!("[WebSearch] Metin okuma hatası: {}", e),
                }
            }
            Err(e) => {
                eprintln!("[WebSearch] Bağlantı hatası: {}", e);
            }
        }
        
        "İnternet bağlantısı kurulamadı veya sonuç bulunamadı.".to_string()
    }

    fn parse_duckduckgo(&self, html: &str) -> String {
        let document = Html::parse_document(html);
        
        // Selector::parse unwrap yerine expect/handle edilebilir ama static string olduğu için güvenli
        let result_sel = Selector::parse(".result__body").unwrap();
        let title_sel = Selector::parse(".result__a").unwrap();
        let snippet_sel = Selector::parse(".result__snippet").unwrap();
        
        let mut results = Vec::new();
        
        for element in document.select(&result_sel).take(3) { 
            let title = element.select(&title_sel).next()
                .map(|e| e.text().collect::<String>())
                .unwrap_or_default();
                
            let snippet = element.select(&snippet_sel).next()
                .map(|e| e.text().collect::<String>())
                .unwrap_or_default();
            
            if !title.is_empty() {
                results.push(format!("- **{}**: {}", title.trim(), snippet.trim()));
            }
        }

        if results.is_empty() {
            return "İlgili sonuç bulunamadı.".to_string();
        }

        results.join("\n")
    }
}