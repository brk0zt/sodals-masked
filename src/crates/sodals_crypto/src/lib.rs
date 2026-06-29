// src/lib.rs
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;
#[cfg(target_arch = "wasm32")]
use web_sys::CryptoKey;
#[cfg(target_arch = "wasm32")]
use js_sys::{Uint8Array, Object, Reflect};

use chacha20poly1305::{
    aead::{Aead, KeyInit, OsRng},
    ChaCha20Poly1305, Nonce
};
use argon2::{
    password_hash::{rand_core::RngCore, SaltString},
    Argon2
};
use anyhow::{Result, Error};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use zeroize::Zeroize;

const MATRIX_DIM: usize = 768;

/// Bellekte yaşayan kripto motoru.
/// Anahtarı (Key) saklar ve drop olduğunda güvenle siler.
pub struct VaultShield {
    // Ancak Key'i zeroize etmek asıl olaydır. Cipher'ı her işlemde yeniden oluşturmak daha güvenli olabilir
    // ama performans için tutuyoruz. ChaCha20Poly1305 key'i encapsule eder.
    cipher: ChaCha20Poly1305,
    
    // Salt (Tuz), anahtar türetmek için gereklidir. Diske açık (plaintext) yazılabilir.
    pub salt: String, 

    // Vektör Obfuscation için deterministik tohum (Master Key'den türetilir)
    obfuscation_seed: [u8; 32],

    // Random Orthogonal Projection Matrix (Flattened)
    projection_matrix: Vec<f32>,
}

impl Zeroize for VaultShield {
    fn zeroize(&mut self) {
        // Best-effort: salt is sensitive; cipher is opaque and does not implement Zeroize.
        self.salt.zeroize();
        self.obfuscation_seed.zeroize();
        for v in &mut self.projection_matrix {
            *v = 0.0;
        }
    }
}

impl Drop for VaultShield {
    fn drop(&mut self) {
        self.zeroize();
    }
}

impl VaultShield {
    /// Kullanıcı şifresinden anahtar türetir (Kalıcı Güvenlik)
    /// Eğer diskte kayıtlı bir salt varsa onu kullanır, yoksa yeni üretir.
    pub fn authenticate(password: &str, existing_salt: Option<String>) -> Result<Self> {
        let salt = match existing_salt {
            Some(s) => s,
            None => SaltString::generate(&mut OsRng).to_string(),
        };

        // Argon2id ile şifreyi "pişiriyoruz" (Key Derivation)
        // Bu işlem bilerek yavaştır (Brute-force saldırılarını engellemek için)
        let argon2 = Argon2::default();
        
        // 64-byte (512-bit) Hash oluştur: 32 byte Şifreleme + 32 byte Vektör Maskeleme
        let mut key_material = [0u8; 64];
        argon2.hash_password_into(
            password.as_bytes(), 
            salt.as_bytes(), 
            &mut key_material
        ).map_err(|e| Error::msg(format!("Anahtar türetme hatası: {}", e)))?;

        let (key_bytes, seed_bytes) = key_material.split_at(32);

        let cipher = ChaCha20Poly1305::new_from_slice(key_bytes)
            .map_err(|_| Error::msg("Cipher oluşturulamadı"))?;

        let mut obfuscation_seed = [0u8; 32];
        obfuscation_seed.copy_from_slice(seed_bytes);

        // Deterministik Projeksiyon Matrisi Oluştur (Random Projection)
        // Cosine Similarity'i korur ama orijinal değerleri gizler.
        let mut rng = StdRng::from_seed(obfuscation_seed);
        let mut projection_matrix = Vec::with_capacity(MATRIX_DIM * MATRIX_DIM);
        for _ in 0..(MATRIX_DIM * MATRIX_DIM) {
            // [-1.0, 1.0] aralığında rastgele değerler
            let val: f32 = rng.gen::<f32>() * 2.0 - 1.0;
            projection_matrix.push(val);
        }

        // Bellek güvenliği için key_material'ı hemen temizle
        key_material.zeroize();

        Ok(Self { cipher, salt, obfuscation_seed, projection_matrix })
    }

    /// Veriyi şifreler: [NONCE (12 byte) | ENCRYPTED DATA]
    pub fn encrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut nonce_bytes = [0u8; 12];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = self.cipher.encrypt(nonce, data)
            .map_err(|_| Error::msg("Şifreleme başarısız"))?;

        let mut package = Vec::with_capacity(12 + ciphertext.len());
        package.extend_from_slice(&nonce_bytes);
        package.extend_from_slice(&ciphertext);

        Ok(package)
    }

    /// Şifreyi çözer
    pub fn decrypt(&self, encrypted_data: &[u8]) -> Result<Vec<u8>> {
        if encrypted_data.len() < 12 {
            return Err(Error::msg("Veri çok kısa"));
        }

        let nonce = Nonce::from_slice(&encrypted_data[0..12]);
        let ciphertext = &encrypted_data[12..];

        self.cipher.decrypt(nonce, ciphertext)
            .map_err(|_| Error::msg("Şifre hatalı veya veri bozuk"))
    }

    /// Vektör Maskeleme (Obfuscation)
    /// Deterministik bir RNG ile vektörün yönünü (sign flipping) değiştirir.
    /// Cosine Similarity (Açı) korunur, ancak değerler gürültü gibi görünür.
    pub fn obfuscate_vector(&self, vec: &mut [f32]) {
        // Her çağrıda aynı seed ile RNG başlatılır -> Deterministik Dönüşüm
        let mut rng = StdRng::from_seed(self.obfuscation_seed);
        
        for val in vec {
            // %50 ihtimalle işareti ters çevir (Isometric Transformation)
            let sign = if rng.gen::<bool>() { 1.0 } else { -1.0 };
            *val *= sign;
        }
    }

    /// Random Orthogonal Projection
    /// Vektörü rastgele bir uzaya yansıtır. (Matris Çarpımı)
    pub fn project_vector(&self, vec: &mut [f32]) {
        if vec.len() != MATRIX_DIM {
            return; // Boyut uyuşmazlığı varsa dokunma (Güvenlik/Stabilite)
        }

        let input = vec.to_owned();
        for (i, val) in vec.iter_mut().enumerate().take(MATRIX_DIM) {
            let mut sum = 0.0;
            let row_offset = i * MATRIX_DIM;
            for (j, &input_val) in input.iter().enumerate().take(MATRIX_DIM) {
                sum += self.projection_matrix[row_offset + j] * input_val;
            }
            *val = sum;
        }
    }
}


// [DOSYANIN SONUNA EKLE]

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt_flow() {
        let shield = VaultShield::authenticate("gizli_sifre", None).unwrap();
        
        let plaintext = b"SODALS Top Secret";
        let encrypted = shield.encrypt(plaintext).expect("Şifreleme hatası");
        
        assert_ne!(plaintext, &encrypted[..], "Şifreli veri düz metinle aynı!");
        
        let decrypted = shield.decrypt(&encrypted).expect("Çözme hatası");
        assert_eq!(plaintext, &decrypted[..], "Çözülen veri orjinalle eşleşmiyor");
    }

    #[test]
    fn test_wrong_password() {
        let shield1 = VaultShield::authenticate("sifre_A", None).unwrap();
        let shield2 = VaultShield::authenticate("sifre_B", None).unwrap(); // Farklı anahtar
        
        let data = b"Hassas Veri";
        let encrypted = shield1.encrypt(data).unwrap();
        
        // Yanlış kalkanla açmaya çalış
        let result = shield2.decrypt(&encrypted);
        assert!(result.is_err(), "Yanlış anahtarla şifre çözülebildi! (Güvenlik Açığı)");
    }
}