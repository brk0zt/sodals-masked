// src/lib.rs - TEMİZLENMİŞ VERSİYON

use wasm_bindgen::prelude::*;
use sha2::Sha256;
use hkdf::Hkdf;
use chacha20poly1305::{
    aead::{Aead, KeyInit},
    ChaCha20Poly1305, Key, Nonce
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

// --- RESTORED: Neuro-Symbolic Modules (delegated to sodals_neuro) ---
pub use sodals_neuro::semantic_network;
pub use sodals_neuro::fast_math;

// WASM Panic Hook (Tarayıcı konsolunda hata görmek için)
#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

/// İstemci Kasası: Tarayıcı Belleğinde Yaşar
#[wasm_bindgen]
pub struct ClientVault {
    master_key: Vec<u8>,
}

impl Default for ClientVault {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl ClientVault {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ClientVault {
        let mut key = [0u8; 32];
        let _ = getrandom::getrandom(&mut key); // RNG failure usually panics anyway in wasm
        ClientVault { master_key: key.to_vec() }
    }

    pub fn encrypt_file(&self, file_id_str: &str, data: &[u8]) -> Result<Vec<u8>, JsValue> {
        let hk = Hkdf::<Sha256>::new(Some(file_id_str.as_bytes()), &self.master_key);
        let mut file_key = [0u8; 32];
        hk.expand(b"sodals_file_enc", &mut file_key)
            .map_err(|_| JsValue::from_str("Key expansion failed"))?;

        let cipher = ChaCha20Poly1305::new(Key::from_slice(&file_key));
        
        let mut nonce_bytes = [0u8; 12];
        getrandom::getrandom(&mut nonce_bytes).expect("RNG Failure");
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = cipher.encrypt(nonce, data)
            .map_err(|e| JsValue::from_str(&format!("Encryption failed: {}", e)))?;

        let mut packet = Vec::with_capacity(12 + ciphertext.len());
        packet.extend_from_slice(&nonce_bytes);
        packet.extend_from_slice(&ciphertext);
        
        Ok(packet)
    }

    pub fn blind_vector(&self, vector: &[f32], context_id: &str) -> Result<Vec<f32>, JsValue> {
        let hk = Hkdf::<Sha256>::new(Some(context_id.as_bytes()), &self.master_key);
        let mut seed = [0u8; 32];
        hk.expand(b"sodals_vector_blind", &mut seed)
            .map_err(|_| JsValue::from_str("Seed derivation failed"))?;

        let mut rng = ChaCha20Rng::from_seed(seed);
        let mut blinded = vector.to_vec();

        let len = blinded.len();
        let rotations = len / 2; 

        for _ in 0..rotations {
            let i = rng.gen_range(0..len);
            let j = rng.gen_range(0..len);
            if i == j { continue; }

            let theta = rng.gen::<f32>() * 2.0 * std::f32::consts::PI;
            let c = theta.cos();
            let s = theta.sin();

            let vi = blinded[i];
            let vj = blinded[j];

            blinded[i] = vi * c - vj * s;
            blinded[j] = vi * s + vj * c;
        }

        Ok(blinded)
    }
}

// --- PYTHON ENTEGRASYONU (SodalsEngine) ---
// Sadece backend derlemesinde aktif olur
#[cfg(not(target_arch = "wasm32"))]
use pyo3::{prelude::*, types::PyModule, Bound};

#[cfg(not(target_arch = "wasm32"))]
#[pyclass(name = "SodalsEngine")]
pub struct SodalsEngineWrapper {
    inner: ClientVault,
}

#[cfg(not(target_arch = "wasm32"))]
#[pymethods]
impl SodalsEngineWrapper {
    #[new]
    fn new() -> Self {
        SodalsEngineWrapper { inner: ClientVault::new() }
    }

    fn encrypt_file(&self, file_id: &str, data: &[u8]) -> PyResult<Vec<u8>> {
        self.inner.encrypt_file(file_id, data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e)))
    }

    fn blind_vector(&self, vector: Vec<f32>, context_id: &str) -> PyResult<Vec<f32>> {
        self.inner.blind_vector(&vector, context_id)
             .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e)))
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[pymodule]
fn sodals_core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SodalsEngineWrapper>()?;
    Ok(())
}