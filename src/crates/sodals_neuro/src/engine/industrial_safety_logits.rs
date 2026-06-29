// src/engine/industrial_safety_logits.rs

use anyhow::Result;
use candle_core::{Tensor, Device};
use std::collections::HashSet;

pub struct IndustrialSafetyLogitsProcessor {
    pub safety_keywords: Vec<String>,
    pub safety_token_ids: HashSet<u32>,
    pub boost_magnitude: f32,
    pub active: bool,
}

impl IndustrialSafetyLogitsProcessor {
    pub fn new(keywords: &[&str], tokenizer: &tokenizers::Tokenizer, boost_magnitude: f32) -> Self {
        let mut safety_token_ids = HashSet::new();
        for &kw in keywords {
            if let Ok(encoding) = tokenizer.encode(kw, false) {
                for &id in encoding.get_ids() {
                    safety_token_ids.insert(id);
                }
            }
            let spaced = format!(" {}", kw);
            if let Ok(encoding) = tokenizer.encode(spaced.as_str(), false) {
                for &id in encoding.get_ids() {
                    safety_token_ids.insert(id);
                }
            }
        }
        Self {
            safety_keywords: keywords.iter().map(|s| s.to_string()).collect(),
            safety_token_ids,
            boost_magnitude,
            active: true,
        }
    }

    pub fn process(&self, logits: &Tensor, device: &Device) -> Result<Tensor> {
        if !self.active || self.safety_token_ids.is_empty() {
            return Ok(logits.clone());
        }

        let vocab_size = logits.dims().last().copied().unwrap_or(0);
        if vocab_size == 0 {
            return Ok(logits.clone());
        }

        let mut bias_data = vec![-2.0f32; vocab_size];
        for &id in &self.safety_token_ids {
            if (id as usize) < vocab_size {
                bias_data[id as usize] = self.boost_magnitude;
            }
        }

        let bias_tensor = Tensor::from_vec(bias_data, vocab_size, device)?;
        logits.broadcast_add(&bias_tensor)
            .map_err(|e| anyhow::anyhow!("Industrial safety bias application failed: {}", e))
    }
}
