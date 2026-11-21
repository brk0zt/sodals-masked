    pub fn from_safetensors(
        vb: candle_nn::VarBuilder,
        cfg: &Qwen2Config,
        device: &candle_core::Device,
    ) -> candle_core::Result<Self> {
        use candle_nn::Module;
        
        let base_prefix = if vb.contains_tensor("model.language_model.embed_tokens.weight") {
            "model.language_model"
        } else {
            "model"
        };
        
        let embed_weight = vb.pp(format!("{}.embed_tokens", base_prefix)).get((cfg.vocab_size, cfg.hidden_size), "weight")?;
        let embed_tokens = candle_nn::Embedding::new(embed_weight, cfg.hidden_size);
        let rotary_emb = std::sync::Arc::new(RotaryEmbedding::new(cfg, device)?);
        
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let is_ssm_model = vb.contains_tensor(&format!("{}.layers.0.ssm_out.weight", base_prefix));
        if is_ssm_model {
            eprintln!("[SSM-ST] Detected Qwen3.5 SSM architecture in safetensors");
        }
        
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer_vb = vb.pp(&format!("{}.layers.{}", base_prefix, layer_idx));
            let layer_has_ssm = is_ssm_model && layer_vb.contains_tensor("ssm_out.weight");
