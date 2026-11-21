    pub fn from_safetensors(
        vb: candle_nn::VarBuilder,
        cfg: &Qwen2Config,
        device: &candle_core::Device,
    ) -> candle_core::Result<Self> {
        use candle_nn::Module;
        
        let embed_weight = vb.pp("model.embed_tokens").get((cfg.vocab_size, cfg.hidden_size), "weight")?;
        let embed_tokens = candle_nn::Embedding::new(embed_weight, cfg.hidden_size);
        let rotary_emb = std::sync::Arc::new(RotaryEmbedding::new(cfg, device)?);
        
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let is_ssm_model = vb.contains_tensor("model.layers.0.ssm_out.weight");
        if is_ssm_model {
            eprintln!("[SSM-ST] Detected Qwen3.5 SSM architecture in safetensors");
        }
        
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer_vb = vb.pp(&format!("model.layers.{}", layer_idx));
            let layer_has_ssm = is_ssm_model && layer_vb.contains_tensor("ssm_out.weight");
            
            let (self_attn, ssm_block) = if layer_has_ssm {
                // SSM Layer
                // Load fused qkv or separate
                let attn_vb = layer_vb.pp("self_attn");
                let ssm_in_proj = if attn_vb.contains_tensor("qkv_proj.weight") {
                    UnifiedLinear::Regular(candle_nn::linear_no_bias(cfg.hidden_size, cfg.hidden_size * 3, attn_vb.pp("qkv_proj"))?)
                } else {
                    let w = attn_vb.get((cfg.hidden_size * 3, cfg.hidden_size), "weight")?;
                    UnifiedLinear::Regular(candle_nn::Linear::new(w, None)) // Fallback or needs proper separate loading
                };
                
                let attn_gate = if layer_vb.contains_tensor("attn_gate.weight") {
                    UnifiedLinear::Regular(candle_nn::linear_no_bias(cfg.hidden_size, cfg.hidden_size, layer_vb.pp("attn_gate"))?)
                } else {
                    UnifiedLinear::Regular(candle_nn::linear_no_bias(cfg.hidden_size, cfg.hidden_size, attn_vb.pp("gate_proj"))?) // standard hf name
                };
                
                let conv1d_w = layer_vb.pp("ssm_conv1d").get((cfg.hidden_size, 1, 4), "weight")?; // Example shape
                let conv1d = conv1d_w; // simplified
                
                let ssm_out_proj = UnifiedLinear::Regular(candle_nn::linear_no_bias(cfg.hidden_size, cfg.hidden_size, layer_vb.pp("ssm_out"))?);
                let ssm_norm = layer_vb.pp("ssm_norm").get(cfg.hidden_size, "weight").ok().and_then(|w| QRmsNorm::new(w, 1e-6).ok());
                
                let ssm_a = layer_vb.get(32, "ssm_a")?;
                let ssm_dt = layer_vb.pp("ssm_dt").get(32, "bias")?;
                
                let ssm_beta = UnifiedLinear::Regular(candle_nn::linear_no_bias(cfg.hidden_size, cfg.hidden_size, layer_vb.pp("ssm_beta"))?);
                let ssm_alpha = UnifiedLinear::Regular(candle_nn::linear_no_bias(cfg.hidden_size, cfg.hidden_size, layer_vb.pp("ssm_alpha"))?);

                (None, Some(crate::models::qwen_ssm::QwenSsmBlock::new(
                    ssm_in_proj, attn_gate, conv1d, ssm_out_proj, ssm_norm, ssm_a, ssm_dt, ssm_beta, ssm_alpha, device
                )?))
            } else {
                // Standard MHA Layer
                let attn_vb = layer_vb.pp("self_attn");
                let q_proj = UnifiedLinear::Regular(candle_nn::linear_no_bias(cfg.hidden_size, cfg.num_attention_heads * cfg.hidden_size / cfg.num_attention_heads, attn_vb.pp("q_proj"))?);
                let k_proj = UnifiedLinear::Regular(candle_nn::linear_no_bias(cfg.hidden_size, cfg.num_key_value_heads * cfg.hidden_size / cfg.num_attention_heads, attn_vb.pp("k_proj"))?);
                let v_proj = UnifiedLinear::Regular(candle_nn::linear_no_bias(cfg.hidden_size, cfg.num_key_value_heads * cfg.hidden_size / cfg.num_attention_heads, attn_vb.pp("v_proj"))?);
                let o_proj = UnifiedLinear::Regular(candle_nn::linear_no_bias(cfg.num_attention_heads * cfg.hidden_size / cfg.num_attention_heads, cfg.hidden_size, attn_vb.pp("o_proj"))?);
                
                let q_norm = attn_vb.pp("q_norm").get(cfg.num_attention_heads * cfg.hidden_size / cfg.num_attention_heads, "weight").ok().and_then(|w| QRmsNorm::new(w, 1e-6).ok());
                let k_norm = attn_vb.pp("k_norm").get(cfg.num_key_value_heads * cfg.hidden_size / cfg.num_attention_heads, "weight").ok().and_then(|w| QRmsNorm::new(w, 1e-6).ok());
                
                let self_attn = QAttention::new(
                    q_proj, Some(k_proj), Some(v_proj), o_proj,
                    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.num_attention_heads / cfg.num_key_value_heads,
                    cfg.hidden_size / cfg.num_attention_heads,
                    rotary_emb.clone(), q_norm, k_norm, None, None, None, None
                );
                (Some(self_attn), None)
            };
            
            let mlp_vb = layer_vb.pp("mlp");
            let gate_proj = UnifiedLinear::Regular(candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, mlp_vb.pp("gate_proj"))?);
            let up_proj = UnifiedLinear::Regular(candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, mlp_vb.pp("up_proj"))?);
            let down_proj = UnifiedLinear::Regular(candle_nn::linear_no_bias(cfg.intermediate_size, cfg.hidden_size, mlp_vb.pp("down_proj"))?);
            let mlp = QMlp::new(gate_proj, up_proj, down_proj);
            
            let input_layernorm = QRmsNorm::new(layer_vb.pp("input_layernorm").get(cfg.hidden_size, "weight")?, 1e-6)?;
            let post_attention_layernorm = QRmsNorm::new(layer_vb.pp("post_attention_layernorm").get(cfg.hidden_size, "weight")?, 1e-6)?;
            
            layers.push(QDecoderLayer::new(
                self_attn, ssm_block, mlp, input_layernorm, post_attention_layernorm, None
            ));
        }
        
        let norm = QRmsNorm::new(vb.pp("model.norm").get(cfg.hidden_size, "weight")?, 1e-6)?;
        let lm_head = UnifiedLinear::Regular(candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?);
        
        let qwen_model = Qwen2Model::new(embed_tokens, layers, norm, device.clone())?;
        Ok(Self::new(qwen_model, lm_head))
    }
