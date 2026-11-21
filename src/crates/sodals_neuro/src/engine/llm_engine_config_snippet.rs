            let cfg = if let Ok(config_str) = std::fs::read_to_string(model_dir.join("config.json")) {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&config_str) {
                    let mut config_obj = val.clone();
                    if let Some(text_config) = val.get("text_config") {
                        if let Some(obj) = config_obj.as_object_mut() {
                            if let Some(text_obj) = text_config.as_object() {
                                for (k, v) in text_obj {
                                    obj.insert(k.clone(), v.clone());
                                }
                            }
                        }
                    }
                    
                    let mut cfg = crate::models::qwen_custom::Qwen2Config::default();
                    if let Some(v) = config_obj.get("vocab_size").and_then(|v| v.as_u64()) { cfg.vocab_size = v as usize; }
                    if let Some(v) = config_obj.get("hidden_size").and_then(|v| v.as_u64()) { cfg.hidden_size = v as usize; }
                    if let Some(v) = config_obj.get("intermediate_size").and_then(|v| v.as_u64()) { cfg.intermediate_size = v as usize; }
                    if let Some(v) = config_obj.get("num_hidden_layers").and_then(|v| v.as_u64()) { cfg.num_hidden_layers = v as usize; }
                    if let Some(v) = config_obj.get("num_attention_heads").and_then(|v| v.as_u64()) { cfg.num_attention_heads = v as usize; }
                    if let Some(v) = config_obj.get("num_key_value_heads").and_then(|v| v.as_u64()) { cfg.num_key_value_heads = v as usize; }
                    if let Some(v) = config_obj.get("max_position_embeddings").and_then(|v| v.as_u64()) { cfg.max_position_embeddings = v as usize; }
                    if let Some(v) = config_obj.get("rms_norm_eps").and_then(|v| v.as_f64()) { cfg.rms_norm_eps = v; }
                    if let Some(v) = config_obj.get("rope_theta").and_then(|v| v.as_f64()) { cfg.rope_theta = v; }
                    if let Some(v) = config_obj.get("sliding_window").and_then(|v| v.as_u64()) { cfg.sliding_window = Some(v as usize); }
                    
                    eprintln!("[LlmEngine] ?? Parsed Qwen2Config: vocab_size={}, hidden_size={}", cfg.vocab_size, cfg.hidden_size);
                    cfg
                } else {
                    crate::models::qwen_custom::Qwen2Config::default()
                }
            } else {
                crate::models::qwen_custom::Qwen2Config::default()
            };
