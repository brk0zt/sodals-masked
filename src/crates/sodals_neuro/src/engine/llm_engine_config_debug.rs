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
                    match serde_json::from_value::<crate::models::qwen_custom::Qwen2Config>(config_obj) {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!("[LlmEngine] ?? Failed to parse config.json into Qwen2Config: {}. Falling back to default.", e);
                            crate::models::qwen_custom::Qwen2Config::default()
                        }
                    }
                } else {
                    eprintln!("[LlmEngine] ?? Failed to parse config.json as JSON. Falling back to default.");
                    crate::models::qwen_custom::Qwen2Config::default()
                }
            } else {
                eprintln!("[LlmEngine] ?? No config.json found. Falling back to default.");
                crate::models::qwen_custom::Qwen2Config::default()
            };
