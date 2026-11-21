        if path_str.ends_with(".safetensors") {
            eprintln!("[LlmEngine] ?? Safetensors format detected...");
            
            // To prevent OOM with large safetensors files on 8GB VRAM, force CPU if not overridden
            let safetensors_device = if std::env::var("SODALS_FORCE_CUDA").is_ok() {
                device
            } else {
                eprintln!("[LlmEngine] ?? Forcing CPU for Safetensors to prevent CUDA OOM. Set SODALS_FORCE_CUDA=1 to override.");
                Device::Cpu
            };

            let model_dir = path.parent().unwrap_or(std::path::Path::new("."));
            let model_stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
            
            let mut safetensor_paths: Vec<PathBuf> = Vec::new();
            if let Ok(entries) = std::fs::read_dir(model_dir) {
                for entry in entries.flatten() {
                    let entry_path = entry.path();
                    if let Some(name) = entry_path.file_name().and_then(|n| n.to_str()) {
                        if name.starts_with(model_stem) && name.ends_with(".safetensors") {
                            safetensor_paths.push(entry_path);
                        }
                    }
                }
            }
            safetensor_paths.sort();
            if safetensor_paths.is_empty() {
                safetensor_paths.push(path.clone());
            }
            
            let cfg = if let Ok(config_str) = std::fs::read_to_string(model_dir.join("config.json")) {
                serde_json::from_str(&config_str).unwrap_or_default()
            } else {
                crate::models::qwen_custom::Qwen2Config::default()
            };
            
            let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&safetensor_paths, candle_core::DType::F32, &safetensors_device)? };
            let model = QwenModel::from_safetensors(vb, &cfg, &safetensors_device)?;
            Ok((model, safetensors_device))
        }
