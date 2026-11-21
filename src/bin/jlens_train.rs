use anyhow::{bail, Context, Result};
use candle_core::{quantized::gguf_file, Device, DType, Module, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use sodals_neuro::models::qwen_custom::Qwen2ForCausalLM;
use tokenizers::Tokenizer;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Tuned-Lens training (Belrose et al., 2023, architecture-neutral readout).
///
/// Freezes the Qwen model and learns, per workspace layer l, a linear map
///   J_l: h_l -> h_L   (minimize MSE over a local corpus)
/// so that `lm_head(ln_f(J_l h_l + b_l))` approximates the true final-token logits.
/// The learned `layer_{l}.weight/bias` tensors are saved to jlens_weights.safetensors,
/// which JLensReadout loads at runtime (identity fallback when absent).
///
/// The model forward runs with SODALS_JLENS_CAPTURE=1 so Qwen2Model::forward copies
/// the raw residual-stream h_l (CPU) per captured layer plus a usize::MAX sentinel
/// holding the final pre-norm h_L (the training target).
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = "models/Qwen3.5-9B_Q4_K_M.gguf".to_string();
    let mut tokenizer_path = "models/tokenizer.json".to_string();
    let mut corpus_dir = ".".to_string();
    let mut layers = vec![26usize, 27, 28, 29, 30, 31];
    let mut steps = 40usize;
    let mut batch = 2usize;
    let mut seq_len = 256usize;
    let mut lr = 1e-3f64;
    let mut out = "jlens_weights.safetensors".to_string();
    let mut seed = 1337u64;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { i += 1; model_path = args.get(i).context("--model value")?.clone(); }
            "--tokenizer" => { i += 1; tokenizer_path = args.get(i).context("--tokenizer value")?.clone(); }
            "--corpus" => { i += 1; corpus_dir = args.get(i).context("--corpus value")?.clone(); }
            "--layers" => {
                i += 1;
                let spec = args.get(i).context("--layers value")?.clone();
                layers = parse_layers(&spec)?;
            }
            "--steps" => { i += 1; steps = args.get(i).context("--steps value")?.parse()?; }
            "--batch" => { i += 1; batch = args.get(i).context("--batch value")?.parse()?; }
            "--seq" => { i += 1; seq_len = args.get(i).context("--seq value")?.parse()?; }
            "--lr" => { i += 1; lr = args.get(i).context("--lr value")?.parse()?; }
            "--out" => { i += 1; out = args.get(i).context("--out value")?.clone(); }
            "--seed" => { i += 1; seed = args.get(i).context("--seed value")?.parse()?; }
            "--help" | "-h" => {
                eprintln!("usage: jlens_train --model <gguf> --tokenizer <json> --corpus <dir> --layers 26,27,28,29,30,31 --steps 40 --batch 2 --seq 256 --lr 1e-3 --out jlens_weights.safetensors");
                return Ok(());
            }
            other => bail!("unknown arg: {other}"),
        }
        i += 1;
    }

    let device = pick_device()?;
    eprintln!("[jlens_train] device = {:?}", device);
    eprintln!("[jlens_train] layers = {:?}", layers);
    eprintln!("[jlens_train] steps = {}, batch = {}, seq = {}, lr = {}", steps, batch, seq_len, lr);

    // Capture mode is read inside Qwen2Model::forward at each call.
    std::env::set_var("SODALS_JLENS_CAPTURE", "1");
    let layer_spec = layers.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(",");
    std::env::set_var("SODALS_JLENS_CAPTURE_LAYERS", layer_spec.clone());

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("loading tokenizer from {}: {}", tokenizer_path, e))?;

    let mut file = std::fs::File::open(&model_path)
        .with_context(|| format!("opening model {}", model_path))?;
    let content = gguf_file::Content::read(&mut file)
        .with_context(|| format!("reading GGUF content from {}", model_path))?;
    eprintln!("[jlens_train] loading model from {} ...", model_path);
    let model = Qwen2ForCausalLM::from_gguf(&content, &mut file, &device)?;
    let hidden = model.hidden_size();
    eprintln!("[jlens_train] model loaded. hidden_size = {}", hidden);

    // Local corpus (no HLE questions — contamination protocol).
    let corpus_ids = load_corpus_tokens(&PathBuf::from(&corpus_dir), &tokenizer, 2_000_000)?;
    if corpus_ids.len() < seq_len {
        bail!("corpus too small ({} tokens < seq_len {})", corpus_ids.len(), seq_len);
    }
    eprintln!("[jlens_train] corpus tokens = {}", corpus_ids.len());

    // Trainable linears + single AdamW over all of them.
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let mut linears: HashMap<usize, candle_nn::Linear> = HashMap::new();
    for &layer_idx in &layers {
        let lin = candle_nn::linear(hidden, hidden, vb.pp(format!("layer_{}", layer_idx)))?;
        linears.insert(layer_idx, lin);
    }
    // Identity init (Belrose et al. 2023): the readout starts as the raw hidden
    // state and AdamW refines it toward h_L. Critically, for the LAST workspace
    // layer (target == input) identity IS the optimum, so we never regress below
    // the runtime's identity fallback.
    let mut varmap = varmap;
    for &layer_idx in &layers {
        varmap.set_one(
            format!("layer_{}.weight", layer_idx),
            Tensor::eye(hidden, DType::F32, &device)?,
        )?;
        varmap.set_one(
            format!("layer_{}.bias", layer_idx),
            Tensor::zeros(hidden, DType::F32, &device)?,
        )?;
    }
    let mut optimizer = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW { lr, ..Default::default() },
    )?;

    let mut rng = XorShift::new(seed);
    for step in 0..steps {
        let ids = sample_batch(&corpus_ids, batch, seq_len, &mut rng);
        let ids_t = Tensor::new(ids, &device)?;
        let _ = model.forward_hidden_states(&ids_t, 0)?;

        // Drain capture buffer for this batch.
        let captured: Vec<(usize, Tensor)> = {
            let mut guard = model.model().jlens_capture.lock().unwrap();
            std::mem::take(&mut *guard)
        };
        let mut feats: HashMap<usize, Tensor> = HashMap::new();
        let mut target: Option<Tensor> = None;
        for (idx, t) in captured {
            if idx == usize::MAX {
                target = Some(t);
            } else {
                feats.insert(idx, t);
            }
        }
        let targets_t = ids_t.narrow(1, 1, seq_len - 1)?;
        let flat_targets = targets_t.reshape(batch * (seq_len - 1))?;

        let mut total_loss: Option<Tensor> = None;
        for &layer_idx in &layers {
            if let Some(h_l_cpu) = feats.get(&layer_idx) {
                let h_l_gpu = h_l_cpu.to_device(&device)?;
                let lin = linears.get(&layer_idx).context("linear missing")?;
                let pred = lin.forward(&h_l_gpu)?;
                let norm_out = model.model().norm.forward(&pred)?;
                let logits = model.lm_head.forward(&norm_out)?;
                let logits_slice = logits.narrow(1, 0, seq_len - 1)?;
                let vocab_size = logits.dim(2)?;
                let flat_logits = logits_slice.reshape((batch * (seq_len - 1), vocab_size))?;
                let ce = candle_nn::loss::cross_entropy(&flat_logits, &flat_targets)?;
                let layer_loss = if let Some(ref target_cpu) = target {
                    let target_gpu = target_cpu.to_device(&device)?;
                    let mse = pred.sub(&target_gpu)?.sqr()?.mean_all()?;
                    let scaled_mse = (mse * 0.1)?;
                    ce.add(&scaled_mse)?
                } else {
                    ce
                };
                total_loss = Some(match total_loss {
                    Some(t) => t.add(&layer_loss)?,
                    None => layer_loss,
                });
            }
        }
        let total = total_loss.context("no captured layer features for this batch")?;
        let loss_val = total.to_scalar::<f32>()?;
        optimizer.backward_step(&total)?;

        if step % 10 == 0 || step + 1 == steps {
            eprintln!("[jlens_train] step {}/{} Combined (CE + 0.1*MSE) loss = {:.6}", step + 1, steps, loss_val);
        }
    }

    // Validation & Acceptance Criteria Gate
    eprintln!("\n========================================================");
    eprintln!("[jlens_train] AUTOMATED ACCEPTANCE CRITERIA GATE (KABUL KRİTERLERİ)");
    eprintln!("========================================================");

    let val_ids = sample_batch(&corpus_ids, batch, seq_len, &mut rng);
    let val_t = Tensor::new(val_ids, &device)?;
    let val_targets = val_t.narrow(1, 1, seq_len - 1)?;
    let val_flat_targets = val_targets.reshape(batch * (seq_len - 1))?;
    let _ = model.forward_hidden_states(&val_t, 0)?;
    let captured: Vec<(usize, Tensor)> = {
        let mut guard = model.model().jlens_capture.lock().unwrap();
        std::mem::take(&mut *guard)
    };
    let mut feats: HashMap<usize, Tensor> = HashMap::new();
    let mut target: Option<Tensor> = None;
    for (idx, t) in captured {
        if idx == usize::MAX {
            target = Some(t);
        } else {
            feats.insert(idx, t);
        }
    }
    let target_cpu = target.context("validation capture missing sentinel")?;
    let target_gpu = target_cpu.to_device(&device)?;
    let target_norm = target_gpu.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    
    let mut all_criteria_passed = true;
    for &layer_idx in &layers {
        if let Some(h_l_cpu) = feats.get(&layer_idx) {
            let h_l_gpu = h_l_cpu.to_device(&device)?;
            let lin = linears.get(&layer_idx).context("linear missing")?;
            let pred = lin.forward(&h_l_gpu)?;
            let norm_out = model.model().norm.forward(&pred)?;
            let logits = model.lm_head.forward(&norm_out)?;
            let logits_slice = logits.narrow(1, 0, seq_len - 1)?;
            let vocab_size = logits.dim(2)?;
            let flat_logits = logits_slice.reshape((batch * (seq_len - 1), vocab_size))?;
            let val_ce = candle_nn::loss::cross_entropy(&flat_logits, &val_flat_targets)?;
            let val_ce_val = val_ce.to_scalar::<f32>()?;

            let dot = pred.mul(&target_gpu)?.sum_all()?.to_scalar::<f32>()?;
            let p_norm = pred.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
            let cos = if p_norm > 1e-12 && target_norm > 1e-12 { dot / (p_norm * target_norm) } else { 0.0 };
            
            let ce_passed = val_ce_val <= 10.0;
            let cos_passed = cos >= 0.90;
            let layer_passed = ce_passed && cos_passed;
            if !layer_passed {
                all_criteria_passed = false;
            }
            
            let status = if layer_passed { "✅ PASSED" } else { "⚠️ WARNING" };
            eprintln!(
                "[jlens_train] VALIDATION layer {:>2}: CE loss = {:.4} (<=10.0: {}), Cosine = {:.4} (>=0.90: {}) -> {}",
                layer_idx, val_ce_val, ce_passed, cos, cos_passed, status
            );
        }
    }

    if all_criteria_passed {
        eprintln!("\n[jlens_train] 🚀 ACCEPTANCE CRITERIA PASSED! Saving weights to {}", out);
    } else {
        eprintln!("\n[jlens_train] ⚠️ ACCEPTANCE CRITERIA WARNING: Some layers below target threshold. Saving baseline refined weights to {}", out);
    }
    varmap.save(&out)?;
    eprintln!("[jlens_train] Weights successfully saved to {}.", out);

    Ok(())
}

fn pick_device() -> Result<Device> {
    if candle_core::utils::cuda_is_available() && std::env::var("SODALS_FORCE_CPU").is_err() {
        match Device::new_cuda(0) {
            Ok(cuda) => {
                eprintln!("[jlens_train] CUDA detected, using GPU.");
                Ok(cuda)
            }
            Err(e) => {
                eprintln!("[jlens_train] WARNING: CUDA init failed: {}. Falling back to CPU.", e);
                Ok(Device::Cpu)
            }
        }
    } else {
        Ok(Device::Cpu)
    }
}

fn parse_layers(spec: &str) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if let Some((a, b)) = part.split_once('-') {
            let a: usize = a.trim().parse().context("range start")?;
            let b: usize = b.trim().parse().context("range end")?;
            for v in a..=b {
                out.push(v);
            }
        } else if !part.is_empty() {
            out.push(part.parse().context("layer index")?);
        }
    }
    if out.is_empty() {
        bail!("--layers produced an empty set");
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

const CORPUS_EXTS: &[&str] = &["md", "rs", "py", "txt", "toml", "json", "c", "h", "cpp", "hpp", "html", "tex", "yml", "yaml", "sh"];
const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;

fn load_corpus_tokens(dir: &Path, tokenizer: &Tokenizer, cap: usize) -> Result<Vec<u32>> {
    let mut all = Vec::with_capacity(cap.min(1_000_000));
    let mut stack: Vec<PathBuf> = vec![dir.to_path_buf()];
    let mut files_read = 0usize;
    while let Some(p) = stack.pop() {
        if all.len() >= cap {
            break;
        }
        if p.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&p) {
                for entry in entries.flatten() {
                    let ep = entry.path();
                    let name = ep.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_lowercase();
                    if name == ".git" || name == "target" || name == "node_modules" || name == "hle-main" {
                        continue;
                    }
                    stack.push(ep);
                }
            }
        } else if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
            if CORPUS_EXTS.contains(&ext.to_lowercase().as_str()) {
                if let Ok(meta) = p.metadata() {
                    if meta.len() > MAX_FILE_BYTES {
                        continue;
                    }
                }
                let text = match std::fs::read_to_string(&p) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                if text.trim().is_empty() {
                    continue;
                }
                if let Ok(enc) = tokenizer.encode(text, true) {
                    for id in enc.get_ids() {
                        if all.len() >= cap {
                            break;
                        }
                        all.push(*id);
                    }
                    files_read += 1;
                }
            }
        }
    }
    eprintln!("[jlens_train] corpus files read = {} (dir: {})", files_read, dir.display());
    Ok(all)
}

fn sample_batch(corpus: &[u32], batch: usize, seq_len: usize, rng: &mut XorShift) -> Vec<Vec<u32>> {
    let n = corpus.len();
    let mut out = Vec::with_capacity(batch);
    for _ in 0..batch {
        let start = if n > seq_len { rng.next() as usize % (n - seq_len) } else { 0 };
        let mut row = corpus[start..start + seq_len].to_vec();
        while row.len() < seq_len {
            // Wrapping fill for degenerate small corpora.
            row.extend_from_slice(&corpus[..(seq_len - row.len()).min(n)]);
        }
        out.push(row);
    }
    out
}

struct XorShift {
    state: u64,
}

impl XorShift {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed } }
    }
    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}
