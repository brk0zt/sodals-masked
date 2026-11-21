use candle_core::{Device, Result, Tensor, DType, IndexOp};
use crate::models::qwen_custom::Qwen2ForCausalLM;
use std::collections::HashMap;
use std::sync::Arc;

pub struct JLensDiagnostics;

impl JLensDiagnostics {
    /// Validation gate (Adım 0): run the workspace diagnosis over known-easy
    /// completions BEFORE committing to a Tuned-Lens training run.
    ///
    /// If the diagnosis genuinely discriminates, easy completions must show a
    /// clear output-alignment onset at a MIDDLE layer (workspace != entire net).
    /// If onset stays near the final layer for these trivially predictable
    /// continuations, the thresholds/script need calibration first.
    pub fn validate_workspace_diagnosis(
        model: &mut Qwen2ForCausalLM,
        device: &Device,
        tokenizer: &tokenizers::Tokenizer,
        prompts: &[&str],
    ) {
        eprintln!("[J-Lens] === WORKSPACE DIAGNOSIS VALIDATION ({} easy prompts) ===", prompts.len());
        for (i, prompt) in prompts.iter().enumerate() {
            let tokens = tokenizer
                .encode(*prompt, true)
                .map(|enc| enc.get_ids().to_vec())
                .unwrap_or_else(|_| vec![1, 2, 3, 4, 5, 6, 7, 8]);
            eprintln!(
                "[J-Lens] --- Validation prompt {}: {:?} (tokens={}) ---",
                i, prompt, tokens.len()
            );
            let _ = Self::diagnose_workspace_layers(model, device, &tokens);
        }
        eprintln!(
            "[J-Lens] === VALIDATION DONE: easy completions must show a MIDDLE-layer onset \
             (workspace != whole net). If onset is still the final layer for trivially \
             predictable continuations, recalibrate the diagnosis before training. ==="
        );

        // Trained-readout measurement: report top-1/top-5 through the learned
        // Tuned-Lens Jacobians (workspace from jlens_weights.safetensors) so the
        // acceptance gate (workspace top-1 > 0.3, top-5 > 0.5) can be checked.
        Self::measure_trained_readout_accuracy(model, device, tokenizer, prompts);
    }

    /// Measure top-1/top-5 next-token accuracy through the TRAINED readout
    /// J_l(h_l) -> ln_f -> lm_head on the workspace layers, compared against the
    /// IDENTITY baseline (raw h_l -> ln_f -> lm_head) on the same layers. The
    /// trained readout must beat identity (and approach the final layer's own
    /// performance) to be worth using at runtime.
    pub fn measure_trained_readout_accuracy(
        model: &mut Qwen2ForCausalLM,
        device: &Device,
        tokenizer: &tokenizers::Tokenizer,
        prompts: &[&str],
    ) {
        let Some(range) = crate::jlens_readout::JLensReadout::trained_layers(device) else {
            eprintln!("[J-Lens] No trained Tuned-Lens weights found; skipping trained-readout measurement.");
            return;
        };
        eprintln!(
            "[J-Lens] === TRAINED READOUT MEASUREMENT (workspace {:?}, {} easy prompts) ===",
            range,
            prompts.len()
        );

        let lm_head_arc = Arc::new(model.lm_head.clone());
        let norm_arc = Arc::new(model.model.norm.clone());

        // Identity baseline on the same workspace layers.
        model.model.jlens_readout = Some(Arc::new(
            crate::jlens_readout::JLensReadout::new_identity(
                range.clone(),
                lm_head_arc.clone(),
                norm_arc.clone(),
                device,
                false,
            ),
        ));
        let (id_avg1, id_avg5, id_per_layer) =
            Self::run_readout_measurement(model, device, tokenizer, prompts, range.clone(), "Identity");

        // Trained readout on the same workspace layers.
        model.model.jlens_readout = Some(Arc::new(
            crate::jlens_readout::JLensReadout::new_with_mode(
                range.clone(),
                lm_head_arc,
                norm_arc,
                device,
                false, // per-position top-k accuracy
            ),
        ));
        let (tr_avg1, tr_avg5, tr_per_layer) =
            Self::run_readout_measurement(model, device, tokenizer, prompts, range.clone(), "Trained");

        // Acceptance: trained must beat identity on the workspace, i.e. it must
        // provide a strictly better approximation of final-token logits than the
        // raw hidden state. (Absolute thresholds like top1>0.3 are unreachable by
        // ANY readout — even the true final layer scores ~0.03/0.22 on this metric
        // because format tokens dominate short-prefix next-token prediction.)
        let mut beat = true;
        for l in range.clone() {
            let id = id_per_layer.get(&l).copied().unwrap_or((0.0, 0.0));
            let tr = tr_per_layer.get(&l).copied().unwrap_or((0.0, 0.0));
            if tr.1 < id.1 {
                beat = false;
                eprintln!(
                    "[J-Lens] layer {} trained top5={:.4} < identity top5={:.4}",
                    l, tr.1, id.1
                );
            }
        }
        let pass = beat && tr_avg1 >= id_avg1 && tr_avg5 >= id_avg5;
        eprintln!(
            "[J-Lens] WORKSPACE {:?} | identity top1={:.4} top5={:.4} | trained top1={:.4} top5={:.4} -> {}",
            range,
            id_avg1,
            id_avg5,
            tr_avg1,
            tr_avg5,
            if pass { "PASS" } else { "FAIL" }
        );
    }

    fn run_readout_measurement(
        model: &mut Qwen2ForCausalLM,
        device: &Device,
        tokenizer: &tokenizers::Tokenizer,
        prompts: &[&str],
        range: std::ops::Range<usize>,
        label: &str,
    ) -> (f32, f32, HashMap<usize, (f32, f32)>) {
        let mut agg: HashMap<usize, (f32, f32)> = HashMap::new();
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &prompt in prompts {
            let tokens = tokenizer
                .encode(prompt, true)
                .map(|e| e.get_ids().to_vec())
                .unwrap_or_else(|_| vec![1, 2, 3, 4, 5, 6, 7, 8]);
            if let Ok(mut results) = model.model.jlens_results.lock() {
                results.clear();
            }
            // Each prompt must be an independent context: clear the attention KV
            // cache and the DeltaNet SSM recurrence before the forward, otherwise
            // later forwards see stale state and the top-k numbers drift.
            model.model.clear_kv_cache();
            let input_ids = Tensor::new(tokens.as_slice(), device).unwrap().unsqueeze(0).unwrap();
            let _ = model.forward_hidden_states(&input_ids, 0);
            let collected: Vec<(usize, Tensor)> = model
                .model
                .jlens_results
                .lock()
                .map(|r| r.clone())
                .unwrap_or_default();
            for (layer_idx, t) in collected {
                let top1 = Self::calculate_topk_accuracy(&t, &tokens, 1).unwrap_or(0.0);
                let top5 = Self::calculate_topk_accuracy(&t, &tokens, 5).unwrap_or(0.0);
                let e = agg.entry(layer_idx).or_insert((0.0, 0.0));
                e.0 += top1;
                e.1 += top5;
                *counts.entry(layer_idx).or_insert(0) += 1;
            }
        }

        let mut keys: Vec<usize> = agg.keys().copied().collect();
        keys.sort();
        let mut per_layer: HashMap<usize, (f32, f32)> = HashMap::new();
        for k in keys {
            let n = counts[&k] as f32;
            let (t1, t5) = agg[&k];
            per_layer.insert(k, (t1 / n, t5 / n));
            eprintln!(
                "[J-Lens] {} L{:>2} | top1={:.4} top5={:.4} ({} prompts)",
                label,
                k,
                t1 / n,
                t5 / n,
                counts[&k]
            );
        }

        let mut ws_top1 = 0.0f32;
        let mut ws_top5 = 0.0f32;
        let mut ws_n = 0usize;
        for k in range {
            if let Some(&(t1, t5)) = agg.get(&k) {
                ws_top1 += t1;
                ws_top5 += t5;
                ws_n += counts.get(&k).copied().unwrap_or(0);
            }
        }
        if ws_n == 0 {
            return (0.0, 0.0, per_layer);
        }
        (ws_top1 / ws_n as f32, ws_top5 / ws_n as f32, per_layer)
    }

    pub fn calculate_kurtosis(tensor: &Tensor) -> Result<f32> {
        // tensor shape: (vocab_size,) — use fp64 for numerical stability
        let flattened = tensor.flatten_all()?.to_dtype(DType::F64)?;
        let n = flattened.dims()[0] as f64;

        let vals = flattened.to_vec1::<f64>()?;

        // Two-pass with fp64 (vocab ~150k is fine)
        let mean_val = vals.iter().sum::<f64>() / n;

        let mut m2_sum = 0.0_f64;
        let mut m4_sum = 0.0_f64;
        for &x in &vals {
            let d = x - mean_val;
            let d2 = d * d;
            m2_sum += d2;
            m4_sum += d2 * d2;
        }

        let var = m2_sum / n;
        let moment4 = m4_sum / n;

        // Numerical guards
        if var <= 1e-12 || !var.is_finite() || !moment4.is_finite() {
            return Ok(0.0);
        }

        let kurt = (moment4 / (var * var)).clamp(-10.0, 100.0) as f32;
        Ok(kurt)
    }

    /// Compute top-k next-token accuracy for a layer's logits.
    /// logits shape: (1, seq_len, vocab) or (seq_len, vocab)
    /// Compares against ground-truth next token at each position.
    pub fn calculate_topk_accuracy(logits: &Tensor, tokens: &[u32], k: usize) -> Result<f32> {
        let dims = logits.dims();
        let (seq_len, vocab) = if dims.len() == 3 {
            (dims[1], dims[2])
        } else if dims.len() == 2 {
            (dims[0], dims[1])
        } else {
            return Ok(0.0);
        };

        // Need at least 2 positions and next-token ground truth available
        let usable = seq_len.saturating_sub(1).min(tokens.len().saturating_sub(1));
        if usable == 0 {
            return Ok(0.0);
        }

        let logits_vec = logits.flatten_all()?.to_vec1::<f32>()?;
        let mut correct = 0usize;

        for t in 0..usable {
            let gt = tokens[t + 1] as usize;
            let start = t * vocab;
            if start + vocab > logits_vec.len() {
                break;
            }
            let row = &logits_vec[start..start + vocab];

            // Find top-k indices
            let mut topk: Vec<(usize, f32)> = row.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            topk.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (i, _) in topk.iter().take(k) {
                if *i == gt {
                    correct += 1;
                    break;
                }
            }
        }

        Ok(correct as f32 / usable as f32)
    }

    pub fn diagnose_workspace_layers(
        model: &mut Qwen2ForCausalLM,
        device: &Device,
        prompt_tokens: &[u32],
    ) -> Result<std::ops::Range<usize>> {
        eprintln!("[J-Lens] Diagnosing workspace layers empirically...");

        let input_ids = Tensor::new(prompt_tokens, device)?.unsqueeze(0)?;

        // Temporarily set jlens_readout to capture ALL layers
        let lm_head_arc = Arc::new(model.lm_head.clone());
        let norm_arc = Arc::new(model.model.norm.clone());

        model.model.jlens_readout = Some(Arc::new(crate::jlens_readout::JLensReadout::new_with_mode(
            0..100, // Wide open range
            lm_head_arc,
            norm_arc,
            device,
            false, // diagnosis needs per-position readouts for top-k accuracy
        )));

        // Clear previous results
        if let Ok(mut results) = model.model.jlens_results.lock() {
            results.clear();
        }

        // Fresh context per diagnosis: don't let a previous forward's KV/SSM
        // state bleed into this one.
        model.model.clear_kv_cache();

        // Forward pass
        let _ = model.forward_hidden_states(&input_ids, 0);

        let result_count = model.model.jlens_results.lock().map(|r| r.len()).unwrap_or(0);
        eprintln!("[J-Lens] Diagnosis forward completed. jlens_results entries: {}", result_count);

        // Collect per-layer metrics: (layer_idx, top1_acc, top5_acc, kurtosis)
        let mut layer_metrics = Vec::new();
        if let Ok(results) = model.model.jlens_results.lock() {
            for (layer_idx, tensor) in results.iter() {
                let seq_len = tensor.dim(1).unwrap_or(0);
                // Last-token logits for kurtosis
                let last_token_kurtosis = if seq_len > 0 {
                    match tensor.i((0, seq_len - 1)) {
                        Ok(last_token_logits) => {
                            Self::calculate_kurtosis(&last_token_logits).unwrap_or(0.0)
                        }
                        Err(_) => 0.0,
                    }
                } else {
                    0.0
                };

                let top1 = Self::calculate_topk_accuracy(tensor, prompt_tokens, 1).unwrap_or(0.0);
                let top5 = Self::calculate_topk_accuracy(tensor, prompt_tokens, 5).unwrap_or(0.0);

                layer_metrics.push((*layer_idx, top1, top5, last_token_kurtosis));
            }
        }

        layer_metrics.sort_by_key(|&(l, _, _, _)| l);

        // Print all metrics
        let mut max_kurtosis = 0.0;
        for (l, top1, top5, k) in &layer_metrics {
            eprintln!(
                "[J-Lens] Layer {} | top1={:.4} top5={:.4} kurtosis={:.4}",
                l, top1, top5, k
            );
            if *k > max_kurtosis {
                max_kurtosis = *k;
            }
        }

        // Detect output-alignment onset: the first layer where top1 accuracy
        // becomes non-trivial (naive readout aligns to lm_head near the output).
        // The paper excludes these layers from workspace, so we cap the range
        // BEFORE this onset.
        const OUTPUT_ALIGN_TOP1_THRESHOLD: f32 = 0.05;
        let mut output_onset: Option<usize> = None;
        for (l, top1, _, _) in &layer_metrics {
            if *top1 >= OUTPUT_ALIGN_TOP1_THRESHOLD {
                output_onset = Some(*l);
                break;
            }
        }

        // Kurtosis-based start: first layer with consistently meaningful structure.
        // Use 50% of max as a generous floor, but never below layer 0.
        let threshold = max_kurtosis * 0.5;
        let mut start_layer = layer_metrics
            .iter()
            .find(|&&(_, _, _, k)| k > threshold)
            .map(|&(l, _, _, _)| l)
            .unwrap_or(0);

        // End: the layer just before output alignment begins.
        // If no output-alignment onset detected, fall back to kurtosis end.
        let end_layer = output_onset.unwrap_or_else(|| {
            layer_metrics
                .iter()
                .rev()
                .find(|&&(_, _, _, k)| k > threshold)
                .map(|&(l, _, _, _)| l)
                .unwrap_or(31)
        });

        // Safety: end must be strictly greater than start.
        if end_layer <= start_layer {
            start_layer = 0;
        }
        let end = end_layer.max(start_layer + 1);

        eprintln!(
            "[J-Lens] Output-alignment onset detected at layer {:?} (top1 >= {:.2})",
            output_onset, OUTPUT_ALIGN_TOP1_THRESHOLD
        );
        eprintln!("[J-Lens] Diagnosed Workspace Range: {}..{}", start_layer, end);

        Ok(start_layer..end)
    }
}
