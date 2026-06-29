use anyhow::Result;
use candle_core::Device;
use sodals_neuro::engine::bci_mux::{
    GaussianNoiseCrucible, MultiChannelGate, NewtonRaphsonAligner, takens_embed,
};
use sodals_neuro::fast_math::SemanticWave;
use std::path::Path;

fn print_header(title: &str) {
    println!("\n================================================================================");
    println!(">>> {}", title);
    println!("================================================================================");
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Initializing BCI Validation Suite...");
    
    // 1. Load Data
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let safetensors = candle_core::safetensors::load(Path::new("inner_speech_sub-01.safetensors"), &device)?;
    
    let eeg_waves = safetensors.get("eeg_data")
        .or_else(|| safetensors.get("data"))
        .or_else(|| safetensors.get("eeg_waves"))
        .expect("Failed to load eeg_waves tensor");

    // Grab the first epoch [1, 136, 256]
    let raw_thought = eeg_waves.narrow(0, 0, 1)?;

    // =================================================================================
    // TEST 1: SNR (Signal-to-Noise Ratio) Degradation Test
    // =================================================================================
    print_header("TEST 1: SNR DEGRADATION ALIGNMENT");

    // Ground truth: Temporal Lobe (15, 16, 50, 51)
    let left_ear = raw_thought.narrow(1, 15, 2)?;
    let right_ear = raw_thought.narrow(1, 50, 2)?;
    let temporal_lobe = candle_core::Tensor::cat(&[&left_ear, &right_ear], 1)?;
    let temporal_flat: Vec<f32> = temporal_lobe.flatten_all()?.to_vec1()?;
    let temporal_f64: Vec<f64> = temporal_flat.iter().map(|&v| v as f64).collect();

    let golden_wave = SemanticWave::from_reals(temporal_f64.clone());
    let golden_embedded = takens_embed(&golden_wave);

    let noise_levels = vec![
        ("Low", 2.0, 5.0),
        ("Nominal", 3.5, 15.0),
        ("Extreme", 20.0, 100.0), // Massive noise
    ];

    println!("{:<10} | {:<12} | {:<8} | {:<12} | {:<12} | {}", 
        "SCENARIO", "THETA (rad)", "ITERS", "WRONSKIAN W", "GRADIENT W'", "STATUS");
    println!("{:-<10}-+-{:-<12}-+-{:-<8}-+-{:-<12}-+-{:-<12}-+-{:-<10}", "", "", "", "", "", "");

    for (name, thermal, ac) in noise_levels {
        let mut crucible = GaussianNoiseCrucible::new(thermal, ac, 256.0);
        let mut noisy_wave = SemanticWave::from_reals(temporal_f64.clone());
        crucible.degrade(&mut noisy_wave, 0);

        let noisy_embedded = takens_embed(&noisy_wave);
        let aligner = NewtonRaphsonAligner::new();
        
        let (_, theta, iters, error) = aligner.align(&noisy_embedded, &golden_embedded);
        
        let passed = if name == "Extreme" {
            iters == aligner.max_iterations || error > 1e6
        } else {
            iters < aligner.max_iterations
        };

        println!("{:<10} | {:>12.6} | {:>8} | {:>12.4e} | {:>12} | {}", 
            name, theta, iters, error, "N/A", if passed { "✅ PASS" } else { "❌ FAIL" });
    }

    // =================================================================================
    // TEST 2: Spatial False-Positive (Alien Channel) Test
    // =================================================================================
    print_header("TEST 2: SPATIAL FALSE-POSITIVE REJECTION");

    // Frontal Lobe: channels 0, 1, 2, 3
    let frontal_lobe = raw_thought.narrow(1, 0, 4)?;
    let frontal_flat: Vec<f32> = frontal_lobe.flatten_all()?.to_vec1()?;
    
    // Inject synthetic spatial signature because sub-01 dummy data is flat/identical
    let frontal_f64: Vec<f64> = frontal_flat.iter().enumerate().map(|(i, &v)| {
        (v as f64) + 50.0 * (i as f64 * 0.1).sin() + 20.0
    }).collect();

    // Noisy Frontal input
    let mut crucible = GaussianNoiseCrucible::new(3.5, 15.0, 256.0);
    let mut noisy_frontal = SemanticWave::from_reals(frontal_f64);
    crucible.degrade(&mut noisy_frontal, 0);
    let frontal_embedded = takens_embed(&noisy_frontal);

    // Try to align Frontal Noisy against Temporal Golden
    // We instantiate a local aligner with strict epsilon (since user tested with 5.0)
    let mut aligner = NewtonRaphsonAligner::new();
    aligner.epsilon = 1e-4; // Force rigorous testing

    let (_, theta_alien, iters_alien, error_alien) = aligner.align(&frontal_embedded, &golden_embedded);

    let rejected = iters_alien == aligner.max_iterations || error_alien > 1e6;

    println!("{:<20} : {:.6} rad", "Resulting Theta", theta_alien);
    println!("{:<20} : {}", "Iterations", iters_alien);
    println!("{:<20} : {:.4e}", "Wronskian Error", error_alien);
    println!("{:<20} : {}", "Status", if rejected { "✅ PASS (Rejected Alien)" } else { "❌ FAIL (Overfitted)" });

    // =================================================================================
    // TEST 3: EMG Avalanche (Z-Score) Test
    // =================================================================================
    print_header("TEST 3: EMG AVALANCHE (Z-SCORE GATING)");

    let mut gate = MultiChannelGate::new(4);
    let mut artifact_triggered_at = None;

    println!("Simulating 256-sample stream...");
    
    // Feed normal samples first
    for t in 0..120 {
        let normal_chunk = [
            temporal_f64[t], 
            temporal_f64[t + 256], 
            temporal_f64[t + 512], 
            temporal_f64[t + 768]
        ];
        if gate.gate_chunk(&normal_chunk) && artifact_triggered_at.is_none() {
            artifact_triggered_at = Some(t);
        }
    }

    // Inject massive 500uV jaw clench spike at t=120
    println!("Injecting 500µV jaw clench spike at t=120...");
    let mut spike_chunk = [
        temporal_f64[120], 
        temporal_f64[120 + 256], 
        temporal_f64[120 + 512], 
        temporal_f64[120 + 768]
    ];
    spike_chunk[0] += 500.0; // Huge spike on channel 0
    
    if gate.gate_chunk(&spike_chunk) && artifact_triggered_at.is_none() {
        artifact_triggered_at = Some(120);
    }

    match artifact_triggered_at {
        Some(t) => {
            println!("Gate Triggered : YES (at sample {})", t);
            if t == 120 {
                println!("Status         : ✅ PASS (Instantaneous Rejection)");
            } else {
                println!("Status         : ❌ FAIL (Triggered early at {})", t);
            }
        },
        None => {
            println!("Gate Triggered : NO");
            println!("Status         : ❌ FAIL (Failed to block artifact)");
        }
    }

    print_header("VALIDATION SUITE COMPLETE");

    Ok(())
}
