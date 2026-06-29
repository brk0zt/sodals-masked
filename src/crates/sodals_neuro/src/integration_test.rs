
use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use num_complex::Complex64;
use sodals_kernel::sentinel::AdaptiveSentinel;
use sodals_neuro::{
    bridge::SodalsBridge,
    engine::security::CodeSafetyScanner,
    semantic_network::ResonanceNetwork,
    semantic_parser::CodeWeaver,
};
use std::time::{Duration, Instant};

#[tokio::test(flavor = "multi_thread")]
async fn test_full_system_integration_flow() -> Result<()> {
    // This integration test verifies the high-level data flow through SODALS components.
    // It mocks external dependencies like the LLM model itself but tests the logic of each module.

    let start_time = Instant::now();

    // --- 1. Security Filter Verification ---
    // The AST security filter is designed to block malicious code, not prompts.
    // We test its ability to detect forbidden API calls in a code string.
    println!("[TEST] Verifying AST Security Filter...");
    let malicious_code = "fn main() { std::fs::read_to_string(\"secret.txt\").unwrap(); }";
    let validation_result = CodeSafetyScanner::validate(malicious_code);
    assert!(
        validation_result.is_err(),
        "AST scanner should have blocked std::fs"
    );
    let error_msg = validation_result.unwrap_err();
    assert!(
        error_msg.contains("Forbidden API 'std::fs'"),
        "Error message did not mention std::fs"
    );
    println!("[OK] AST Security Filter correctly blocked forbidden code.");

    // --- 2. Physics Network Execution ---
    // The core of SODALS is the physics simulation. We weave a prompt into the network
    // and run the simulation to get a final complex-valued state.
    println!("\n[TEST] Verifying Physics Network (Symplectic Euler)...");
    let mut physics_network = ResonanceNetwork::new(100); // Matrix-based network
    let prompt = "Write a Rust code to read a file.";
    // Phase 23: Instantiate CodeWeaver and call instance method
    let mut weaver = CodeWeaver::new().expect("Failed to create CodeWeaver");
    weaver.populate_from_natural_language(&mut physics_network, prompt);

    // Execute 10 steps of the simulation to let the network settle
    let external_force = vec![(0, Complex64::new(1.0, 0.0))]; // Initial "kick" to start dynamics
    for _ in 0..10 {
        physics_network.step(&external_force);
    }
    let final_wave = physics_network.get_state();
    assert!(
        final_wave.norm() > 0.0,
        "Physics network should have evolved from a zero state."
    );
    println!("[OK] Physics Network executed 10 steps and produced a non-zero state wave.");

    // --- 3. Complex-to-Real Projection (SodalsBridge) ---
    // The final state from the physics network is projected into the LLM's real-valued vector space.
    println!("\n[TEST] Verifying Complex-to-Real Projection (SodalsBridge)...");
    let wave_dim = final_wave.len();
    let embedding_dim = 384; // Standard embedding size
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F64, &Device::Cpu);
    let bridge = SodalsBridge::new(vb, wave_dim, embedding_dim)?;

    let projected_embedding_tensor = bridge.forward(&final_wave)?;
    assert_eq!(
        projected_embedding_tensor.dims(),
        &[1, embedding_dim],
        "Projected tensor has incorrect dimensions."
    );
    let projected_embedding: Vec<f32> = projected_embedding_tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1()?;
    println!(
        "[OK] SodalsBridge projected complex wave to a {}-dimensional real vector.",
        embedding_dim
    );

    // --- 4. Differential Privacy (Laplace Noise) ---
    // The real-valued vector is made differentially private before potential external use.
    println!("\n[TEST] Verifying Differential Privacy (Laplace Mechanism)...");
    let sentinel = AdaptiveSentinel::new(1.0, 0.5, 2.0); // epsilon = 1.0
    let (noisy_vector, noise_level) = sentinel.secure_and_heal(&projected_embedding)?;
    assert_ne!(
        projected_embedding, noisy_vector,
        "Laplace noise should have been added."
    );
    assert!(noise_level > 0.0, "Noise level should be positive.");
    println!("[OK] Laplace noise correctly applied to the projected vector.");

    // --- 5. Final LLM Generation (Mocked) ---
    // The (noisy) vector would bias the LLM. We just confirm the flow is complete.
    println!("\n[TEST] Verifying final LLM generation step (mocked)...");
    let final_output = "fn main() {\n    // Secure, validated file reading logic would be generated here.\n}";
    println!(
        "[OK] Mock LLM generation completed with final output: \n{}",
        final_output
    );

    // --- 6. Performance Assertion ---
    let duration = start_time.elapsed();
    assert!(
        duration < Duration::from_millis(1000),
        "Integration test took too long: {:?}",
        duration
    );
    println!(
        "\n[SUCCESS] Full integration test completed in {:?}.",
        duration
    );

    Ok(())
}














