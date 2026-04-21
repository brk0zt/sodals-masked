// src/crates/sodals_kernel/src/bin/eodb_runner.rs
// EODB Benchmark Runner - Enterprise On-Device Benchmark

use anyhow::Result;
use sodals_kernel::oedb_runner::{run_eodb_benchmark, EodbRunner};
use sodals_neuro::llm_core::LlmEngine;
use sodals_common::config::SystemConfig;
use sodals_common::log_manager::{LogManager, LogLevel};
use std::env;
use tracing_subscriber::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,sodals=debug".into())
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();

    // ── PHASE 47: The Crucible Jury Execution Mode ──
    if args.contains(&"--crucible".to_string()) {
        println!("\n[CRUCIBLE] Launching Deep-Tech Statistical Baseline Runner...");
        sodals_neuro::sodals_alloc::init_memory_monitor();
        let config = SystemConfig::load().unwrap_or_else(|e| {
            eprintln!("Failed to load config: {}", e);
            SystemConfig::default()
        });
        let logger = LogManager::new(LogLevel::INFO);
        let mut engine = LlmEngine::new(&config, &logger)?;
        
        sodals_kernel::jury_report::JuryReport::execute_crucible(&mut engine).await?;
        return Ok(());
    }
    
    if args.len() < 2 {
        eprintln!("Usage: {} <eodb_scenarios.json> [--with-engine]", args[0]);
        eprintln!("  --with-engine: Attach actual LLM engine for real testing");
        eprintln!("  --crucible: Launch Phase 47 Statistical Baseline Runner (Jury Report)");
        std::process::exit(1);
    }
    
    let json_path = &args[1];
    let with_engine = args.contains(&"--with-engine".to_string());
    
    println!("[EODB] Starting benchmark runner...");
    println!("[EODB] Scenarios file: {}", json_path);
    println!("[EODB] LLM Engine: {}", if with_engine { "ENABLED" } else { "SIMULATED" });
    
    // Initialize memory monitor
    sodals_neuro::sodals_alloc::init_memory_monitor();
    
    // Run benchmark
    if with_engine {
        println!("[EODB] Initializing LLM Engine with QMatMul support...");
        
        // Initialize system config and logger
        let config = SystemConfig::load().unwrap_or_else(|e| {
            eprintln!("Failed to load config: {}", e);
            SystemConfig::default()
        });
        
        let logger = LogManager::new(LogLevel::INFO);
        
        // Initialize LLM engine with QMatMul-refactored model
        let engine = LlmEngine::new(&config, &logger)?;
        
        println!("[EODB] ✅ LLM Engine initialized successfully");
        
        // Create runner and attach engine
        let mut runner = EodbRunner::from_file(json_path)?;
        runner.attach_engine(engine);
        
        // Run benchmark with real engine
        let report = runner.run_all().await?;
        runner.print_report_card(&report);
    } else {
        run_eodb_benchmark(json_path, None).await?;
    }
    
    Ok(())
}
