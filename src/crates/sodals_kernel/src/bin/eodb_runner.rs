// src/crates/sodals_kernel/src/bin/eodb_runner.rs
// EODB Benchmark Runner - Enterprise On-Device Benchmark

use anyhow::Result;
use sodals_kernel::oedb_runner::{run_eodb_benchmark, EodbRunner};
use sodals_neuro::llm_core::LlmEngine;
use sodals_common::config::SystemConfig;
use sodals_common::log_manager::{LogManager, LogLevel};
use std::env;
use tracing_subscriber;
use tracing::{info, error};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,sodals=debug".into())
        )
        .with_ansi(true)
        .init();

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();

    // ── PHASE 47: The Crucible Jury Execution Mode ──
    if args.contains(&"--crucible".to_string()) {
        info!("[CRUCIBLE] Launching Deep-Tech Statistical Baseline Runner...");
        sodals_neuro::sodals_alloc::init_memory_monitor();
        let config = SystemConfig::load().unwrap_or_else(|e| {
            error!("Failed to load config: {}", e);
            SystemConfig::default()
        });
        let logger = LogManager::new(LogLevel::INFO);
        let mut engine = LlmEngine::new(&config, &logger)?;
        
        sodals_kernel::jury_report::JuryReport::execute_crucible(&mut engine).await?;
        return Ok(());
    }
    
    if args.len() < 2 {
        error!("Usage: {} <eodb_scenarios.json> [--with-engine]", args[0]);
        error!("  --with-engine: Attach actual LLM engine for real testing");
        error!("  --crucible: Launch Phase 47 Statistical Baseline Runner (Jury Report)");
        std::process::exit(1);
    }
    
    let json_path = &args[1];
    let with_engine = args.contains(&"--with-engine".to_string());
    let icl_mode = args.contains(&"--icl".to_string());
    
    info!("[EODB] Starting benchmark runner...");
    info!("[EODB] Scenarios file: {}", json_path);
    info!("[EODB] LLM Engine: {}", if with_engine { "ENABLED" } else { "AUTO-INITIALIZE" });
    if icl_mode {
        info!("[ICL] Industrial Champions League Mode: ENABLED");
    }
    
    // Initialize memory monitor
    sodals_neuro::sodals_alloc::init_memory_monitor();
    
    // Run benchmark
    if with_engine {
        info!("[EODB] Initializing LLM Engine with QMatMul support...");
        let config = SystemConfig::load().unwrap_or_else(|e| {
            error!("Failed to load config: {}", e);
            SystemConfig::default()
        });
        let logger = LogManager::new(LogLevel::INFO);
        let engine = LlmEngine::new(&config, &logger)?;
        
        info!("[EODB] ✅ LLM Engine initialized successfully");
        
        if icl_mode {
            let mut runner = EodbRunner::from_file(json_path)?;
            runner.attach_engine(engine);
            runner.set_icl_mode(true);
            let report = runner.run_all().await?;
            runner.print_report_card(&report);
        } else {
            run_eodb_benchmark(json_path, Some(engine)).await?;
        }
    } else {
        // Auto-initialization within the benchmark runner
        if icl_mode {
            let mut runner = EodbRunner::from_file(json_path)?;
            runner.set_icl_mode(true);
            let report = runner.run_all().await?;
            runner.print_report_card(&report);
        } else {
            run_eodb_benchmark(json_path, None).await?;
        }
    }
    
    Ok(())
}
