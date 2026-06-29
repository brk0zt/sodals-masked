// src/crates/sodals_kernel/src/bin/sodals_crucible.rs
// Phase 47.5: The Bloodbath - Industrial Statistical Crucible

use anyhow::Result;
use sodals_kernel::jury_report::JuryReport;
use sodals_neuro::llm_core::LlmEngine;
use sodals_common::config::SystemConfig;
use sodals_common::log_manager::{LogManager, LogLevel};
use tracing_subscriber::prelude::*;
use colored::Colorize;

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

    println!("{}", "═══════════════════════════════════════════════════════════════".bright_red().bold());
    println!("{}", "    PHASE 47.5: THE BLOODBATH - INDUSTRIAL CRUCIBLE".bright_red().bold());
    println!("{}", "═══════════════════════════════════════════════════════════════".bright_red().bold());
    println!();
    
    // Initialize memory monitor for VRAM tracking
    sodals_neuro::sodals_alloc::init_memory_monitor();
    
    // Load system configuration
    let config = SystemConfig::load().unwrap_or_else(|e| {
        eprintln!("[CONFIG] Failed to load config: {}", e);
        eprintln!("[CONFIG] Using default configuration");
        SystemConfig::default()
    });
    
    // Initialize logger
    let logger = LogManager::new(LogLevel::INFO);
    
    // Initialize the neuro-engine with QMatMul support
    println!("[ENGINE] Initializing SODALS Neuro-Engine with QMatMul support...");
    let mut engine = LlmEngine::new(&config, &logger)?;
    println!("[ENGINE] ✅ Neuro-Engine initialized successfully");
    println!();
    
    // Execute the industrial crucible
    println!("[CRUCIBLE] Launching Phase 47.5 Industrial Benchmark...");
    println!("[CRUCIBLE] Target: Causal Chains (Vibration→Bearing→Heat), Multimodal Integration, Sub-Second Latency");
    println!();
    
    match JuryReport::execute_crucible(&mut engine).await {
        Ok(_) => {
            println!();
            println!("{}", "═══════════════════════════════════════════════════════════════".bright_green().bold());
            println!("{}", "    CRUCIBLE COMPLETE - Industrial Grade Validation".bright_green().bold());
            println!("{}", "═══════════════════════════════════════════════════════════════".bright_green().bold());
        }
        Err(e) => {
            println!();
            println!("{}", "═══════════════════════════════════════════════════════════════".bright_red().bold());
            println!("{}", "    CRUCIBLE FAILED - Architecture Requires Bloodbath".bright_red().bold());
            println!("{}", "═══════════════════════════════════════════════════════════════".bright_red().bold());
            eprintln!("[FATAL] {}", e);
            std::process::exit(1);
        }
    }
    
    Ok(())
}
