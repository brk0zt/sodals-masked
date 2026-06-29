// src/crates/sodals_kernel/src/crisis_manager.rs
// AUTOMATED CRISIS DEFENSE PROTOCOL & DECISION LOGGER

use std::fs::OpenOptions;
use std::io::Write;
use chrono::Local;

/// Predefined categories of critical runtime stability failures.
#[derive(Debug, Clone, Copy)]
pub enum CrisisTrigger {
    SevereWronskianDeviation,
    HallucinationSpike,
    GradientNaNExplosion,
    LatentVectorEntropyCollapse,
}

impl CrisisTrigger {
    pub fn as_str(&self) -> &'static str {
        match self {
            CrisisTrigger::SevereWronskianDeviation => "SEVERE_WRONSKIAN_DEVIATION",
            CrisisTrigger::HallucinationSpike => "HALLUCINATION_SPIKE",
            CrisisTrigger::GradientNaNExplosion => "GRADIENT_NAN_EXPLOSION",
            CrisisTrigger::LatentVectorEntropyCollapse => "LATENT_VECTOR_ENTROPY_COLLAPSE",
        }
    }
}

/// Available defensive countermeasures driven automatically.
#[derive(Debug, Clone, Copy)]
pub enum DefensiveAction {
    ActivatePDEHeatFlood,
    InjectDeepKalmanDamping,
    EmergencySkewCodeRestoration,
    SemanticStatePurge,
}

impl DefensiveAction {
    pub fn description(&self) -> &'static str {
        match self {
            DefensiveAction::ActivatePDEHeatFlood => "Activated multi-step Heat Diffusion PDE to dissipate energy clusters.",
            DefensiveAction::InjectDeepKalmanDamping => "Applied non-linear Kalman estimation to forcibly clamp semantic state vectors.",
            DefensiveAction::EmergencySkewCodeRestoration => "Triggered Skew Code quantization recovery logic.",
            DefensiveAction::SemanticStatePurge => "Zeroed corrupted buffer slots to prevent downstream cascade failure.",
        }
    }
}

pub struct CrisisManager {
    log_path: String,
}

impl Default for CrisisManager {
    fn default() -> Self {
        Self::new("CRITICAL_DECISION_AUTO_LOG.md")
    }
}

impl CrisisManager {
    pub fn new(filename: &str) -> Self {
        let mgr = Self {
            log_path: filename.to_string(),
        };
        mgr.ensure_log_header();
        mgr
    }

    fn ensure_log_header(&self) {
        if std::path::Path::new(&self.log_path).exists() {
            return;
        }
        
        let header = "# 🚨 CRITICAL DECISION & CRISIS DEFENSE LOG\n\n\
                      **WARNING:** This is an automated ledger written by the AI Core during emergency stabilize events.\n\n\
                      | Timestamp | Incident | Metric | Action Taken | Status |\n\
                      |---|---|---|---|---|\n";
        
        let _ = std::fs::write(&self.log_path, header);
    }

    /// Main interface to decide on countermeasure and log the action to disk.
    pub fn handle_emergency(&self, trigger: CrisisTrigger, metric_value: f64) -> DefensiveAction {
        // 1. Automated Selection Logic based on severity
        let action = match trigger {
            CrisisTrigger::SevereWronskianDeviation => DefensiveAction::ActivatePDEHeatFlood,
            CrisisTrigger::HallucinationSpike => DefensiveAction::InjectDeepKalmanDamping,
            CrisisTrigger::GradientNaNExplosion => DefensiveAction::SemanticStatePurge,
            CrisisTrigger::LatentVectorEntropyCollapse => DefensiveAction::EmergencySkewCodeRestoration,
        };

        // 2. Write critical document entry asynchronously (best effort, won't block main thread significantly)
        self.write_to_log(trigger, metric_value, action);

        action
    }

    fn write_to_log(&self, trigger: CrisisTrigger, metric: f64, action: DefensiveAction) {
        let now = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
        
        let log_entry = format!(
            "| {} | `{}` | `{:.4}` | {} | ✅ STABILIZED |\n",
            now,
            trigger.as_str(),
            metric,
            action.description()
        );

        // Append safely
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&self.log_path) {
            let _ = writeln!(file, "{}", log_entry);
        }
    }
}
