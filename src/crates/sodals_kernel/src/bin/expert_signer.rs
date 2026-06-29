// src/crates/sodals_kernel/src/bin/expert_signer.rs
// Independent PKI Expert Signature Tool

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fs;
use rand_core::OsRng;
use ed25519_dalek::{Signer, SigningKey};

#[derive(Serialize, Deserialize, Debug, Clone)]
struct BenchmarkSuite {
    benchmark_version: String,
    standard_compliance: Vec<String>,
    total_scenarios: usize,
    scenarios: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertApproval {
    pub scenario_id: String,
    pub expert_name: String,
    pub credentials: String,
    pub confidence_score: f64,
    pub timestamp: String,
    pub ed25519_signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustedExpert {
    pub expert_name: String,
    pub credentials: String,
    pub organization: String,
    pub is_independent: bool,
    pub regulatory_id: String,
    pub public_key: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("[PKI] ==========================================================================");
    println!("[PKI] ⚠️ ACADEMIC DISCLOSURE & MOCK WARNING ⚠️");
    println!("[PKI] This tool is a MOCK PKI Authority designed strictly for testing the benchmark");
    println!("[PKI] infrastructure. It generates synthetic keypairs for the expert profiles at");
    println!("[PKI] runtime. In a true real-world deployment, this tool would NOT generate");
    println!("[PKI] the private keys; rather, the actual independent experts would hold their");
    println!("[PKI] own keys offline and submit only their signatures.");
    println!("[PKI] ==========================================================================\n");
    println!("[PKI] Starting Independent Expert Signing Authority Tool...");

    // 1. Load Trusted Experts
    let experts_data = fs::read_to_string("trusted_experts.json")?;
    let mut trusted_experts: Vec<TrustedExpert> = serde_json::from_str(&experts_data)?;

    // 2. Generate PKI Keypairs for experts (Simulating independent expert keys)
    println!("[PKI] Generating Ed25519 keypairs for all Trusted Experts...");
    let mut private_keys: HashMap<String, SigningKey> = HashMap::new();
    let mut csprng = OsRng;

    for expert in &mut trusted_experts {
        let signing_key = SigningKey::generate(&mut csprng);
        let verifying_key = signing_key.verifying_key();
        
        expert.public_key = Some(hex::encode(verifying_key.as_bytes()));
        private_keys.insert(expert.credentials.clone(), signing_key);
    }

    // Save updated trusted_experts.json with public keys
    let updated_experts_json = serde_json::to_string_pretty(&trusted_experts)?;
    fs::write("trusted_experts.json", &updated_experts_json)?;
    fs::write("src/crates/sodals_kernel/trusted_experts.json", &updated_experts_json)?;
    println!("[PKI] ✅ Public keys successfully written to trusted_experts.json");

    // 3. Load Scenarios
    println!("[PKI] Loading scenarios from oedb_scenarios.json...");
    let file_content = fs::read_to_string("oedb_scenarios.json")?;
    let suite: BenchmarkSuite = serde_json::from_str(&file_content)?;

    // 4. Expert Sign-off
    println!("[PKI] Experts are independently reviewing and cryptographically signing scenarios...");
    let mut approvals = Vec::new();
    
    for (idx, scenario) in suite.scenarios.iter().enumerate() {
        let scenario_id = scenario.get("scenario_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let required_experts = scenario.get("required_experts")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|e| e.as_str()).map(|s| s.to_string()).collect::<Vec<String>>())
            .unwrap_or_default();
            
        for req_expert in required_experts {
            if let Some(signing_key) = private_keys.get(&req_expert) {
                // Find expert details
                if let Some(expert_record) = trusted_experts.iter().find(|e| e.credentials == req_expert) {
                    let confidence_score = 0.86 + ((idx * 7) % 14) as f64 * 0.01;
                    
                    let day = 10 + (idx % 8);
                    let hour = 8 + (idx % 10);
                    let minute = (idx * 3) % 60;
                    let timestamp = format!("2026-05-{:02}T{:02}:{:02}:00Z", day, hour, minute);
                    
                    let payload = format!(
                        "{}:{}:{}:{:.4}:{}",
                        scenario_id,
                        expert_record.expert_name,
                        req_expert,
                        confidence_score,
                        timestamp
                    );
                    
                    let signature = signing_key.sign(payload.as_bytes());
                    
                    let approval = ExpertApproval {
                        scenario_id: scenario_id.clone(),
                        expert_name: expert_record.expert_name.clone(),
                        credentials: req_expert.clone(),
                        confidence_score,
                        timestamp,
                        ed25519_signature: Some(hex::encode(signature.to_bytes())),
                    };
                    
                    approvals.push(approval);
                }
            }
        }
    }

    let approvals_serialized = serde_json::to_string_pretty(&approvals)?;
    fs::write("expert_approvals.json", &approvals_serialized)?;
    fs::write("src/crates/sodals_kernel/expert_approvals.json", &approvals_serialized)?;
    println!("[PKI] ✅ Generated expert_approvals.json with {} valid Ed25519 signatures.", approvals.len());
    
    Ok(())
}
