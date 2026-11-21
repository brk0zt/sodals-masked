pub mod physics_neuron;
pub mod fast_math;
pub mod fluid_algebra;
pub mod llm_core;
pub mod neuro_kernel;
pub mod bridge;
pub mod interface_layer;
pub mod semantic_network;
#[cfg(feature = "experimental")]
pub mod spectral_memory;
#[cfg(feature = "experimental")]
pub mod file_system;
pub mod semantic_parser;
pub mod train_projector;
pub mod engine;
pub mod sodals_alloc;
pub mod global_gaze;
pub mod complex_tensor;
pub mod complex_layers;
pub mod complex_activations;
pub mod compiler;
pub mod vision;
pub mod nvme_direct;
pub mod qmem_compiler;
pub mod arkhe_pipeline;
pub mod qmem_builder;
pub mod distillation;
#[cfg(feature = "experimental")]
pub mod arc_agi_solver;
pub mod arc_environment;
pub mod mcts_exploration;
pub mod agent_loop;
pub mod resonance_chunker;
pub mod retrieval_engine;
pub mod context_integration_test;
#[cfg(test)]
pub mod end_to_end_acceptance_test;

// BrainFlow Integration - Zero-copy biosignal acquisition (only with brainflow feature)
#[cfg(feature = "brainflow")]
pub mod brainflow_bridge;
#[cfg(feature = "brainflow")]
pub mod signal_processor;

#[cfg(all(feature = "cuda", feature = "experimental"))]
pub mod gpu_math;

pub mod models {
    pub mod qwen_custom;
    pub mod qwen_ssm;
}

pub mod quant;
pub mod qmem_loader;
pub mod phase45_gpr_qmem_router;
#[cfg(feature = "experimental")]
pub mod benchmarks_phase45;
#[cfg(feature = "experimental")]
pub mod antigravity;
pub mod jlens_offline;
pub mod jlens_readout;
pub mod jlens_diagnostics;

pub mod gpu_ops;
#[cfg(feature = "experimental")]
pub mod tui;
// NOTE: socket_actor, protocol and web_search REMOVED from here.















