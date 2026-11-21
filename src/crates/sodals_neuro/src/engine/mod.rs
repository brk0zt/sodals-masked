// src/engine/mod.rs

pub mod chimera_router;
pub mod logits_fusion;
pub mod oracle_worker;
pub mod llm_engine;
pub mod generation_loop;
pub mod security;
pub mod mcp_client;
pub mod pondering_core;
pub mod bci_ipc;
pub mod autonomy_gate;
pub mod mcp_hdc_bridge;
pub mod lsp_watcher;
pub mod fuzzy_parser;
pub mod chronos_weaver;
pub mod bci_mux;
pub mod bci_translator_node;
pub mod qmem_compiler;
pub mod silent_reasoning;

pub use llm_engine::{LlmEngine, PipelineAttempt, AgenticPipeline, DependencyGuard, LogicDefender, PhaseResult};
pub use generation_loop::generate_stream;
pub use chimera_router::{NeuroConfig, TaskPlan, ANALOGICAL_REASONING_PROTOCOL};
pub use logits_fusion::{PhysicsGuidedLogitsProcessor, SyntaxPerspective, CausalPerspective, GeometerPerspective, LogitPerspective, YieldedToken, RawToken};
pub use oracle_worker::{OracleMessage, OracleHandle, PrecomputedMask, HspOracle, HSP_DIMENSIONS, SPARSITY_K};
pub use pondering_core::PonderingCore;
pub use autonomy_gate::{AutonomyGate, AutonomyLevel, AutonomyVerdict, HaltInterrupt};
pub use mcp_hdc_bridge::{McpSensoryCortex, DefaultMcpSensoryCortex, PhasorBoundaryCondition, BoundaryConditionBatch};
pub use lsp_watcher::{DiagnosticWatcher, DiagnosticEvent, PhysicsImpedance};
pub use fuzzy_parser::{AstHunter, AstNode, AstNodeKind};
pub use chronos_weaver::{ChronosWeaver, AnchorId, TemporalAnchor, TimeReversalResult, ReversalCompleteness};
pub use bci_mux::BciStreamer;
pub use bci_translator_node::BciTranslator;
pub use qmem_compiler::{QmemCrystal, QmemFactory, QMEM_MAGIC, QMEM_VERSION};

pub mod sdr_core;
pub mod ram_kernel;
pub mod semantic_navigator;
pub mod semg_phoneme_hunter;
pub mod bci_logits_processor;
pub mod industrial_safety_logits;
pub mod paca_gate;

pub use bci_logits_processor::BciLogitsProcessor;
pub use industrial_safety_logits::IndustrialSafetyLogitsProcessor;
pub use paca_gate::{ZScoreGate, ZScoreResult, NewtonRaphsonAligner, PacaMaskSet};
