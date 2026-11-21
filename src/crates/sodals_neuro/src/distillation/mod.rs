// src/distillation/mod.rs
//
// Selective Distillation Framework for SODALS
//
// Purpose: Distill general reasoning patterns (Arkhe stage transitions, goal-precondition satisfaction chains)
// into trainable LoRA adapters while keeping atomic facts in QMEM cartridges.
//
// Key safeguards:
//   1. Synthetic QMEM curriculum (no web-scraping contamination)
//   2. Dual-category training: Analogical patterns AND STRIPS/PDDL goal-precondition satisfaction
//   3. Parameter drift regularization (L2 delta penalty + KL divergence penalty against frozen base model)
//   4. Held-out axiom generalization testing + general capability regression suite

pub mod curriculum;
pub mod lora_adapter;
pub mod trainer;
pub mod evaluator;

pub use curriculum::{CurriculumGenerator, SyntheticReasoningSample, ReasoningCategory};
pub use lora_adapter::{LoRaAdapterConfig, SplitComplexLoRaAdapter};
pub use trainer::{DistillationTrainer, TrainerConfig};
pub use evaluator::{EvaluationSuite, GeneralizationReport, RegressionReport, MathematicsGeneralizationReport};
