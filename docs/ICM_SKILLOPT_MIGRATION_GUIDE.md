# ICM + SkillOpt Hybrid Architecture Migration Guide

**Version**: 1.0  
**Date**: 2026-06-27  
**Status**: Production Ready

---

## Overview

This guide explains the hybrid architecture combining ICM (Interpretable Context Methodology) and SkillOpt (Microsoft's skill optimizer) with the existing SODALS system. The hybrid approach preserves your superior real-time safety and physics-based reasoning while adding filesystem-based context management and validation-gated skill optimization.

---

## Architecture Summary

### What Changed

**Added:**
- **ICM Filesystem Hierarchy**: `sodals_workspace/` with Layer 0-3 context management
- **Skill Optimizer**: `tools/skill_optimizer.py` for validation-gated skill evolution
- **Workspace Loader**: `src/crates/sodals_mcp/src/workflows/workspace_loader.rs` for workspace traversal
- **Benchmark Integration**: `scripts/benchmark/optimize_skill_with_benchmark.py` for optimization with benchmark data

**Preserved (No Changes):**
- **Real-Time Orchestrator**: `sodals_ai_orchestrator.rs` (50ms tick loop, SentinelGuard)
- **Physics-Augmented Reasoning**: PonderingCore, HolographicAttention, SemanticWave
- **Deterministic Guarantees**: Seed-based initialization, RK4 integration
- **QMEM Cartridges**: Compiled knowledge representation for runtime

---

## Workspace Structure

```
sodals_workspace/
├── AGENT.md                    # Layer 0: Root Identity (persona, safety rules)
├── CONTEXT.md                  # Layer 1: Global Context (mission, objectives)
├── 01_perception/              # Stage 1: Multi-modal input processing
│   ├── CONTEXT.md             # Stage-specific input/output contract
│   └── references/            # Layer 3: Static skills
│       ├── vision_cvnn.md
│       ├── iot_streams.md
│       ├── pdf_parser.md
│       └── structured_data.md
├── 02_embedding/              # Stage 2: SemanticWave complex embedding
│   ├── CONTEXT.md
│   └── references/
│       ├── semantic_wave.md
│       ├── fluid_algebra.md
│       ├── fast_math.md
│       └── f4_quantization.md
├── 03_fusion/                 # Stage 3: Cross-modal holographic attention
│   ├── CONTEXT.md
│   └── references/
│       ├── holographic_attention.md
│       ├── multi_mask_fusion.md
│       ├── newton_raphson_phase.md
│       └── cross_modal_correlation.md
├── 04_reasoning/              # Stage 4: PonderingCore + LLM reasoning
│   ├── CONTEXT.md
│   └── references/
│       ├── pondering_core.md
│       ├── llm_engine.md
│       ├── root_cause_analysis.md
│       └── predictive_insights.md
└── 05_compliance/             # Stage 5: Security + regulatory guard
    ├── CONTEXT.md
    └── references/
        ├── outbound_sanitizer.md
        ├── stream_sanitizer.md
        ├── adversarial_rejection.md
        ├── regulatory_compliance.md
        └── logic_nodes_verification.md
```

---

## Migration Steps

### Phase 1: Workspace Setup (Completed)

The workspace structure has been created with all required files. No action needed.

### Phase 2: Integration with Existing System

#### 2.1 Update System Prompts (Optional)

If you want to use workspace context instead of hardcoded prompts:

**Before (hardcoded in code):**
```rust
let system_prompt = "You are QYPTOS, an analytical core...";
```

**After (load from workspace):**
```rust
use sodals_mcp::workflows::WorkspaceLoader;

let loader = WorkspaceLoader::new("sodals_workspace");
let workspace = loader.load()?;
let agent_persona = workspace.agent.persona;
```

#### 2.2 Keep Real-Time Path Unchanged

**Critical**: Do NOT use workspace traversal in the real-time 50ms tick loop. The workspace is for development/optimization only.

**Real-time path (unchanged):**
```rust
// sodals_ai_orchestrator.rs - NO CHANGES
let sentinel = SentinelGuard::new(...);
let logits = [0.1, 0.2, ...]; // Stubbed Qwen output
if let Err(e) = sentinel.validate_ai_output(&logits, &sensor) {
    // Fail-safe handling
}
```

**Development/batch path (new):**
```rust
// Use workspace for offline processing
let executor = WorkspaceExecutor::new(workspace);
let result = executor.execute(input).await?;
```

### Phase 3: Skill Optimization

#### 3.1 Run Baseline Benchmark

```bash
cd scripts/benchmark
python optimize_skill_with_benchmark.py `
    --skill-path ../../sodals_workspace/01_perception/references/vision_cvnn.md `
    --benchmark sre `
    --model-path ../../models/Qwen3-8B-Q4_K_M.gguf `
    --dataset ../../sre `
    --epochs 5 `
    --edit-budget 3
```

#### 3.2 Review Optimized Skill

The optimizer will create:
- `vision_cvnn_optimized.md`: Optimized skill file
- `vision_cvnn_optimized_history.json`: Validation history

#### 3.3 Deploy Optimized Skill

**Manual Review Required:**
1. Review `vision_cvnn_optimized.md`
2. Check edits in `_history.json`
3. If acceptable, replace original:
```bash
mv vision_cvnn_optimized.md vision_cvnn.md
```

### Phase 4: Testing

#### 4.1 Test Workspace Structure

```bash
python scripts/test_workspace_traversal.py --workspace-path sodals_workspace
```

Expected output:
```
✅ AGENT.md found
✅ CONTEXT.md found
✅ 01_perception: 4 reference skills
✅ 02_embedding: 4 reference skills
✅ 03_fusion: 4 reference skills
✅ 04_reasoning: 4 reference skills
✅ 05_compliance: 5 reference skills
```

#### 4.2 Test Workspace Traversal (Rust)

```rust
use sodals_mcp::workflows::{WorkspaceLoader, WorkspaceExecutor};

#[tokio::main]
async fn main() {
    let loader = WorkspaceLoader::new("sodals_workspace");
    let workspace = loader.load().unwrap();
    
    let executor = WorkspaceExecutor::new(workspace);
    let input = json!({"vision": {...}, "iot": {...}});
    let result = executor.execute(input).await.unwrap();
    
    println!("Execution time: {}ms", result.execution_time_ms);
}
```

---

## Usage Patterns

### Pattern 1: Development Mode (Workspace Traversal)

Use workspace traversal for development, debugging, and offline processing.

```python
# Python example (requires PyO3 bindings or Rust binary)
from sodals_mcp import WorkspaceLoader, WorkspaceExecutor

loader = WorkspaceLoader("sodals_workspace")
workspace = loader.load()

executor = WorkspaceExecutor(workspace)
result = executor.execute(input_data)
```

### Pattern 2: Runtime Mode (Compiled QMEM)

Use compiled QMEM cartridges for real-time operation (unchanged).

```rust
// sodals_ai_orchestrator.rs - NO CHANGES
let cartridge = load_qmem("/opt/sodals/qmem/reference.qmem")?;
let logits = model.generate_with_cartridge(&cartridge, &sensor)?;
```

### Pattern 3: Skill Optimization

Use skill optimizer to improve skills based on benchmark data.

```bash
python tools/skill_optimizer.py \
    --skill-path sodals_workspace/01_perception/references/vision_cvnn.md \
    --validation-set benchmark_results/sre_validation.json \
    --epochs 10 \
    --edit-budget 5
```

---

## Key Differences from ICM

### What We Adopted from ICM

1. **Filesystem Hierarchy**: Layer 0-3 context management
2. **Human-Readable Context**: Markdown files for transparency
3. **Stage-Specific Context**: CONTEXT.md per stage
4. **Reference Skills**: Layer 3 static skill library

### What We Did NOT Adopt from ICM

1. **Layer 4 (Working Artifacts)**: Not needed (we use QMEM cartridges)
2. **Human-in-the-Loop Review**: Not in real-time path (safety-critical)
3. **Filesystem as State Machine**: Real-time path uses Rust state machine (faster)
4. **Numbered Folder State**: Real-time path uses atomic state transfer

### Why These Differences?

- **Layer 4**: QMEM cartridges are more efficient for runtime
- **Human Review**: Real-time safety requires automated validation (SentinelGuard)
- **State Machine**: Rust state machine is faster than filesystem traversal
- **Numbered Folders**: Atomic state transfer is more deterministic

---

## Key Differences from SkillOpt

### What We Adopted from SkillOpt

1. **Validation-Gated Optimization**: Accept only if strictly better
2. **Rollout → Reflect → Validate Loop**: Core optimization pattern
3. **Rejected-Edit Buffer**: Track failed edits for analysis
4. **Best-Skill Artifact**: Output optimized skill file

### What We Adapted from SkillOpt

1. **Offline Only**: Optimization runs offline (not in real-time loop)
2. **Benchmark Integration**: Use SRE/FinOps/FSIQ results as validation set
3. **Manual Approval**: Human review required before deployment
4. **Edit Budget**: Limit edits per epoch to prevent overfitting

### Why These Adaptations?

- **Offline Only**: Real-time loop must remain deterministic
- **Benchmark Integration**: Leverage existing benchmark infrastructure
- **Manual Approval**: Safety-critical system requires human oversight
- **Edit Budget**: Prevent skill degradation from aggressive edits

---

## Performance Impact

### Real-Time Path (No Impact)

- **Orchestrator**: Unchanged (50ms tick loop)
- **SentinelGuard**: Unchanged (spectral validation)
- **QMEM Loading**: Unchanged (compiled cartridges)
- **Memory**: Unchanged (<8GB VRAM)

### Development Path (New Overhead)

- **Workspace Loading**: ~50ms (one-time)
- **Stage Execution**: ~100ms per stage (5 stages = ~500ms)
- **Skill Optimization**: ~2-5 minutes per epoch (offline only)

---

## Risk Mitigation

### Risk 1: Workspace Traversal Too Slow for Real-Time

**Mitigation**: Workspace traversal is NOT used in real-time path. Real-time path uses compiled QMEM cartridges (unchanged).

### Risk 2: Skill Optimization Degrades Performance

**Mitigation**: Validation gate ensures only strictly better skills are accepted. Manual approval required before deployment.

### Risk 3: Git Conflicts on Skill Evolution

**Mitigation**: Branching strategy + human review before merge. Optimized skills saved as `_optimized.md` until approved.

### Risk 4: Validation Set Overfitting

**Mitigation**: Hold-out validation set + cross-validation. Edit budget limits aggressive changes.

---

## Rollback Plan

If issues arise with the hybrid architecture:

### Step 1: Disable Workspace Traversal

```rust
// Comment out workspace loader usage
// let loader = WorkspaceLoader::new("sodals_workspace");
// let workspace = loader.load()?;
```

### Step 2: Revert to Hardcoded Prompts

```rust
// Restore original system prompts
let system_prompt = "You are QYPTOS, an analytical core...";
```

### Step 3: Skip Skill Optimization

```bash
# Do not run skill optimizer
# Continue with manual skill updates
```

### Step 4: Delete Workspace (Optional)

```bash
rm -rf sodals_workspace
```

---

## Support and Troubleshooting

### Common Issues

**Issue**: Workspace loader fails to load AGENT.md  
**Solution**: Ensure `sodals_workspace/AGENT.md` exists and is valid markdown

**Issue**: Skill optimizer fails to load validation set  
**Solution**: Ensure benchmark results are in JSON format and contain required fields

**Issue**: Workspace traversal is slow  
**Solution**: This is expected for development path. Use compiled QMEM for runtime

**Issue**: Optimized skill is worse than original  
**Solution**: Validation gate should prevent this. Check validation history for errors

### Getting Help

- **Documentation**: See `docs/` directory for technical specifications
- **Benchmark Results**: See `benchmark_results/` for performance data
- **Run Logs**: See `run_logs/` for execution traces
- **Issues**: Report via Git issues with full context and reproduction steps

---

## Summary

The hybrid architecture successfully integrates ICM's filesystem-based context management and SkillOpt's validation-gated optimization while preserving your superior real-time safety and physics-based reasoning. The migration is non-breaking (real-time path unchanged) and provides clear rollback procedures if needed.

**Key Benefits:**
- ✅ Human-readable context hierarchy
- ✅ Git-traceable skill evolution
- ✅ Validation-gated skill optimization
- ✅ Preserved real-time safety guarantees
- ✅ Preserved physics-based reasoning

**Next Steps:**
1. Test workspace structure: `python scripts/test_workspace_traversal.py`
2. Run skill optimization: `python scripts/benchmark/optimize_skill_with_benchmark.py`
3. Review optimized skills before deployment
4. Monitor benchmark performance for improvements
