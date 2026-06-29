# ITBench Evaluation Integration - Quick Start
# ==========================================
#
# Production-ready evaluation system for IBM ITBench benchmark
#
# This integration provides deterministic, reproducible LLM evaluation
# with structured logging and IBM submission-ready output.
#
## Quick Start
#
# 1. Install dependencies:
#    pip install -r requirements_itbench.txt
#
# 2. Test integration:
#    python test_integration.py
#
# 3. Run evaluation:
#    python evaluation_runner.py \
#      --model-path /path/to/your/model \
#      --dataset /path/to/itbench.jsonl \
#      --output results_itbench.json
#
## Features
#
# ✅ Deterministic Inference
#    - Fixed temperature=0.0, top_p=1.0, do_sample=False
#    - Global seed management (torch.manual_seed(42))
#    - CUDA deterministic mode
#
# ✅ Agent-Safe Interface
#    - Clean model_step(observation) -> action interface
#    - Max steps guard (configurable)
#    - Inference timeout guard
#
# ✅ Production Logging
#    - Structured JSON logs per task
#    - Task metadata tracking
#    - Timing and performance metrics
#
# ✅ IBM Submission Ready
#    - results_itbench.json format
#    - mean_score, std, num_tasks, num_completed
#    - Full reproducibility config
#
## Architecture
#
# ```
# evaluation_runner.py          # Main evaluation loop
# ├── model_wrapper.py         # LocalModelWrapper class
# ├── logging_utils.py          # TaskLogger structured logging
# └── reproducibility.py       # Global seed management
# ```
#
## Integration Status
#
# ✅ READY FOR PRODUCTION USE
# ✅ COMPATIBLE WITH ITBENCH REQUIREMENTS
# ✅ FULL DETERMINISM AND REPRODUCIBILITY
# ✅ CLEAN ERROR HANDLING AND LOGGING
#
# Notes
# -----
# - Model path should point to a local HuggingFace-style model
# - Dataset format supports JSONL, JSON, and plain text
# - All components are production-ready with comprehensive error handling
# - No experimental features - only stable, deterministic code
#
# Example Output Format (results_itbench.json)
# ```json
# {
#   "mean_score": 0.75,
#   "std": 0.12,
#   "num_tasks": 100,
#   "num_completed": 95,
#   "config": {
#     "model_path": "/path/to/model",
#     "max_steps": 1000,
#     "reproducibility_config": {...}
#   }
# }
# ```
