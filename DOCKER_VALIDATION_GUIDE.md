# SODALS Closed-Box Docker Validation Guide

This document describes the design, execution, and compliance architecture of the isolated (air-gapped) Docker validation environment prepared for İrem Hoca and IBM Research.

## 1. Compliance & Anti-Cheating Architecture

To ensure strict academic and technical integrity, the validation framework enforces the following rules at the container level:

1. **Physical Air-Gap (`--network none`)**:
   - The validation runner is launched with Docker network disabled. It has **zero internet connection** and cannot make external API calls or download weights dynamically.
   - Environment variables `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are baked into the container.

2. **No Logit Steering / Pure Inference (`SODALS_PURE_INFERENCE=1`)**:
   - The engine runs under pure inference mode. This disables active QMEM logit steering adapters during generation.
   - The model must perform raw token generation based purely on standard input prompts and standard local context.

3. **No Ground Truth Leakage (`is_multi` Mitigation)**:
   - The information leakage concern regarding `is_multi` (checking MCQ answer count) has been fully resolved.
   - The MCQ question format selector (`is_multi`) is now derived **dynamically from the dataset name** (checking if `"multi"` is in the filename) rather than looking at the `correct` answers field. The ground truth answer field is never read before inference.

4. **Deterministic Settings**:
   - Decoding temperature is locked to `0.0` (argmax greedy decoding).
   - Global random seed is hardcoded to `42`.

---

## 2. Docker Setup

To avoid baking the large 5GB model file into the Docker build context (which would cause slow builds and disk bloat), we use **Docker Volume Mounting** to map the model and outputs.

### Prerequisite

Ensure the GGUF model and tokenizer are present in your host project directory:
- `models/model.gguf`
- `models/tokenizer.json`

---

## 3. How to Run Validation

You can use the provided automated wrapper scripts or run manual commands.

### Option A: Using Wrapper Scripts (Recommended)

#### Linux / macOS:
Ensure the script is executable and run it:
```bash
chmod +x run_validation_sandbox.sh
./run_validation_sandbox.sh
```

#### Windows (PowerShell or Command Prompt):
Run the batch file:
```cmd
run_validation_sandbox.bat
```

### Option B: Manual Execution

1. **Build the Container**:
   ```bash
   docker build -f Dockerfile.validation -t sodals-validation-sandbox .
   ```

2. **Run the Container Air-Gapped**:
   ```bash
   docker run --network none --gpus all --name sodals-sandbox \
     -v "$(pwd)/run_logs:/app/run_logs" \
     -v "$(pwd)/models:/app/models" \
     sodals-validation-sandbox
   ```

*(If you do not have an NVIDIA GPU, omit the `--gpus all` flag; the Rust engine will automatically fall back to CPU execution without crashing.)*

---

## 4. Running a Blind Evaluation on Custom Datasets

To evaluate the system against new, unseen test datasets, you can mount your local dataset directories over the default benchmark folders inside the container:

```bash
docker run --network none --gpus all --name sodals-sandbox \
  -v "$(pwd)/run_logs:/app/run_logs" \
  -v "$(pwd)/models:/app/models" \
  -v "/path/to/custom_sre:/app/sre" \
  -v "/path/to/custom_finops:/app/finops" \
  -v "/path/to/custom_failuresensoriq:/app/FailureSensorIQ" \
  sodals-validation-sandbox
```

### Dataset Structure Requirements

- **SRE**: A directory containing JSON scenarios (same schema as `/app/sre`).
- **FinOps**: A directory containing JSON scenarios (same schema as `/app/finops`).
- **FailureSensorIQ**: A directory containing `all.jsonl` (single answer MCQ) and `all_multi_answers.jsonl` (multiple answer MCQ) under `failuresensoriq_standard/`.
