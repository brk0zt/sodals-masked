# SODALS Core — Neurosymbolic Reasoning, Soft-MoA, EdgeSSM & Svon-CGW

**SODALS Core** (*Self-Organizing Dual-Adapter Latent System*) is an offline-first neurosymbolic AI architecture built across **Python/PyTorch** (`sft/`), a bare-metal **Rust/CUDA** inference kernel (`src/crates/`), and an isolated **SODALS-OS** micro-VM sandbox (`sodals_os/`), targeting consumer 8 GB GPUs and dual-T4 cloud verification.

---

## arXiv Research Portfolio & Pre-Registered Protocols

All manuscripts, LaTeX sources, reproducible audit tables, and pre-registered statistical protocols live directly in [`arxiv/`](arxiv/) and [`sft/`](sft/):

| # | Paper / Protocol | LaTeX / Markdown Sources | Scope & Verified Status |
| :---: | :--- | :--- | :--- |
| **1** | **Master Reasoning & Epistemic Verification** *(SODALS v2.1)* | [`arxiv/paper1_master_reasoning.tex`](arxiv/paper1_master_reasoning.tex) · [`arxiv/arXiv_SODALS_v2.1_Master_Paper.md`](arxiv/arXiv_SODALS_v2.1_Master_Paper.md) | Clean-125 & Pristine-50 Always-On $G=4$ ablation, Lens-Free Six-Arkhe selective prediction (`89.17% @ 96% cov`, `90.91% @ 88% cov`), next-token Shannon entropy vs. linear probe (`0.7583` / `0.8791` OOD AUROC), symmetrical frozen baseline rescoring (`398/500`), human-audited blind discordant check (`38/39` match, `Pristine30_Q180` extraction fix), and `Python 3.13.7` runtime lock (`139/175`). |
| **2** | **Soft-MoA & The Subspace Interference Law** | [`arxiv/paper2_soft_moa.tex`](arxiv/paper2_soft_moa.tex) | Empirical proof of cross-domain LoRA subspace interference ($A_1 A_2^\top \neq 0$) and inference-time continuous Soft LoRA Blending (`138/170`, `81.18%`) with Gaussian Phase Routing ($\tau = 0.004913$) and auxiliary identity-adapter isolation. |
| **3** | **EdgeSSM: Deterministic Rust/CUDA State-Space Inference for Industrial CPS** *(IEEE TII)* | [`arxiv/paper3_edgessm_ieee_tii.tex`](arxiv/paper3_edgessm_ieee_tii.tex) | Bare-metal Rust/CUDA GGUF hybrid SSM/Attention runtime (`28.5 ms/tok`, `0.22 ms` host cache reset + `12.5–46.0 ms` chunk re-prefill), $O(N \log \text{deg}_{\max})$ relational GNN attention mask, and `NogoodClauseCache` backtracking. |
| **4** | **Svon-CGW & Epistemic Mid-Turn Thought-Freezing** *(Epoch II–III)* | [`arxiv/paper4_svon_cgw_epistemic_freeze.tex`](arxiv/paper4_svon_cgw_epistemic_freeze.tex) · [`arxiv/arXiv_SODALS_Paper4_Svon_CGW_Epistemic.md`](arxiv/arXiv_SODALS_Paper4_Svon_CGW_Epistemic.md) | Tweezer + AlphaEdit Null-Space PGD identity surgery (`mlp.down_proj`, v13: `20/20` world preserved, `KL=0.00292`, `0` repetition collapse; identity gate `>=27/30` open for v14), 24 Holomorphic CVNN-DeltaNet + 8 Two-Hop Sheaf-GNN layers (Exact Submanifold Theorem $<10^{-6}$), Kuramoto phase synchronization, and CPU-offloaded `DynamicCache` mid-turn thought-freezing. |
| **—** | **Pre-Registered Confirmatory Plan (`Q201..Q500`)** | [`arxiv/Q201_Q500_CONFIRMATORY_ANALYSIS_PLAN.md`](arxiv/Q201_Q500_CONFIRMATORY_ANALYSIS_PLAN.md) | Dated & SHA-256-locked single-shot confirmatory protocol for the untouched `Q201..Q500` (`N=300`) reserve (exact two-sided McNemar $\alpha = 0.05$, movement-conditioned `GRPO-1`/`GRPO-2` prerequisites, hard stopping rule). |
| **—** | **Portfolio Strategy & Blind Human Audit** | [`arxiv/SODALS_Three_Paper_Portfolio_Strategy.md`](arxiv/SODALS_Three_Paper_Portfolio_Strategy.md) · [`sft/BLIND_HUMAN_DISCORDANT_AUDIT.md`](sft/BLIND_HUMAN_DISCORDANT_AUDIT.md) | Cross-paper separation rules and full 13-question (39-prediction) blind human-verified discordant audit. |

---

## Architecture Overview

- **`sft/` — Python Neurosymbolic Serving, Training & Evaluation:**
  - `sodals_serve.py` & `sodals_chat.py`: Offline local chat, OpenAI/Ollama/Anthropic-compatible HTTP server (localhost-only CORS, LFI-hardened), multi-turn sliding context, offline local code agent (`local_code_agent.py`), document reader (`document_reader.py`), and Epistemic Thought-Freezing v2 (`epistemic_clarify_gate.py`, `clarification_freeze.py`).
  - `cst_symbolic_verifier.py` & `arkhe_neurosymbolic_pipeline.py`: SHA-256-locked Epoch I `NormalizedMathMatcher`, `SymbolicStepChecker`, and Six Cognitive Arkhes pipeline.
  - `svon_alphaedit_surgeon.py`, `svon_cvnn_deltanet.py`, `svon_sheaf_gnn_attention.py`, `svon_kuramoto_equilibrium.py`, `train_svon_cgw_architecture.py`: Epoch II/III identity surgery and holomorphic wave-graph modules.
- **`src/crates/` — Bare-Metal Rust/CUDA Engine:**
  - `sodals_neuro`, `sodals_net`, `sodals_kernel`, `sodals_crypto`, `sodals_industrial`, `sodals_common`: Quantized GGUF inference, `.qmem` loader, fail-closed industrial sentinel guard, and optional `cuda` / `python` Cargo features for CPU-only CI testing.
- **`sodals_os/` — Isolated Alpine Linux Micro-VM Sandbox:**
  - `vm_manager.py`, `iso_builder.py`, `guest/os_daemon.py`: QEMU/WHPX isolated guest OS with redirect/DNS-rebinding SSRF protection, `ip6tables`/`iptables` egress firewall, strict `note_id` validation, one-time dashboard bootstrap auth, and untrusted web-content boundaries.

---

## Quick Verification (CPU-Only, Zero GPU Required)

```powershell
# Run the full 98-test production unit & integration suite on CPU
$env:CUDA_VISIBLE_DEVICES=""
python -B -m unittest discover -s sft/tests/prod_suite -p "test_*.py" -v
```

---

## Security & Repository Hygiene

- **Zero Secrets / Zero Credentials:** `.env`, API tokens, SSH/TLS keys, `kaggle.json`, and `sodals_os/secrets/` are strictly ignored via [`.gitignore`](.gitignore).
- **Zero Heavy Binaries:** Model weights (`*.safetensors`, `*.gguf`, `*.qmem`, `*.pt`), checkpoints, wheels (`*.whl`), and VM disk images (`*.qcow2`, `*.iso`) are excluded from Git tracking.
- **Serving Hardening:** Localhost-only CORS, HTTP local-file-inclusion (LFI) disabled by default, SSRF redirect & private-IP blocking in `sodals_os`, and `<untrusted_web_content>` prompt-injection boundaries.
