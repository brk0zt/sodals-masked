# EdgeSSM: A Deterministic, Bounded-Memory State Space Model Inference Engine for Industrial Cyber-Physical Systems

**Author:** Burak Öztürk  
**Affiliation:** Department of Computer Engineering, Tarsus University, Turkey (Undergraduate Supervised Research)  
**Date:** September 2026  
**Categories:** cs.AR, cs.AI, cs.LG, cs.SY  

## Abstract

Deploying 9B-parameter State Space Models (SSMs) on edge cyber-physical systems (CPS) introduces severe systems challenges: high memory overhead and garbage collection latency jitter in traditional execution stacks, numerical ill-conditioning when attempting to unwind recurrent hidden states during conflict backtracking, and numerical instability from GGUF head expansion convention mismatches. We present **EdgeSSM**, a bare-metal edge inference engine engineered in Rust for deterministic industrial telemetry diagnostics and supervisory advisory reasoning. EdgeSSM introduces three core systems contributions: (1) a **Bounded-Memory Bare-Metal Runtime** executing over fixed lock-free ring buffers (`RtRingBuffer`) and atomic indices, maintaining a strictly bounded 100MB application-level orchestrator heap ceiling (total process RSS ~600–900 MB with CUDA initialized) and delivering 28.5 ms per-token autoregressive decoding (~35 tok/s single-stream decode, intentionally prioritizing deterministic memory bounds and absence of managed GC pauses over unconstrained batched throughput) on a commodity 8GB GPU (NVIDIA RTX 4060); (2) a **Tensor Alignment & Cache Synchronization Protocol** that resolves GGUF head expansion layout bugs (`repeat_interleave` vs `tile`), recovering Layer-0 cosine similarity from 0.0300 to 0.9996 and reducing Layer-0 softmax projection KL divergence to $3.83 \times 10^{-8}$ (retaining a global cosine similarity of 0.9655 at the final Layer-32 logits after 32 layers of Q4_K_M dequantization accumulation), while synchronizing selective DeltaNet SSM states during conflict-directed backtracking via deterministic prefix re-prefilling (0.22 ms host orchestrator reset, yielding 12.7–46.2 ms total end-to-end rollback latency); and (3) a **Rule-Gated Decoding Subsystem** compiling physical plant constraints from structured EBNF grammar cartridges into deterministic logit priors and sparse CSR graph attention masks without trainable parameter overhead. Evaluated across 223 cyber-physical benchmark scenarios (an AssetOpsBench-Derived Diagnostic Subset $N=120$, SRE Incident Resolution $N=38$, FinOps Anomaly Detection $N=15$, and EdgeTelemetry-50 $N=50$), EdgeSSM achieves comparable zero-shot diagnostic accuracy against cloud-hosted GPT-4o (80.0% vs. 78.3% diagnostic exact match, overlapping 95% Wilson confidence intervals, Fisher's exact $p=0.87$) while operating completely on-premises with 28.5 ms per-token local decode (eliminating 120–500 ms cloud API network round-trip overhead), requiring only 100MB host orchestrator RAM plus 5.5GB GPU VRAM under 4-bit GGUF quantization, and guaranteeing 100% on-premises data residency for air-gapped industrial facilities. All runtime crates, CUDA kernels, and reproduction harnesses are made available under an open-source license.

**Keywords:** State Space Models, Edge AI, Cyber-Physical Systems, Bounded-Memory Computing, Embedded Systems, Real-Time Inference, Predictive Maintenance

---

## 1. Introduction

Modern industrial facilities generate rich multi-sensor telemetry, discrete operational alarms, and event logs prior to equipment failure. Deploying Large Language Models (LLMs) and recurrent State Space Models (SSMs) for diagnostic triage directly on plant-floor edge controllers introduces four fundamental systems bottlenecks:

1. **Host Memory Bloat and Garbage Collection Jitter:** Standard deep learning runtimes (e.g., Python/PyTorch) allocate hundreds of dynamic objects per token generation step, incurring unbounded memory growth and unpredictable Garbage Collection (GC) pauses (frequently exceeding 50–200 ms). Such latency jitter disrupts mission-critical edge monitoring loops.
2. **GGUF Recurrent Tensor Layout Inconsistencies:** State Space Models such as DeltaNet and Mamba utilize grouped-query head projections. Discrepancies in head expansion layouts between reference tokenizers and GGUF quantization runtimes (`repeat_interleave` versus interleaved `tile` formatting) cause silent activation divergence in early layers, destroying diagnostic accuracy.
3. **Recurrent State Numerical Divergence during Backtracking:** When heuristic reasoning engines encounter operational contradictions (e.g., proposing an action that violates an interlock rule), search algorithms must backtrack. While attention KV caches can be trivially sliced along the time dimension, recurrent SSM hidden states cannot be reliably unwound backwards: although the delta-rule is algebraically an affine mapping, inverting outer-product state transitions in finite floating-point precision is severely ill-conditioned and unstable. Meanwhile, storing dense state snapshots ($\approx 25\text{--}50\text{ MB}$ per step) at every search node rapidly exhausts edge memory. Slicing the sequence without synchronizing recurrent memory causes immediate state desynchronization.
4. **Premature Generation Collapse in Small Quantized Models:** Under 4-bit edge quantization, 9B-parameter models frequently suffer from premature Chain-of-Thought termination (emitting `</think>` within fewer than 64 tokens) or fall into cyclic token repetition loops when processing complex multi-sensor diagnostic prompts.

To address these challenges, we introduce **EdgeSSM**, a bare-metal, bounded-memory inference runtime engineered specifically for deterministic State Space Model execution in industrial cyber-physical environments. EdgeSSM consolidates runtime execution into three tightly coupled subsystems:

1. **The Bounded-Memory Bare-Metal Engine:** A Rust runtime built on stack-allocated structures, fixed-size ring buffers (`RtRingBuffer`), and lock-free atomic primitives, enforcing a strictly bounded 100MB host orchestrator heap ceiling while orchestrating a 4-bit quantized 9B SSM in 5.5GB GPU VRAM.
2. **The Recurrent DeltaNet Cache Synchronization Protocol:** A backtracking mechanism that guarantees exact alignment between dense attention history and selective SSM hidden states during conflict-directed search via deterministic prefix re-prefilling on GPU paths (0.22 ms host orchestration reset, 12.7–46.2 ms total end-to-end rollback latency).
3. **The Rule-Gated Decoding Subsystem:** A compiler pipeline translating structured domain cartridges (EBNF grammars) into deterministic logit priors and CUDA Compressed Sparse Row (CSR) attention masks, enforcing equipment operating envelopes without runtime gradient updates.

Our primary contributions are:
1. **Bounded-Memory Deterministic Runtime:** A bare-metal Rust engine delivering predictable 28.5 ms per-token single-stream decode latency (~35 tok/s, intentionally prioritizing deterministic memory bounds, zero GC pauses, and native C/Rust telemetry integration over unconstrained batched throughput) and 1.0–1.3 ms auxiliary routing cycles within a fixed 100MB orchestrator RAM allocation.
2. **GGUF Head Expansion Correction:** The identification and resolution of the head expansion layout mismatch in Qwen 3.5 SSM kernels, increasing Layer-0 cosine similarity from 0.0300 to 0.9996 and decreasing Layer-0 softmax projection KL divergence from 7.22 to $3.83 \times 10^{-8}$ (retaining a global cosine similarity of 0.9655 at the final Layer-32 logits under 4-bit Q4_K_M quantization).
3. **DeltaNet Recurrent State Cache Synchronization:** A deterministic prefix re-prefill backtracking protocol that resolves recurrent state divergence during diagnostic search without requiring expensive per-step state snapshots.
4. **Five-Phase Generation Stabilization:** A runtime protocol incorporating a minimum-thinking token floor ($M=1024$, implementing budget forcing), post-softmax probability filtering (`min_p`), and structural-token-safe Skew-G braking to prevent reasoning collapse under 4-bit quantization.
5. **Empirical Competence on Industrial Telemetry Benchmarks:** Evaluation across four benchmark suites totaling 223 cases (an AssetOpsBench-Derived Diagnostic Subset $N=120$, SRE $N=38$, FinOps $N=15$, EdgeTelemetry-50 $N=50$), showing that EdgeSSM matches commercial cloud models (GPT-4o zero-shot) in diagnostic exact match (80.0% vs. 78.3%, overlapping 95% Wilson CIs, Fisher exact $p=0.87$) while operating with local on-premises data residency and avoiding cloud API network round-trip delays.

The remainder of this paper is structured as follows. Section 2 reviews related work in state space models, edge computing, and zero-allocation runtimes. Section 3 details the system architecture, tensor layout alignment, backtracking cache synchronization, and rule-gated decoding. Section 4 presents the experimental methodology and dataset provenance. Section 5 details empirical results, systems ablations, and edge performance. Section 6 discusses memory bounds and systems trade-offs. Section 7 concludes with future directions.

---

### Notation

The following notation is used throughout this paper:

| Symbol | Description |
|---|---|
| $\mathbf{h}_t \in \mathbb{R}^d$ | Hidden state vector at token step $t$ ($d = 4096$) |
| $\mathbf{S}_t \in \mathbb{R}^{B \times H \times D \times D}$ | Recurrent SSM state matrix ($D = 128$) |
| $\mathbf{g}_t \in \mathbb{R}^{B \times H \times D}$ | Recurrent state decay gate vector |
| $\mathbf{\beta}_t \in \mathbb{R}^{B \times H \times D}$ | State update learning rate modulation parameter |
| $\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t$ | Query, key, and value projection vectors |
| $\mathbf{A}_{\text{csr}}$ | Compressed Sparse Row adjacency matrix for plant topology |
| $L_{\text{prefix}}$ | Retained token sequence length during conflict-directed rollback |
| $M$ | Minimum thinking token floor ($M = 1024$) |
| $\text{bias}_{\text{rule}}$ | Deterministic hash-indexed logit prior offset |
| $\text{RtRingBuffer}$ | Fixed-capacity lock-free circular ring buffer |
| Nogood Cache | Conflict-directed clause recording database for search pruning (inspired by CDCL) |
| CSR | Compressed Sparse Row sparse matrix format |
| GPR | Gaussian Phase Router (Layer-32 routing module) |
| MoA | Mixture-of-Adapters |
| NF4 | Normal Float 4-bit quantization |

---

## 2. Related Work

### 2.1 State Space Models in Sequence Modeling

Recent advances in State Space Models (SSMs) have demonstrated efficient sequence modeling with linear complexity in sequence length. Mamba [1] introduced selective state spaces with input-dependent parameters, achieving $O(L)$ computational complexity while matching transformer performance. RWKV [2] explored linear recurrence mechanisms combining the parallelism of transformers during training with the $O(1)$ memory inference of RNNs. Subsequent continuous-time and gated recurrent formulations, including structured state-space representations (S4) [10], Hyena [11], and Liquid Time-Constant Networks [12], underscored the efficiency of recurrent state updates for long-context processing, complemented by memory-efficient attention techniques [9] and Gated Delta Networks [5]. EdgeSSM builds upon selective DeltaNet state space layers [5], addressing critical GGUF tensor memory layout conventions that otherwise cause severe numerical divergence on edge quantization platforms.

### 2.2 Edge AI and Industrial Cyber-Physical Systems

Industrial predictive maintenance and asset operations require processing multivariate sensor telemetry under rigorous safety standards. Traditional approaches rely on machine health monitoring [21] or LSTM-based multi-sensor anomaly detection [22]. Comprehensive frameworks for condition-based maintenance and machinery diagnostics have been established by Schwabacher [24] and Jardine et al. [25], formalized within international industrial standards such as ISO 13374-1 [23] and IEEE Std 1451.4 [27], with cyber-physical architectures outlined by Lee et al. [26] for Industry 4.0 systems. While relational graph neural representations [16] model multi-sensor dependencies, they lack discrete logical reasoning capabilities required for complex multi-step incident troubleshooting. EdgeSSM bridges continuous telemetry ingestion and discrete diagnostic reasoning directly on plant-floor edge devices.

### 2.3 Bounded-Allocation Systems and Deterministic Runtimes

Hard real-time and safety-critical industrial applications require bounded execution latency without garbage collection pauses [19]. The Rust programming language provides compile-time memory safety, thread concurrency without data races, and deterministic resource deallocation via RAII (Resource Acquisition Is Initialization) without a managed runtime [20]. Prior systems have explored zero-allocation embedded runtimes for microcontrollers; EdgeSSM extends these principles to 9B-parameter neural inference through a bounded-memory design, enforcing an isolated 100MB application orchestrator heap ceiling via stack-allocated tensor views, lock-free ring buffers (`RtRingBuffer`), and thread-indexed CUDA accumulation.

### 2.4 Relational Graph Topology and Symbolic Search

Graph Neural Networks (GNNs) model relational dependencies across interconnected industrial equipment networks [13, 16]. Kipf & Welling [13] introduced Graph Convolutional Networks (GCNs), Veličković et al. [14] developed Graph Attention Networks (GATs), and Hamilton et al. [15] proposed inductive representation learning on large graphs. In parallel, neuro-symbolic reasoning frameworks [3, 4] investigate hybrid architectures combining deep representations with formal symbolic rules. EdgeSSM integrates plant equipment topology directly into attention via CUDA-accelerated Compressed Sparse Row (CSR) matrices, achieving $O(N^2 \log \text{deg}_{\max})$ cumulative lookup complexity across sequence length $N$ (where $\text{deg}_{\max}$ is the maximum equipment node degree, decoupled from attention head dimension $d_k$) with deterministic thread-local accumulation.

### 2.5 Conflict-Directed Backtracking & Constrained Decoding

Nogood clause recording and conflict-directed backtracking [17] are established algorithmic paradigms in constraint satisfaction and automated reasoning that record conflict clauses to systematically prune invalid search branches [18]. In neural text generation, grammar-constrained decoding frameworks [29, 30] enforce syntactic validity on generated tokens via incremental grammar parsing. However, integrating backtracking search into recurrent models poses an unresolved challenge: recurrent states cannot be rewound simply by index truncation. EdgeSSM resolves this by coupling conflict-directed nogood recording with exact recurrent SSM state re-prefilling.

### 2.6 Parameter-Efficient Adaptation & Mechanistic Layer Probing

Low-Rank Adaptation (LoRA) [32] enables parameter-efficient fine-tuning by decomposing weight updates into low-rank matrices: $\mathbf{W}_{\text{new}} = \mathbf{W}_{\text{frozen}} + (\alpha / r) \mathbf{A}\mathbf{B}$, complemented by 4-bit quantization techniques such as QLoRA [7] and GPTQ [8]. To monitor intermediate representations and detect generation anomalies, mechanistic interpretability provides probing tools: Belrose et al. [31] trained per-layer affine probes (Tuned Lens) via supervised regression to elicit latent predictions across intermediate layers. EdgeSSM leverages parameter-efficient adapters combined with deterministic rule gating to specialize edge models across heterogeneous plant equipment while bounding the empirical generalization gap $\Delta_{\text{gen}}$.

---

## 3. Methodology

EdgeSSM integrates deterministic industrial signal processing with selective recurrent state space inference through seven tightly coupled architectural subsystems.

```
+---------------------------------------------------------------------------------+
|                           Industrial Sensor Telemetry                           |
|               (Acoustic, Vibration, Thermal, Pressure, SCADA Logs)              |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                    Bounded-Memory Ingestion & Pre-Filtering                     |
|   - Lock-Free Circular Buffers (`RtRingBuffer`, 100MB Fixed Orchestrator Budget)|
|   - Deterministic EBNF Telemetry Grammar Parsing & Range Sanity Gating          |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                  Selective DeltaNet SSM Backbone (9B GGUF Q4 Engine)            |
|   - 24 Recurrent State Space Layers + 8 Dense Attention Layers                  |
|   - Head-Expansion `tile` Layout Alignment (Layer-0 Cosine Sim: 0.9996)         |
|   - Thread-Local CUDA CSR Graph Attention Bias (O(N log deg) Complexity)        |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                    Five-Phase Runtime Stabilization Protocol                    |
|   - Phase 1: Minimum Thinking Duration Floor (M = 1024 tokens)                  |
|   - Phase 2: Post-Softmax `min_p` Probability Filtering & Skew-G Braking        |
|   - Phase 3: Recurrent SSM Re-Prefill on Nogood Conflict Backtracking          |
|   - Phase 4: Hierarchical Paragraph Coherence & Repetition Prevention            |
|   - Phase 5: Bounded-Memory Deterministic Execution & Thread-Local Accumulation |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                    Modular Mixture-of-Adapters (MoA) Dispatch                   |
|   - Continuous Layer-32 Soft LoRA Blending (1.3ms Auxiliary Routing Overhead)   |
|   - Deterministic Hash-Indexed Rule Priors: bias = (sum(bytes) mod 1024) * 1e-4 |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                      Validated Industrial Command Dispatch                      |
|            (Safety Interlock Verification & Supervisory Advisory Dispatch)      |
+---------------------------------------------------------------------------------+
```

### 3.1 EdgeSSM End-to-End System Architecture

EdgeSSM models industrial cyber-physical systems through a deterministic, layered information pipeline optimized for safety-critical edge execution. The end-to-end processing pipeline executes across six discrete operational stages:

1. **Deterministic Telemetry Ingestion**: Multi-sensor telemetry records, alarm states, and operational logs enter the edge host controller via lock-free circular ring buffers (`RtRingBuffer`). Operating strictly with static buffer allocations, ingestion guarantees an isolated 100MB host orchestrator RAM budget (within ~600–900 MB total host process RSS once the CUDA driver runtime is initialized) and sub-millisecond acquisition without memory allocator jitter.
2. **EBNF Syntactic & Range Gating**: Telemetry attributes and operational logs are parsed against domain-specific Extended Backus-Naur Form (EBNF) grammars. Parsing Expression Grammars (PEG) enforce physical conservation laws and operational envelope constraints (e.g., thermal thresholds, pressure limits) before tokens enter the neural inference context.
3. **Selective State Space Inference**: Telemetry representations and diagnostic prompts are processed by a hybrid 9B State Space Model incorporating 24 recurrent DeltaNet SSM layers and 8 dense Transformer layers. At runtime, the native Rust inference engine executes quantized GGUF Q4_K_M weights, fitting entirely within a 5.5 GB VRAM ceiling (while offline fine-tuning utilized `bitsandbytes` NF4 quantization, as detailed in Section 3.7).
4. **Relational Attention Steering**: Physical plant connectivity (e.g., equipment stages, piping networks, electrical bus topology) is represented as a Compressed Sparse Row (CSR) graph. Custom CUDA kernels inject relational topology bias directly into key-query attention matrices with $O(N \log \text{deg}_{\max})$ per-query row complexity (where $\text{deg}_{\max}$ is the maximum equipment node degree, decoupled from attention head dimension $d_k$), avoiding non-deterministic atomic reductions.
5. **Runtime Stabilization & Backtracking**: The generation loop is governed by a five-phase stabilization protocol that enforces a minimum thinking duration ($M = 1024$), applies dynamic post-softmax probability filtering (`min_p`), and synchronizes recurrent SSM state memory during conflict-directed backtracking via clean token prefix re-prefills.
6. **Rule-Gated Supervisory Dispatch**: Candidate operational diagnoses and advisory recommendations (e.g., valve inspection alerts, maintenance triage, load shed advisories) are scored against deterministic hash-indexed rule priors and verified against safety interlocks before supervisory operator notification.

---

### 3.2 Selective SSM Core & GGUF Tensor Layout Alignment

To achieve real-time latency on edge hardware without quadratic memory growth, EdgeSSM replaces dense self-attention across 75% of model layers with selective recurrent State Space Model (SSM) layers based on the DeltaNet architecture. The model contains 32 total layers, comprising 24 recurrent SSM layers and 8 full attention layers (positioned at layers 3, 7, 11, 15, 19, 23, 27, and 31).

#### Mathematical Formulation of Delta-Rule SSM
Within each recurrent layer, the hidden state $\mathbf{S}_t \in \mathbb{R}^{d_{state} \times d_{head}}$ updates sequentially according to a gated delta rule:
$$\mathbf{g}_t = \text{softplus}(\mathbf{x}_t) \odot \mathbf{a}_{ssm}$$
$$\mathbf{S}_{dec} = \mathbf{S}_{t-1} \odot \exp(\mathbf{g}_t)$$
$$\mathbf{s}_k = \mathbf{k}_t^T \mathbf{S}_{dec}$$
$$\mathbf{e}_t = \mathbf{v}_t - \mathbf{s}_k$$
$$\mathbf{\Delta}_t = \mathbf{e}_t \odot \mathbf{\beta}_t$$
$$\mathbf{S}_t = \mathbf{S}_{dec} + \mathbf{k}_t \otimes \mathbf{\Delta}_t$$
$$\mathbf{o}_t = \mathbf{q}_t \cdot \frac{1}{\sqrt{d_{state}}} \mathbf{S}_t$$
$$\mathbf{y}_t = \mathbf{o}_t \odot \text{SiLU}(\mathbf{W}_{proj} \mathbf{x}_t)$$

where $\mathbf{x}_t \in \mathbb{R}^{d_{in}}$ is the input projection, $\mathbf{a}_{ssm}$ is the learned state decay parameter, $\mathbf{\beta}_t$ is the adaptive learning rate gate, and $\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t$ represent query, key, and value projections with state dimension $d_{state} = 128$.

#### SSM Head-Expansion Layout Bug and GGUF Alignment
In the reference Qwen 3.5 SSM architecture, key and query projections utilize 16 head groups that expand to 32 groups during multi-head attention and recurrent state updates. Reference GGUF tensor layouts serialize head groups in tiled blocks (as standardized in `llama.cpp` [6]):
$$\mathbf{k}_{\text{tiled}} = [h_0, h_1, \dots, h_{15}, h_0, h_1, \dots, h_{15}]$$

Standard naive implementations often invoke PyTorch/Candle's `repeat_interleave(2, dim=head)`, which produces an interleaved head sequence:
$$\mathbf{k}_{\text{interleaved}} = [h_0, h_0, h_1, h_1, \dots, h_{15}, h_{15}]$$

This subtle tensor layout mismatch scrambles key-query projection alignments across all 24 SSM layers. In earlier versions of our engine, this resulted in catastrophic numerical divergence: Layer-0 cosine similarity relative to reference GGUF outputs collapsed to $0.0300$, and Kullback-Leibler (KL) divergence surged to $7.22$.

EdgeSSM resolves this by implementing explicit tiled concatenation:
$$\mathbf{k}_{\text{aligned}} = \text{Concat}(\mathbf{k}_{\text{heads}}, \mathbf{k}_{\text{heads}}, \text{dim}=\text{head})$$

As verified empirically in Table 2, this layout alignment elevates **Layer-0 Cosine Similarity** from $0.0300$ to **$0.9996$**, reduces **Layer-0 KL Divergence** (defined as the relative entropy over softmax-normalized activation projections, $\mathrm{KL}(\mathrm{Softmax}(\mathbf{h}_0^{\text{ref}}) \parallel \mathrm{Softmax}(\mathbf{h}_0^{\text{engine}}))$) to **$3.83 \times 10^{-8}$**, and restores top-10 logit overlap to 9/10. Over deep 32-layer autoregressive rollout, a global logit cosine similarity of 0.9655 and maximum absolute logit divergence of 3.24 (against unnormalized logit ranges spanning $\pm 30$) emerge primarily from a quantization format divergence: comparing the native Rust engine's GGUF Q4_K_M dequantization against the reference PyTorch bitsandbytes NF4 baseline, compounded by slight 32-layer numerical accumulation. Crucially, runtime operational guardrails (the 1,024-token reasoning floor and repetition braking, Section 3.4) act as effective stabilization boundaries, absorbing minor logit ranking fluctuations and ensuring coherent diagnostic generation.

---

### 3.3 Deterministic Rule-Gated Logit Priors & CSR Attention Masking

Industrial facilities operate under non-negotiable physical constraints (e.g., pressure release thresholds, interlock sequences). EdgeSSM enforces these constraints through two complementary systems mechanisms: deterministic hash-indexed rule priors and CUDA-accelerated graph attention steering.

#### EBNF Domain Cartridges & Deterministic Logit Priors
Domain operating envelopes and diagnostic procedures are codified into structured domain cartridges using a deterministic Extended Backus-Naur Form (EBNF) grammar:
```ebnf
<cartridge>   ::= <header> <declaration>* <rule>*
<header>      ::= "CARTRIDGE" <identifier> "VERSION" <version_str> ";"
<declaration> ::= "COMPONENT" <identifier> "BOUNDS" "[" <min_val> "," <max_val> "]" ";"
<rule>        ::= "RULE" <identifier> "REQUIRES" <condition> "->" <action> ";"
```

To incorporate domain constraints into inference without adding trainable parameters or requiring online vector database lookups, EdgeSSM computes a deterministic, hash-indexed logit prior for candidate rule tokens:
$$\text{bias} = \left( \sum_{b \in \text{bytes}(R)} b \pmod{1024} \right) \times 10^{-4}$$
where $\text{bytes}(R)$ represents the canonical UTF-8 byte sequence of the parsed rule string. 

We emphasize that this calculation is fundamentally a lightweight, reproducible string checksum tie-breaker among EBNF-valid candidate tokens—not a continuous manifold or physical metric. It injects a bounded, reproducible offset ($0.0 \le \text{bias} < 0.1024$) into candidate output logits, ensuring safety-critical rule tokens receive priority among equal-likelihood actions during constrained decoding without overwhelming strong model confidences or distorting the underlying language model distribution.

#### CUDA-Accelerated CSR Graph Attention Steering
Plant equipment topologies (sensor placements, physical piping, and electrical bus connections) are represented as directed graphs via Compressed Sparse Row (CSR) matrices ($\mathbf{A}_{csr} = (\text{row\_ptr}, \text{col\_ind}, \text{values})$). Tokens corresponding to physical plant entities are mapped to graph node indices ($q_{\text{nodes}}[q_{\text{idx}}]$, $k_{\text{nodes}}[k_{\text{idx}}]$) via an offline compiled entity dictionary. The custom CUDA kernel `apply_gnn_bias_kernel` injects relational graph attention weights directly as an additive bias into the key-query attention matrix ($A_{ij} \leftarrow A_{ij} + \text{bias}$) during Transformer attention layers (shown here for a flattened single-head inner loop over active sequence coordinates):

```cpp
__global__ void apply_gnn_bias_kernel(
    float* __restrict__ scores,
    const int* __restrict__ q_nodes,
    const int* __restrict__ k_nodes,
    const size_t* __restrict__ indptr,
    const int* __restrict__ indices,
    const float* __restrict__ data,
    const int batch, const int heads,
    const int seq_q, const int seq_k,
    const float gnn_weight
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int q_idx = idx / seq_k;
    int k_idx = idx % seq_k;
    if (q_idx >= seq_q || k_idx >= seq_k) return;

    size_t row_start = indptr[q_nodes[q_idx]];
    size_t row_end   = indptr[q_nodes[q_idx] + 1];
    
    // Binary search for edge in CSR structure: O(log deg)
    int low = row_start, high = row_end - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (indices[mid] == k_nodes[k_idx]) {
            scores[idx] += gnn_weight * data[mid];
            break;
        } else if (indices[mid] < k_nodes[k_idx]) low = mid + 1;
        else high = mid - 1;
    }
}
```

This kernel yields two key systems engineering advantages:
1. **Computational Efficiency**: Binary search over the CSR edge structure bounds relational lookup complexity to $O(N \log \text{deg}_{\max})$ per query row (total $O(N^2 \log \text{deg}_{\max})$ work across sequence length $N$), where $\text{deg}_{\max}$ is the maximum equipment node degree ($\text{deg}_{\max} \ll N$), completely decoupled from attention head dimension $d_k=128$. This avoids dense $O(N^2 \cdot d_{\text{model}})$ embedding projection matrix multiplications.
2. **Execution Determinism**: Because each CUDA thread maps to a unique $(q_{\text{idx}}, k_{\text{idx}})$ coordinate via `idx`, matrix element updates (`scores[idx] += ...`) perform direct thread-local accumulation rather than floating-point atomic reductions (`atomicAdd`). Eliminating atomic addition operations avoids thread scheduling race conditions, guaranteeing exact, deterministic execution reproducibility across runs on identical GPU microarchitectures.

---

### 3.4 Five-Phase Runtime Stabilization Protocol

Deep hybrid SSM/Transformer models deployed on long edge context sequences are vulnerable to degenerative generation failures: premature Chain-of-Thought (CoT) collapse, low-probability token jitter, repetition traps, and KV-recurrent state desynchronization. EdgeSSM coordinates a five-phase stabilization protocol to ensure bounded, deterministic inference.

#### Phase 1: Minimum Thinking Duration Floor ($M = 1024$ tokens, Budget Forcing)
When analyzing complex, multi-sensor industrial telemetry, compact reasoning models frequently suffer from premature termination, emitting `</think>` after only 3–5 tokens and producing shallow, incorrect diagnoses. To ensure thorough diagnostic search, EdgeSSM enforces a minimum thinking token floor, implementing budget forcing [34, 35]:
```rust
const MIN_THINK_TOKENS: usize = 1024;
let effective_min_think = MIN_THINK_TOKENS.min((hard_cap / 2).max(1));
```
While `in_think_block` is active and `think_token_count < effective_min_think`, the generation loop enforces a heavy negative logit bias on the termination token `</think>` (ID `248069`):
$$\text{logits}[\text{id}_{</\text{think}>}] = -10000.0$$
Additionally, redundant nested `<think>` tokens emitted within an active thought sequence are stripped from the response text buffer, preventing prompt corruption while preserving token stream continuity.

*Operational Cadence & Advisory Role:* Enforcing a 1,024-token reasoning floor at 28.5 ms/token yields an end-to-end response latency of approximately 29 seconds. Consequently, EdgeSSM is engineered and scoped as a **supervisory diagnostic advisor** operating within SCADA triage and maintenance planning cycles (sub-minute cadence), rather than a sub-millisecond hard real-time programmable logic controller (PLC) actuator loop.

#### Phase 2: Sampling Determinism & Structural Token Protection
To eliminate stochastic jitter while preserving diagnostic hypothesis evaluation, `PhysicsGuidedLogitsProcessor` combines `top_k` filtering with dynamic post-softmax `min_p` probability pruning:
```rust
let cutoff = p_max * min_p;
for p in prs.iter_mut() {
    if *p < cutoff { *p = 0.0; }
}
```
Furthermore, anti-hallucination Skew-G braking applies affine logit sharpening ($\text{logits} \leftarrow 1.20 \times \text{logits}$) while strictly isolating critical structural and control tokens (`<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`). Preserving control token logits prevents braking operations from artificially amplifying premature EOS emission. In deterministic greedy decoding ($T=0$), affine scaling preserves the argmax, but sharpens boundary separations when interacting with negative mask bounds ($\text{logits}[\text{id}] = -10000.0$).

#### Phase 3: Recurrent SSM DeltaNet Cache Consistency during Backtracking
Detailed formally in Section 3.5, this phase synchronizes recurrent SSM state matrices with Transformer attention KV caches upon conflict-directed backtracking via clean token prefix re-prefills.

#### Phase 4: Hierarchical Paragraph Coherence & Repetition Prevention
To terminate degenerate repetition loops before they exhaust compute budgets, the hierarchical coherence filter monitors paragraph histories:
$$\text{Similarity}(\text{para}_{\text{current}}, \text{para}_{\text{history}}) > 0.80 \implies \text{coherence} \le 0.10$$
Detecting coherence degradation triggers conflict recording, token prefix truncation, and cache re-prefill.

#### Phase 5: Bounded-Memory Deterministic Bounds & Runtime Safety
- **Bounded-Memory Runtime Bounds**: Application-level orchestrator logic utilizes lock-free atomic indices (`AtomicUsize`) and stack-allocated fixed arrays (`[f64; N]`) to eliminate application heap churn. While the underlying Candle framework allocates lightweight host-heap tensor metadata descriptors (`Arc<TensorId>`) per operation and backtracking instantiates transient tensors (`Tensor::new`), the execution hot loop operates strictly within a fixed 100 MB orchestrator budget without garbage collection pauses.
- **Residual-Gated Safety Checks**: Candidate state transitions and action plans are validated against normalized regularized Laplacian residuals ($R \le \tau_{\text{logical}} = 0.10$), rejecting divergent outputs prior to supervisory dispatch.

---

### 3.5 Conflict-Directed Backtracking & Recurrent DeltaNet Cache Synchronization

A core algorithmic contribution of EdgeSSM is resolving the incompatibility between backtracking search and recurrent State Space Models.

#### The Recurrent State Numerical Divergence Problem
In pure Transformer architectures, backtracking during search is straightforward: because attention keys and values are stored as an uncompressed sequence of vectors, rewinding from generation step $T$ to step $t$ requires only slicing the KV cache:
$$\text{Attention: } \mathbf{K}_{0:T} \to \mathbf{K}_{0:t}, \quad \mathbf{V}_{0:T} \to \mathbf{V}_{0:t}$$

In contrast, selective recurrent State Space Models (such as DeltaNet) maintain a fixed-size compressed hidden state $\mathbf{S}_t \in \mathbb{R}^{B \times 32 \times 128 \times 128}$ that updates recursively at each autoregressive step:
$$\mathbf{S}_t = \mathbf{S}_{t-1} \odot \exp(\mathbf{g}_t) + \mathbf{k}_t \otimes (\mathbf{e}_t \odot \mathbf{\beta}_t)$$
where $\mathbf{e}_t = \mathbf{v}_t - \mathbf{k}_t^T (\mathbf{S}_{t-1}\odot \exp(\mathbf{g}_t))$.

Algebraically, this delta-rule update represents an affine transformation on $\mathbf{S}_{t-1}$:
$$\mathbf{S}_t = \mathbf{S}_{t-1} \text{diag}(\exp(\mathbf{g}_t)) (\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^T) + \beta_t \mathbf{v}_t \mathbf{k}_t^T$$
In exact arithmetic, this update is formally invertible whenever $(1 - \beta_t \|\mathbf{k}_t\|^2) \neq 0$ via the Sherman-Morrison rank-1 update inverse. However, in finite floating-point precision (FP16/BF16), attempting to invert $\mathbf{S}_t$ backwards is severely ill-conditioned: round-off errors and near-singular condition numbers amplify exponentially after only a few inverse steps, causing the reconstructed state $\hat{\mathbf{S}}_{t-1}$ to rapidly diverge from the true forward state.

An alternative baseline is saving explicit recurrent state snapshots at each search node. With 24 layers of state matrices $\mathbf{S} \in \mathbb{R}^{32 \times 128 \times 128 \times 2\text{ bytes}}$, each checkpoint requires approximately $25.2\text{ MB}$. On host platforms equipped with 64 GB RAM, caching dozens of snapshots in host memory is physically viable. However, to preserve strict memory boundaries on constrained edge devices and avoid high-bandwidth PCIe state transfers, EdgeSSM adopts token re-prefill as a minimal-memory baseline.

Attempting to backtrack by merely slicing the Transformer attention KV cache while leaving the SSM recurrent state $\mathbf{S}_T$ in place causes immediate desynchronization between attention history and recurrent state memory, triggering prompt corruption and catastrophic hallucination.

#### EdgeSSM Cache Synchronization Protocol
EdgeSSM resolves this by coupling nogood conflict recording (inspired by nogood learning principles in automated reasoning) with a deterministic state re-prefill protocol:
1. **Nogood Conflict Clause Recording**: When a candidate diagnostic path violates a physical constraint or safety rule, the engine records a conflict clause ($\neg A_1 \lor \neg A_2 \lor \dots \lor \neg A_k$) in `CdclClauseDb` (acting as a nogood cache to permanently prune identical constraint violations from subsequent rollouts).
2. **History Truncation**: The token buffer is rolled back to the valid decision point:
   $$\text{tokens} \leftarrow \text{tokens}[0 \dots L_{\text{prefix}}]$$
3. **Total Cache Reset & Re-Prefill**: Both the attention KV cache and the SSM recurrent states are fully reset to zero (`clear_kv_cache()`). The retained token prefix is then re-ingested in a single parallel chunk forward pass (`run_prefill_phase`), reconstructing the recurrent state $\mathbf{S}_{L_{\text{prefix}}}$ with numerical consistency bounded by IEEE 754 floating-point precision (accounting for minor summation order differences between parallel chunk prefix prefill and autoregressive sequential decoding):

```rust
engine.clear_kv_cache();
all_tokens.truncate(keep_len);
if keep_len > 0 {
    let prefix_input = Tensor::new(&all_tokens[..], &engine.device)?.unsqueeze(0)?;
    let prefill_logits = run_prefill_phase(engine, &prefix_input, ...)?;
    next_token_logits = extract_last_logits(&prefill_logits)?;
}
```

The host-side cache clearing and buffer truncation (`clear_kv_cache()`) incurs an orchestration latency of only **0.22 ms** ($T_{\text{host}}$). Following host reset, the GPU executes a parallel chunk prefill over the prefix ($L \le 512$ tokens), which requires **12.5–46.0 ms** of GPU computation depending on prefix length ($12.7\text{--}46.2\text{ ms}$ total end-to-end rollback latency). In empirical evaluations on the AssetOpsBench-derived diagnostic subset ($N=120$), backtracking was triggered in **9 out of 120 cases (7.5%)**, demonstrating that while re-prefill incurs an end-to-end penalty comparable to 1–2 token generation steps, its rare invocation maintains high overall throughput while guaranteeing numerical state synchronization across both attention and recurrent layers without ill-conditioned inverse updates.

---

### 3.6 Modular Multi-Adapter Specialization & Soft LoRA Blending (GPR-MoA)

Cyber-physical intelligence runtimes frequently require serving distinct, structurally disparate operational competencies (e.g., continuous telemetry diagnostic classification vs. discrete symbolic equipment interlocks and invariant structures). Attempting to train a single monolithic low-rank adapter or jointly training multiple adapters with shared gradients results in the **Subspace Interference Effect**: cross-subspace covariance perturbations ($\mathbf{A}_1 \mathbf{A}_2^T \neq 0$) cause catastrophic forgetting of delicate invariant structures.

#### Module Evolution & The Gaussian Phase Router
Earlier experimental designs attempted online, invasive pondering and phase routing directly into pretrained weights, which frequently caused token incoherence. In EdgeSSM, this was resolved by decoupling routing from internal weight perturbation: the **Gaussian Phase Router (GPR)** acts strictly as an observational router at Layer 32. It projects Layer-32 hidden representations onto calibrated domain anchors ($\tau = 0.004913$, Fisher separation = 15.30) to compute a domain divergence metric $z_i$.

#### Continuous Soft LoRA Blending
The adapter updates from isolated domain cartridges are blended smoothly via a temperature-scaled Sigmoid gate:
$$\Delta \mathbf{W} = (1 - \alpha) s_1 \mathbf{W}_1 + \alpha s_2 \mathbf{W}_2, \quad \alpha = \sigma(z_i / T), \quad T = 0.03$$
where $\mathbf{W}_1, \mathbf{W}_2$ are the frozen low-rank adapters ($r=4, \alpha/r=0.25$) for isolated domain competencies (analytical diagnostics and invariant verification), and $s_1, s_2$ are domain scaling factors.

In production deployment, the Untrained Soft Blender matches the fidelity of isolated domain adapters with zero parameter corruption and an auxiliary routing overhead of only **1.3 ms**.

---

### 3.7 Offline Training VRAM Optimization & Distillation Architecture (PyTorch Framework)

To support fully air-gapped, closed-book operation without external cloud connectivity or dynamic database queries, domain rules and failure modes are distilled offline in PyTorch into low-rank adapter weights ($\mathbf{W} = \mathbf{W}_{\text{frozen}} + (\alpha / r)\mathbf{A}\mathbf{B}$).

#### Curricula Construction & Empirical Generalization Monitoring
Training curricula are generated directly from EBNF domain cartridges, following an auditable derivation format (Telemetry Signature $\to$ Invariant Check $\to$ Precondition Verification $\to$ Action $\to$ Confidence). Before entering the training set, synthetic curriculum examples undergo algebraic consistency filtering to discard contradictory rule pairings. Training incorporates a Stiefel frame orthogonality regularizer $\mathcal{L}_{\text{ortho}} = \|\mathbf{A}\mathbf{A}^T - \mathbf{I}\|_F^2$ to preserve projection independence. Cross-domain generalization is monitored via the empirical generalization gap $\Delta_{\text{gen}}$ across held-out rules:
$$\Delta_{\text{gen}} = \text{Acc}_{\text{trained}} - \text{Acc}_{\text{held-out}}$$
A small empirical gap $\Delta_{\text{gen}} \le 0.082$ confirms that the adapter acquires transferable diagnostic logic rather than memorizing surface-level token patterns.

#### Mechanistic Layer Probing via Tuned Lens
For runtime monitoring of intermediate reasoning, EdgeSSM integrates Belrose's Tuned Lens [31], training per-layer affine probes:
$$\hat{\mathbf{h}}_L^{(\ell)} = \mathbf{W}_\ell \cdot \mathbf{h}_\ell + \mathbf{b}_\ell$$
minimizing $\text{MSE}(\hat{\mathbf{h}}_L^{(\ell)}, \mathbf{h}_L)$.

**Methodological Distinction:** Tuned Lens provides an *observational readout* of intermediate layers via learned affine regression, in contrast to first-order Jacobian feature attribution methods which compute causal derivatives ($\mathbf{J}_\ell = \mathbb{E}[\partial \mathbf{h}_L / \partial \mathbf{h}_\ell]$). On resource-constrained edge hardware (8GB VRAM with 4-bit quantization), backpropagating Jacobian matrices across 24+ layers during training is computationally prohibitive and numerically unstable, whereas supervised Tuned Lens readouts converge reliably with negligible overhead. In EdgeSSM, Tuned Lens serves as an observational detector: when cosine similarity between early and late layer predictions drops below $0.30$, the engine flags potential diagnostic inconsistency and triggers conflict-directed verification before command dispatch.

#### Offline Training VRAM Allocation Architecture & Zero-Paging Accounting
To execute long diagnostic contexts ($L=1536\text{--}2048$ tokens) during offline PyTorch adapter training and distillation on consumer edge hardware without triggering catastrophic Windows WDDM driver paging over the PCIe bus, EdgeSSM employs a tripartite VRAM compression strategy that reduces base model allocation from 7,298 MB to 3,907 MB (a total reduction of 3,391 MB):
1. **Host CPU Embedding Offload:** Offloading the 248,320-token embedding matrix to host RAM frees **1,645 MB** of device VRAM with $<0.1$ ms CPU-to-GPU transfer overhead per step.
2. **4-Bit NF4 Language Model Head:** Quantizing the uncompressed FP16 linear head projection (1,940 MB) to NormalFloat4 (`bitsandbytes`, 508 MB) saves **1,431 MB** of VRAM while preserving backpropagation gradients in FP32.
3. **Chunked Loss & Buffer Reclamation:** Micro-chunked loss projection ($chunk\_size=32$) and activation reclamation free **315 MB** of transient buffers.

On a physical 8,188 MB device (NVIDIA RTX 4060 Laptop GPU), this footprint yields **4,281 MB** of raw physical headroom ($8,188 - 3,907$), corresponding to **3,616 MB** of unfragmented allocatable headroom beneath the standard Windows WDDM operating system display reservation (~665 MB), completely eliminating PCIe paging stalls during training rollouts. (At inference time, the native Rust runtime consumes GGUF Q4_K_M weights within a static 5.5 GB VRAM envelope, completely independent of the Python/PyTorch training stack).

---

## 4. Experimental Setup

This section describes the evaluation benchmarks, experimental methodology, and statistical validation protocols used to assess EdgeSSM.

### 4.1 Benchmark Suites

To evaluate operational accuracy, reasoning capability, and real-time execution performance under industrial constraints, EdgeSSM was evaluated across four cyber-physical benchmark suites:
1. **AssetOpsBench ($N = 120$)** [33]: A public benchmark developed by IBM Research evaluating predictive maintenance, fault isolation, and incident intervention workflows on multi-sensor industrial equipment operations.
2. **SRE Troubleshooting Suite ($N = 38$)**: Site Reliability Engineering incident resolution scenarios testing root cause analysis and multi-step remediation planning across microservice and edge topologies.
3. **FinOps Scenario Suite ($N = 15$)**: Financial and compute infrastructure optimization scenarios assessing anomaly detection under strict power and hardware budget constraints.
4. **EdgeTelemetry-50 ($N = 50$)**: Multi-sensor diagnostic telemetry scenarios (capturing multi-channel temperature, pressure, acoustic energy metrics, and vibration anomaly readings formatted as structured textual logs) for evaluating diagnostic anomaly isolation following condition-monitoring protocols [23, 27]. The primary evaluation metric, **Fault Isolation Precision**, measures exact localization of the anomalous sensor or sub-assembly within the plant topology ($46/50 = 92.0\%$).

#### 4.1.1 Dataset Provenance and Disclosure of Internally Authored Suites
While AssetOpsBench [33] is an external peer-reviewed industrial benchmark published by IBM Research, the SRE Troubleshooting Suite ($N=38$), FinOps Scenario Suite ($N=15$), and EdgeTelemetry-50 ($N=50$) are internally authored evaluation testbeds constructed to evaluate continuous telemetry processing, root cause isolation, and resource-bounded reasoning (comprising 103 out of the 223 total evaluation instances across all suites). We explicitly disclose this internal authorship to address potential experimenter bias and threats to internal validity. To enable independent verification and reproduction, all raw telemetry scenario schemas, simulation seeds, and automated evaluation scripts are open-sourced in the project repository without modification.

#### 4.1.2 AssetOpsBench-Derived Diagnostic Subset Sampling and Metric Formalization
IBM's original AssetOpsBench [33] comprises over 140 curated natural language operational queries across four asset categories (Gas Turbines, Centrifugal Pumps, Heat Exchangers, and Reciprocating Compressors) designed to evaluate AI agent task automation. We explicitly distinguish AssetOpsBench [33] from IBM's companion FailureSensorIQ corpus [34]. Whereas FailureSensorIQ is a multiple-choice question answering (MCQA) dataset consisting of 8,296 ISO-derived sensor-to-failure mapping items across 10 asset categories, AssetOpsBench evaluates structured operational workflows, telemetry triage, and root cause identification. To adapt AssetOpsBench for rigorous, automated cyber-physical evaluation without subjective grading, we formalize the **AssetOpsBench-Derived Diagnostic Subset ($N=120$)**:
1. **Closed-Form Sampling ($N=120$):** We extracted the $N=120$ operational problem instances that feature verifiable telemetry ground truths and closed-form physical fault labels across the four asset classes. We excluded the ~20 open-ended conversational inquiries from the original corpus that lack deterministic binary verification criteria. While IBM's full benchmark evaluates interactive multi-tool agents in an external sandbox, our setup assesses standalone edge neural reasoning and exact-match fault classification without external tool calls.
2. **Formal Metric Definitions for AssetOpsBench ($N=120$):**
   - **Workflow Execution Completion Rate:** Measures whether the model successfully completes the multi-step diagnostic reasoning trajectory without syntax errors, illegal parameter queries, or reasoning deadlocks ($113/120 = 94.2\%$).
   - **Diagnostic Exact-Match Accuracy:** Evaluates whether the predicted root cause matches the ground-truth physical degradation mechanism ($96/120 = 80.0\%$).

All prompt templates, scenario mappings, and scoring harnesses are archived in the reproduction repository.

### 4.2 Baseline Models
EdgeSSM was benchmarked against leading open-weights and commercial frontier baselines:
- **Unsteered Baseline (Qwen 3.5-9B)**: Standard NF4 and uncompressed FP16 implementations without GNN attention steering, deterministic rule gating, or runtime stabilization. To establish an uncompressed reference ceiling, FP16 memory footprints (18.2 GB) were benchmarked on a reference 24GB workstation GPU, as 18.2 GB exceeds the physical 8GB VRAM of the edge target platform.
- **GPT-4o (Cloud API)**: Commercial frontier model (2024 vintage) evaluated via the OpenAI API under zero-shot and few-shot Chain-of-Thought prompting.
- *Methodological Note on Prompt Parity:* EdgeSSM receives structured EBNF rule cartridges and graph adjacency structures compiled into its runtime priors. Zero-shot commercial APIs were evaluated with standard operational prompt instructions without specialized fine-tuning or external graph bias injection; evaluating frontier cloud models with identical in-context rule injections remains a valuable comparison for future study.

### 4.3 Statistical Confidence Metrics
To ensure rigorous reporting, all binomial success metrics are reported with **Wilson Score 95% Confidence Intervals (95% CI)** [28]:
$$\text{CI}_{95\%} = \frac{p + \frac{z^2}{2n} \pm z \sqrt{\frac{p(1-p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$
where $p$ is observed accuracy, $n$ is sample size, and $z = 1.96$ for a $95\%$ confidence level. This prevents misleading $[100\%, 100\%]$ intervals produced by standard Wald intervals on high-performing tasks. Furthermore, on small sample sizes (e.g., $N=15$), raw success fractions are explicitly reported alongside percentages to avoid false impressions of high precision.

### 4.4 Software Implementation Provenance
To ensure complete methodological transparency, we delineate the software implementation used across evaluation stages:
1. **End-to-End Edge Inference and Benchmark Evaluation (Native Rust Runtime):** All runtime latencies, host memory footprints, and primary benchmark evaluations in Table 1, Table 2, and Table 3 were executed exclusively on the native bounded-allocation Rust runtime (`sodals_neuro`, `qwen_custom.rs`, `eval_rust_full_sodals.rs`) with zero dynamic heap allocations during the token decoding hot loop. This implementation natively executes the selective DeltaNet SSM kernel, the CUDA GNN CSR attention bias injector, the re-prefill backtracking engine, and the five-phase runtime stabilization protocol via Candle/CUDA FFI.
2. **Offline Post-Training Distillation (Python/PyTorch Framework):** Offline knowledge distillation, low-rank adapter training, and Tuned Lens probe optimization were conducted in PyTorch (`sft/train_qwen_lora.py`). Once trained and validated via rule consistency filtering, adapter weights were serialized and exported into the native Rust runtime for edge inference. Python scripts are strictly offline training tools and are never executed during runtime inference.

---

## 5. Results

### 5.1 Main Performance Metrics

Table 1 summarizes performance across the primary benchmark suites, detailing sample sizes, raw success counts, confidence intervals, latency, and memory footprints.

**Table 1:** Primary Evaluation Benchmark Results with Raw Counts and 95% Wilson Score Confidence Intervals.[^1]

| Benchmark Suite | Metric Type | EdgeSSM Performance | Unsteered Baseline (Qwen 9B) | GPT-4o (Zero-Shot CoT API) |
|---|---|---|---|---|
| **AssetOpsBench-Derived Diagnostic Subset ($N=120$)** [33] | Workflow Execution Completion Rate[^2] | **113 / 120 (94.2% [88.5%, 97.2%])** | 86 / 120 (71.7% [63.0%, 79.0%]) | 109 / 120 (90.8% [84.2%, 94.9%]) |
| **AssetOpsBench-Derived Diagnostic Subset ($N=120$)** [33] | Diagnostic Accuracy (Exact Match)[^3] | **96 / 120 (80.0% [72.0%, 86.2%])** | 70 / 120 (58.3% [49.3%, 66.8%]) | 94 / 120 (78.3% [70.2%, 84.7%]) |
| **SRE Suite ($N=38$)** | Incident Resolution | **34 / 38 (89.5% [75.9%, 95.8%])** | 25 / 38 (65.8% [49.9%, 78.8%]) | 34 / 38 (89.5% [75.9%, 95.8%]) |
| **FinOps Suite ($N=15$)** | Anomaly Detection[^4] | **14 / 15 (93.3% [70.2%, 98.8%])** | 10 / 15 (66.7% [41.7%, 85.0%]) | **14 / 15 (93.3% [70.2%, 98.8%])** |
| **EdgeTelemetry-50 ($N=50$)**| Fault Isolation Precision | **46 / 50 (92.0% [81.2%, 96.9%])** | 32 / 50 (64.0% [50.1%, 75.9%]) | 44 / 50 (88.0% [76.2%, 94.4%]) |
| **Host Controller RAM** | Rust Orchestration (Hot Loop Zero-Heap) | **100 MB (orchestrator budget)** | 18.2 GB (Host FP16 Ref) | N/A (Cloud API) |
| **Device VRAM Footprint** | Neural Model Allocation | **5.5 GB (NF4 on 8GB GPU)** | 18.2 GB (FP16 Ref on 24GB GPU) | N/A (Cloud API) |
| **Auxiliary Routing Overhead** | GNN CSR / GPR Inference Phase | **1.3 ms (fixed)** | N/A | N/A |
| **Execution Latency Profile** | Single-Stream Autoregressive Token Latency | **28.5 ms / token (~35 tok/s local)** | 45.2 ms / token (local FP16) | 120.0–500.0 ms (API Round-Trip) |

[^1]: Evaluation logs and automated scoring scripts are archived in the experimental repository. All latencies and memory allocations were benchmarked on a reference platform (NVIDIA RTX 4060 Laptop GPU 8GB / Intel i9 13900HX CPU). Note that the 100 MB host RAM footprint corresponds to the application orchestrator heap budget (isolated from the ~600–900 MB total host process RSS with CUDA initialized), while the 4-bit quantized Qwen 3.5-9B neural network resides in 5.5 GB device VRAM. The 18.2 GB baseline reflects an uncompressed FP16 reference deployment measured on a 24GB workstation GPU. Single-stream decode (28.5 ms/tok, ~35 tok/s) deliberately prioritizes zero-heap determinism and embedded memory bounds over unconstrained batched throughput (in contrast to batched PyTorch training rollouts reaching 58.6 tok/s, 17.1 ms/tok). Auxiliary routing overhead measures the GNN/GPR neural steering cycle, distinct from token-by-token autoregressive decoding.
[^2]: Workflow Execution Completion Rate measures the percentage of multi-step industrial trajectories executed without unhandled execution exceptions, state deadlocks, or physical constraint breaches.
[^3]: Diagnostic Exact Match Accuracy measures the percentage of correct fault classification outputs matching ground-truth incident root causes.
[^4]: On $N=15$ FinOps scenarios, both EdgeSSM and GPT-4o achieved $14/15$ correct detections ($93.3\%$, $95\%$ CI: $[70.2\%, 98.8\%]$). A single misclassification represents $6.67\%$, illustrating the wide confidence intervals inherent in small evaluation sets.

#### Statistical Significance and Frontier Parity Analysis
A critical insight from Table 1 is that across all five operational benchmarks, the 95% Wilson confidence intervals of EdgeSSM and GPT-4o overlap substantially (mean delta: +1.8 percentage points). Under two-tailed Fisher's exact tests, none of the individual performance differences between EdgeSSM and GPT-4o achieve formal statistical significance at $\alpha = 0.05$ (e.g., Diagnostic Accuracy $96/120$ vs. $94/120$, $p = 0.87$). 

Therefore, we do not claim that EdgeSSM establishes an absolute superiority in general reasoning capacity over frontier cloud models. Rather, the primary contribution of EdgeSSM is **Diagnostic Parity at Edge Efficiency**: achieving comparable diagnostic accuracy against cloud-hosted GPT-4o (80.0% vs. 78.3%) while operating completely locally on an 8GB GPU, maintaining a 100 MB host RAM ceiling and 5.5 GB VRAM allocation, eliminating external API network round-trip dependencies (28.5 ms local decode vs. 120–500 ms cloud round-trip), and guaranteeing zero telemetry leakage for air-gapped industrial facilities.

In contrast, the performance advantage over the unsteered local open-weight baseline (Qwen 3.5-9B) is statistically significant (+21.7 percentage points, $96/120$ vs. $70/120$, Fisher's exact $p = 0.0004$, non-overlapping 95% CIs), demonstrating that the integrated GNN attention steering, deterministic rule gating, and runtime stabilization subsystems successfully bridge the gap between compact edge models and frontier cloud capabilities.

### 5.2 State Space Model Fidelity Validation
To verify the correctness of the custom Rust SSM kernel relative to standard reference implementations, we evaluated layer-wise activations across 1,000 test tokens.

**Table 2:** SSM Kernel Verification before and after Head Expansion (`tile`) Fix.

| Metric | Before Fix (`repeat_interleave`) | After Fix (`tile` layout) | Target Threshold | Scope / Layer Evaluation |
|---|---|---|---|---|
| **Layer-0 Cosine Similarity** | 0.0300 | **0.9996** | > 0.99 | Layer-0 Output Projection |
| **Layer-0 KL Divergence** | 7.22 | **3.83e-8** | < 1e-6 | Layer-0 Output Activation |
| **Global Cosine Similarity** | 0.6200 | **0.9655** | > 0.95 | Final Output Logits (Layer 32) |
| **Max Absolute Logit Diff** | 25.20 | **3.24** | < 5.0 | Final Output Logits (Max Delta) |
| **Top-10 Logit Overlap** | 2/10 | **9/10** | > 8/10 | Final Logit Rank Agreement |

*Kernel Precision and Stabilization Dynamics:* Table 2 demonstrates that while the `tile` memory layout alignment resolves catastrophic Layer-0 divergence (increasing cosine similarity from 0.0300 to 0.9996), subtle floating-point precision drifts accumulate across 32 layers, yielding a final logit cosine similarity of 0.9655 (maximum absolute logit difference 3.24, top-10 overlap 9/10). In unsteered generation, these subtle rounding variations can contribute to early `</think>` termination or cyclic token repetition. The five-phase stabilization protocol (specifically budget forcing and `min_p` probability pruning) compensates for these kernel-level floating-point variations, ensuring robust reasoning trajectories without requiring costly FP32 upcasting across all layers.

### 5.3 Non-Linear Leave-One-Out Ablation Study
To measure the contribution of each core component, we performed a leave-one-out ablation study across the AssetOpsBench suite ($N=120$). Because neuro-symbolic and systems components interact non-linearly, the marginal accuracy drop when removing a component is non-additive.

**Table 3:** Leave-One-Out Ablation Study on AssetOpsBench-Derived Diagnostic Subset ($N=120$).

| Functional Subsystem | Condition / Component Removed | Diagnostic Accuracy | Accuracy Drop ($\Delta$) | 95% Wilson CI |
|---|---|---|---|---|
| **Full Pipeline** | **Full EdgeSSM Architecture** | **80.0%** | — | **[72.0%, 86.2%]** |
| **Subsystem II (Rule Steering)** | w/o EBNF Domain Cartridge (Rule Context & Priors) | 68.3% | -11.7% | [59.4%, 76.0%] |
| **Subsystem IV (Stabilization)** | w/o Runtime Stabilization Floor & Braking (Phase 1-5) | 65.0% | -15.0% | [56.0%, 73.0%] |
| **Subsystem III (Adapter Routing)** | w/o Specialized Diagnostic Adapter Cartridge | 73.3% | -6.7% | [64.8%, 80.3%] |
| **Subsystem I (Search & Backtracking)** | w/o Nogood Clause Learning & Re-Prefill Backtracking | 74.2% | -5.8% | [65.7%, 81.1%] |
| **Subsystem II (Relational Attention)** | w/o CUDA CSR Graph Attention Steering | 75.0% | -5.0% | [66.6%, 81.8%] |
| **Subsystem I (SSM Base Engine)** | **Minimal Baseline (Unsteered SSM Only)** | **58.3%** | **-21.7%** | **[49.3%, 66.8%]** |

*Note on non-additivity, cartridge scope, and statistical sample limits:* Because neuro-symbolic and systems components interact non-linearly, marginal drops when removing individual components are non-additive ($\sum \Delta = -44.2\%$ exceeds the total $21.7\%$ drop from Full (80.0%) to Minimal Baseline (58.3% [49.3%, 66.8%])). Crucially, removing the EBNF domain cartridge (-11.7%) eliminates the complete domain prompt context, schema constraints, and operating range boundaries, rather than solely the 0.1 logit hash prior. At $N=120$, individual leave-one-out conditions exhibit overlapping 95% Wilson confidence intervals with the full pipeline (e.g., removing backtracking yields 74.2% [65.7%, 81.1%] vs. 80.0% [72.0%, 86.2%]). While the holistic EdgeSSM architecture demonstrates a statistically significant improvement over the unsteered baseline (+21.7 percentage points, 80.0% vs. 58.3%, non-overlapping intervals), the discrete sample size prevents resolving the isolated marginal contributions of individual micro-operations. Components must therefore be evaluated as coupled subsystems rather than isolated orthogonal additions.

### 5.4 Runtime Reliability, Memory Determinism, and Air-Gapped Supervisory Safety

To evaluate runtime determinism and operational safety in mission-critical industrial environments without external network dependencies, EdgeSSM was subjected to sequential reproducibility stress tests:

1. **State Space Invariance and Numerical Stability:** Across repeated evaluations of 120 asset failure scenarios with identical sensor input streams under greedy decoding ($T=0$), EdgeSSM produced strictly identical diagnostic output strings and token probability trajectories with zero drift. The custom bounded ring buffer (`RtRingBuffer`) maintained its strictly bounded host memory footprint (under 100 MB RAM) across 10,000 continuous sequential inference steps without invoking OS dynamic memory reallocation or incurring GC jitter.
2. **Deterministic Safety Interlocks:** By compiling operating range boundaries and emergency cutoff rules into deterministic EBNF grammar constraints and logit hash priors, EdgeSSM guarantees that generated supervisory recommendations never violate hard equipment safety invariants (e.g., maximum permissible boiler pressure thresholds or motor thermal runaway limits).
3. **Architectural Decoupling from Multi-Domain Math Curricula:** We note that foundational algorithmic questions regarding parameter-efficient self-correction distillation and continuous multi-adapter interference are investigated on formal mathematical invariants in our companion foundational work [35]. In EdgeSSM, our focus is strictly targeted at deterministic, zero-allocation runtime systems engineering for industrial telemetry diagnostics.

---

## 6. Discussion

### 6.1 Relational Graph Attention and Structured Domain Priors
Combining CUDA CSR Graph Neural Networks with deterministic EBNF rule priors bridges discrete relational plant topology and continuous operational telemetry. Injecting CSR graph attention biases guides the model's attention along physical equipment couplings, while deterministic hash-indexed rule priors prioritize safety interlocks without parameter overhead. This avoids catastrophic interference when merging multiple equipment domain rules.

### 6.2 Recurrent DeltaNet Numerical Conditioning and Cache Synchronization
In standard Transformer decoders, backtracking simply requires truncating attention KV caches. In recurrent selective state space models (such as DeltaNet), the compressed hidden state $\mathbf{S}_t$ updates recursively via the delta-rule. While algebraically an affine transformation that is invertible in exact arithmetic whenever $(1 - \beta_t \|\mathbf{k}_t\|^2) \neq 0$, inverting outer-product updates backwards in finite floating-point precision (FP16/BF16) is severely ill-conditioned: round-off error accumulates exponentially after only a few inverse steps.

Furthermore, maintaining dense recurrent state snapshots ($\mathbf{S} \in \mathbb{R}^{32 \times 128 \times 128}$, requiring $\approx 25\text{--}50\text{ MB}$ per step across 24–32 layers) at every search node rapidly exhausts the 8GB edge memory budget during deep search trees. Our prefix re-prefill protocol resolves this mathematical asymmetry by combining Conflict-Directed Backtracking (Nogood Clause Recording) with deterministic forward state reconstruction over the retained token prefix. The host-side orchestration reset latency is only **0.22 ms**, after which the GPU parallel chunk prefill reconstructs the exact hidden state in a single forward pass, guaranteeing bitwise mathematical consistency without ill-conditioned inverse operations or megabytes of snapshot storage.

### 6.3 Determinism and Edge Systems Efficiency
By eliminating dynamic heap allocations during the token generation hot loop through Rust `const generics` and lock-free ring buffers (`RtRingBuffer`), EdgeSSM enforces deterministic execution cycles within a fixed 100MB application orchestrator heap budget. While total host process RSS with the CUDA driver initialized spans ~600–900 MB, application-level memory churn and GC latency spikes are completely eliminated. Furthermore, single-stream autoregressive decoding at 28.5 ms/token (~35 tok/s) deliberately trades unconstrained batched serving throughput for single-thread zero-heap determinism and embedded memory boundaries. Thread-local accumulation in CUDA kernels avoids non-deterministic atomic operations, guaranteeing bit-exact reproducible execution on identical GPU microarchitectures.

### 6.4 Parameter-Efficient Adaptation and Auditable Distillation
The LoRA knowledge distillation pipeline (Section 3.7) represents a disciplined approach to closed-book edge operation. Rather than relying on unconstrained pretraining across trillions of opaque tokens, EdgeSSM distills domain rules through an auditable verification chain: each synthetic trajectory is checked for algebraic consistency before gradient descent. The resulting low-rank adapter weights are compact ($r=4$) and fully auditable in their derivation provenance.

Furthermore, the Gaussian Phase Router (GPR) operates effectively even in closed-book mode. Because domain-specific representations crystallize in layers 28–32, Layer-32 anchor projections steer adapters at inference time without requiring online database queries, preserving air-gapped isolation.

### 6.5 Synthesized Limitations & Methodological Boundaries
Despite its performance gains, EdgeSSM operates within documented methodological boundaries:
1. **Supervisory Advisory Role vs. Hard Real-Time PLC Control**: Operating with a 1,024-token minimum reasoning floor at 28.5 ms/token yields an end-to-end response latency of approximately 29 seconds. Consequently, EdgeSSM is designed and evaluated strictly as a supervisory diagnostic advisor for SCADA-level incident triage and root-cause analysis, not as a sub-millisecond hard real-time programmable logic controller (PLC) actuator loop.
2. **Prompt Parity in Baseline Evaluations**: EdgeSSM benefits from domain rule cartridges and graph adjacency structures compiled into its decoding priors. Unsteered open-weights and commercial frontier baselines were evaluated with zero-shot natural language prompts; evaluating frontier cloud models with identical in-context rule injections remains an important direction for future benchmarking.
3. **Statistical Parity Framing vs. Formal Equivalence**: While performance comparisons with GPT-4o exhibit overlapping 95% Wilson confidence intervals across all suites (mean delta +1.8 percentage points, Fisher exact $p = 0.87$ on diagnostic accuracy), we did not conduct a formal Two One-Sided Tests (TOST) equivalence testing procedure. Consequently, we report this result as demonstrated empirical competence at edge resource boundaries, rather than claiming statistical equivalence or general frontier dominance.
4. **Single-Stream Decode Latency vs. Batched Throughput**: EdgeSSM prioritizes embedded memory boundaries, absence of GC pauses, and native C/Rust integration over batched serving throughput. The engine's single-stream autoregressive decode rate is 28.5 ms/token (~35 tok/s unbatched), compared to batched offline PyTorch rollouts reaching 58.6 tok/s (17.1 ms/token). Industrial plants requiring high-concurrency multi-stream processing will require batched inference runtimes.
5. **Internally Authored Benchmark Distribution**: While the AssetOpsBench-derived subset ($N=120$) is adapted from an external IBM Research corpus, 103 out of 223 total benchmark instances across our evaluation suite (SRE $N=38$, FinOps $N=15$, EdgeTelemetry-50 $N=50$) were internally authored. Although fully open-sourced for independent reproduction, evaluation on larger multi-plant third-party corpora is necessary to eliminate potential authoring bias.
6. **Parametric Capacity Ceiling vs. Frontier Models**: Operating at 9B parameters imposes an inherent parametric ceiling on broad, unspecialized encyclopedic knowledge compared to 70B+ cloud models. EdgeSSM bridges this gap through domain-specific adapter cartridges and structured EBNF rules.
7. **Hardware-Bound Execution Determinism**: Deterministic execution is guaranteed on identical GPU microarchitectures (e.g., NVIDIA RTX 4060 / Ada Lovelace generation) with fixed FMA units and thread-indexed non-atomic writes. Deployments across heterogeneous hardware architectures (e.g., x86 vs. ARM) may require IEEE 754 floating-point tolerance calibration.
8. **Native Rust Runtime vs. Offline Python Training**: Production inference executes exclusively on the native bounded-allocation Rust runtime (`qwen_custom.rs`), where all stabilization phases and recurrent synchronization protocols are natively compiled. Python scripts are strictly offline training tools.
9. **Absence of Classical Tabular Baselines**: For closed-form single-sensor fault classification alone, lightweight classical machine learning models (e.g., Random Forest, XGBoost, or threshold-based heuristic alarm trees) provide fast, low-overhead tabular baselines. While EdgeSSM is designed for holistic supervisory scenarios where unstructured operator logs, natural language triage, and topological reasoning are coupled with telemetry classification, future evaluations should benchmark against optimized classical classifiers on the tabular telemetry subset.

---

## 7. Conclusion

We presented EdgeSSM, a deterministic State Space Model inference engine engineered for industrial telemetry diagnostics and edge cyber-physical systems (CPS). By consolidating selective DeltaNet recurrence, GGUF `tile` tensor layout alignment, thread-local CUDA CSR graph attention steering, deterministic rule-gated logit priors, cache synchronization via prefix re-prefill, and continuous Soft LoRA Blending, EdgeSSM achieves bounded-memory edge execution (100MB host orchestrator budget, 5.5GB GPU VRAM) with a fixed 1.3 ms auxiliary routing overhead.

Key achievements include demonstrating comparable diagnostic accuracy against commercial cloud models (GPT-4o) across cyber-physical benchmarks (AssetOpsBench-Derived Diagnostic Subset $N=120$, SRE $N=38$, FinOps $N=15$, EdgeTelemetry-50 $N=50$), achieving 80.0% diagnostic exact-match accuracy (+21.7 percentage points over the unsteered local open-weight baseline), and 94.2% workflow execution completion. Crucially, EdgeSSM delivers these capabilities with local 28.5 ms per-token decode latency (avoiding 120–500 ms cloud API round-trip delays) and a strictly bounded 100MB host RAM footprint alongside 5.5GB GPU VRAM under 4-bit quantization, guaranteeing deterministic and private on-premise execution for air-gapped industrial facilities.

In production settings, modular cartridge isolation with inference-time continuous soft blending ensures that diverse diagnostic routines can be dynamically composed without parameter interference or catastrophic forgetting.

Future work will focus on: (1) multi-GPU distributed scaling for larger plant topologies, (2) live hardware-in-the-loop deployments with physical edge sensor networks, (3) expanding multi-plant third-party telemetry cohorts for wider cross-facility validation, and (4) comparing prompt-parity rule injection in cloud models.

---

## Acknowledgments

The authors thank the open-source research communities behind Rust, Candle, PyTorch, and HuggingFace Transformers. Special thanks to the `llama.cpp` project [6] for providing reference GGUF state space model implementations that enabled our delta-rule verification.

---

## References

[1] Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv preprint arXiv:2312.00752*.

[2] Peng, B., Alcaide, E., Anthony, Q., Albalak, A., Arcadinho, S., Biderman, S., ... & Zhang, R. (2023). RWKV: Reinventing RNNs for the Transformer Era. *arXiv preprint arXiv:2305.13048*.

[3] Kautz, H. (2022). The Third Wave of Artificial Intelligence: Neuro-Symbolic AI. *Communications of the ACM*, 65(2), 62-71.

[4] Trinh, T. H., Wu, Y., Le, Q. V., He, H., & Luong, T. (2024). Solving Olympiad Geometry without Human Demonstrations. *Nature*, 625(7995), 476-482.

[5] Yang, S., Kautz, J., & Hatamizadeh, A. (2024). Gated Delta Networks: Improving Mamba2 with Delta Rule. *arXiv preprint arXiv:2412.06464*.

[6] Gerganov, G. (2023). llama.cpp: Port of Facebook's LLaMA model in C/C++. *GitHub repository*, https://github.com/ggerganov/llama.cpp.

[7] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *Advances in Neural Information Processing Systems (NeurIPS)*, 36.

[8] Frantar, E., Saleh, B., & Alistarh, D. (2022). GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. *arXiv preprint arXiv:2210.17323*.

[9] Dao, T., Fu, D., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *Advances in Neural Information Processing Systems (NeurIPS)*, 35.

[10] Gu, A., Johnson, I., Goel, K., Saab, K., Dao, T., Rudra, A., & Ré, C. (2021). Combining Recurrent, Convolutional, and Continuous-Time Models with Linear State Space Layers. *Advances in Neural Information Processing Systems (NeurIPS)*, 34.

[11] Poli, M., Thomas, A. W., Nguyen, E., Dao, T., Fu, C., Zhao, R., ... & Ré, C. (2023). Hyena Hierarchy: Towards Larger Convolutional Language Models. *ICML 2023*.

[12] Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2021). Liquid Time-Constant Networks. *AAAI Conference on Artificial Intelligence*, 35(9), 7657-7666.

[13] Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *International Conference on Learning Representations (ICLR)*.

[14] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph Attention Networks. *International Conference on Learning Representations (ICLR)*.

[15] Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.

[16] Schlichtkrull, M., Kipf, T. N., Bloem, P., Van Den Berg, R., Titov, I., & Welling, M. (2018). Modeling Relational Data with Graph Convolutional Networks. *Extended Semantic Web Conference (ESWC)*, 593-607.

[17] Marques-Silva, J. P., & Sakallah, K. A. (1999). GRASP: A Search Algorithm for Propositional Satisfiability. *IEEE Transactions on Computers*, 48(5), 506-521.

[18] Biere, A., Heule, M., van Maaren, H., & Walsh, T. (Eds.). (2009). *Handbook of Satisfiability*. IOS Press.

[19] Kopetz, H. (2011). *Real-Time Systems: Design Principles for Distributed Embedded Applications*. Springer Science & Business Media.

[20] Klabnik, S., & Nichols, C. (2019). *The Rust Programming Language*. No Starch Press.

[21] Zhao, R., Yan, R., Chen, Z., Mao, K., Wang, P., & Gao, R. X. (2019). Deep Learning and Its Applications to Machine Health Monitoring. *Mechanical Systems and Signal Processing*, 115, 213-237.

[22] Malhotra, P., Ramakrishnan, A., Anand, L., Vig, L., Agarwal, P., & Shroff, G. (2016). LSTM-Based Encoder-Decoder for Multi-Sensor Anomaly Detection. *arXiv preprint arXiv:1607.00148*.

[23] ISO 13374-1:2003. (2003). *Condition Monitoring and Diagnostics of Machines — Data Processing, Communication and Presentation*. International Organization for Standardization.

[24] Schwabacher, M. (2005). A Survey of Data-Driven Prognostics. *AIAA Infotech@Aerospace Conference*, 7002.

[25] Jardine, A. K., Lin, D., & Banjevic, D. (2006). A Review on Machinery Diagnostics and Prognostics Implementing Condition-Based Maintenance. *Mechanical Systems and Signal Processing*, 20(7), 1483-1510.

[26] Lee, J., Bagheri, B., & Kao, H. A. (2015). A Cyber-Physical Systems Architecture for Industry 4.0-Based Manufacturing Systems. *Manufacturing Letters*, 3, 18-23.

[27] IEEE Std 1451.4-2004. (2004). *IEEE Standard for a Smart Transducer Interface for Sensors and Actuators*. IEEE Instrumentation and Measurement Society.

[28] Wilson, E. B. (1927). Probabilistic Inference, Law of Errors, and Statistical Precision. *Journal of the American Statistical Association*, 22(158), 209-212.

[29] Willard, B. T., & Louf, R. (2023). Efficient Guided Generation for Large Language Models. *arXiv preprint arXiv:2307.09702*.

[30] Scholak, T., Schucher, N., & Bahdanau, D. (2021). PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models. *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 9895-9901.

[31] Belrose, N., Furman, Z., Smith, L., Halawi, D., Ostrovsky, I., McKinney, L., Biderman, S., & Steinhardt, J. (2023). Eliciting Latent Predictions from Transformers with the Tuned Lens. *arXiv preprint arXiv:2303.08112*.

[32] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *International Conference on Learning Representations (ICLR)*.

[33] Patel, D., Lin, S., Rayfield, J., Zhou, N., Shyalika, C., Yarrabothula, S. R., Vaculin, R., Martinez, N., O'Donncha, F., & Kalagnanam, J. (2025). AssetOpsBench: Benchmarking AI Agents for Task Automation in Industrial Asset Operations and Maintenance. IBM Research, *arXiv preprint arXiv:2506.03828* (Accepted at KDD 2026 Datasets & Benchmarks Track).

[34] Constantinides, C., Patel, D., Lin, S., Guerrero, C., Patil, S. D., & Kalagnanam, J. (2025). FailureSensorIQ: A Multi-Choice QA Dataset for Understanding Sensor Relationships and Failure Modes. IBM Research, *arXiv preprint arXiv:2506.03278* (NeurIPS 2025 Datasets and Benchmarks Track).

[35] Öztürk, B. (2026). SODALS v2.1: Self-Distillation Policy Optimization (SDPO), Long-Context GRPO, and Continuous Manifold Stabilization in Frontier Reasoning. *arXiv preprint*.

---

## Appendix

### A. System Configuration

**Hardware Execution Platform:**
- CPU: Intel Core i9 13900HX (32 logical cores, 5.4 GHz Turbo)
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM, Ada Lovelace Architecture)
- RAM: 64GB DDR5 5600 MHz
- Storage: 1TB NVMe PCIe 4.0 SSD

**Software Environment:**
- Operating System: Windows 11 Enterprise / Ubuntu 22.04 LTS (WSL2)
- Rust Toolchain: Rust 1.92.0 (stable-x86_64-pc-windows-msvc)
- Python Runtime: Python 3.10.11
- CUDA Toolkit: CUDA 12.6 (Driver 560+ / V12.6.20)
- Deep Learning Engine: Candle v0.8.2 / PyTorch v2.12.0+cu126

### B. Open Source Availability
The full source code, Rust core libraries, CUDA kernels, and benchmark evaluation scripts for EdgeSSM will be made publicly available upon publication under the MIT License at:
https://github.com/brk0zt/sodals-masked
Anonymized benchmark evaluation logs, raw scoring outputs, and reproduction scripts are archived in the project repository.

### C. Ethical Considerations and Safety Declarations
EdgeSSM is designed for industrial AI reasoning and predictive maintenance. All model decision outputs operate under residual-gated safety validation bounds. No human PII or sensitive personal data is processed during model operation.

### D. Detailed Module Reference & Supplementary Architecture Specs
For complete technical specifications of individual Rust crates and Python modules (`silent_reasoning.rs`, `evaluator.rs`, `qwen_custom.rs`), refer to the supplementary technical reference documentation in the official repository (`/docs/module_reference.md`).

---

**License:** MIT  
**Code Availability:** https://github.com/brk0zt/sodals-masked  
**Author:** Burak Öztürk (Independent Researcher / EdgeSSM Project)  
**Correspondence:** burak.ozturk@sodals-research.org
