# SOURCES - Independent Condition Monitoring Standards Reference

This document compiles the independent engineering standards, international specifications, and textbook diagnostics used to establish the relations in `industrial_topology.txt` / `industrial_topology.qmem`. These mappings are derived entirely from external industry practices, independent of the benchmark's answer keys.

---

## 1. Machinery Protection and Vibration Monitoring (ISO & API Standards)

### ISO 17359:2018
*Condition monitoring and diagnostics of machines — General guidelines.*
* **URL:** [https://www.iso.org/standard/71120.html](https://www.iso.org/standard/71120.html)
* **Application:** Provides the generic procedures for setting up condition monitoring programs, identifying mechanical/electrical failure modes, and establishing parameter-to-fault matrices.
* **Mapped Relationships:**
  - `structural defect` $\rightarrow$ `vibration`, `noise`
  - `cavitation` $\rightarrow$ `vibration`, `noise`, `pressure or vacuum`
  - `lubrication fault` $\rightarrow$ `temperature`, `oil debris`

### ISO 20816-3:2022
*Mechanical vibration — Measurement and evaluation of machine vibration — Part 3: Industrial machines.*
* **URL:** [https://www.iso.org/standard/79011.html](https://www.iso.org/standard/79011.html)
* **Application:** Defines baseline and alarm thresholds for mechanical vibration across industrial rotating machinery.
* **Mapped Relationships:**
  - `unbalance` $\rightarrow$ `vibration`
  - `misalignment` $\rightarrow$ `vibration`
  - `eccentric rotor` $\rightarrow$ `vibration`

### ISO 13374-1 / ISO 13374-2
*Condition monitoring and diagnostics of machines — Data processing, communication and presentation.*
* **URL:** [https://www.iso.org/standard/38072.html](https://www.iso.org/standard/38072.html)
* **Application:** Standardized data processing block diagrams and diagnostic descriptors for mechanical vibration.

### API Standard 670
*Machinery Protection Systems (5th Edition).*
* **URL:** [https://www.api.org/products-and-services/standards](https://www.api.org/products-and-services/standards)
* **Application:** Standard instrumentation configurations for turbomachinery casing vibration, radial shaft vibration, axial position, and bearing temperatures.
* **Mapped Relationships:**
  - `bearing wear / damage` $\rightarrow$ `vibration` (radial proximity probes), `bearing temperature` (RTD sensors), `oil debris` / `oil temperature`.
  - `hogging or sagging rotor` $\rightarrow$ `axial position`, `shaft vibration`, `length measurement` (differential expansion).
  - `damaged rotor blade` $\rightarrow$ `vibration` (casing accelerometer), `exhaust temperature` (EGT spread).

---

## 2. Electrical Machinery Diagnostics (IEEE & FMEA Handbooks)

### IEEE Std 1415-2001
*Guide for Induction Machinery Maintenance Testing and Failure Analysis.*
* **URL:** [https://ieeexplore.ieee.org/document/989257](https://ieeexplore.ieee.org/document/989257)
* **Application:** Guides diagnostic parameters for electrical and mechanical stator/rotor faults.
* **Mapped Relationships:**
  - `stator windings fault` / `rotor windings fault` $\rightarrow$ `current` (Motor Current Signature Analysis - MCSA), `axial flux`, `winding temperature`.
  - `insulation deterioration` $\rightarrow$ `insulation resistance` (megohmmeter), `partial discharge` (for HV winding insulation), `winding temperature`.

### MIL-HDBK-338B / SAE JA1011
*Electronic Reliability Design Handbook / Evaluation Criteria for Reliability-Centered Maintenance (RCM) Processes.*
* **Application:** Root cause and failure symptom matrices for auxiliary electrical components.
* **Mapped Relationships:**
  - `brush(es) fault` $\rightarrow$ `current` harmonics, `temperature`.
  - `loss of input power phase` $\rightarrow$ `current` imbalance, `voltage` sag.

---

## 3. Power Transformer Diagnostics (IEEE & CIGRE Guides)

### IEEE C57.104-2019
*Guide for the Interpretation of Gases Generated in Oil-Immersed Transformers.*
* **URL:** [https://ieeexplore.ieee.org/document/8899887](https://ieeexplore.ieee.org/document/8899887)
* **Application:** Establishes percentile-based concentration limits and rate-of-change metrics for dissolved gas analysis (DGA).
* **Mapped Relationships:**
  - `arcing / electrical discharge` $\rightarrow$ `dissolved gas analysis` (acetylene, hydrogen), `partial discharge`.
  - `overheating` $\rightarrow$ `dissolved gas analysis` (ethylene, methane), `temperature` (winding/oil hot-spot).
  - `oil quality deterioration` $\rightarrow$ `moisture content` / `oil condition` testing.

### CIGRE Brochure 445 / IEEE Std 62
*Guide for Transformer Diagnostics.*
* **URL:** [https://e-cigre.org/publication/445-guide-for-transformer-diagnostics](https://e-cigre.org/publication/445-guide-for-transformer-diagnostics)
* **Mapped Relationships:**
  - `winding distortion` $\rightarrow$ `frequency response analysis (fra)`, `leak reactance flux`.
  - `connection / bushing faults` $\rightarrow$ `bushing capacitance`, `power factor (tan delta)`.

---

## 4. Gas Path Analysis & Turbomachinery Diagnostics (ASME & Textbook Reference)

### Sawyer's Gas Turbine Engineering Handbook
*Section: Gas Path Analysis (GPA) and Compressor Diagnostics.*
* **Application:** Aerothermal performance diagnostics based on process pressure, temperature, and flow telemetry.
* **Mapped Relationships:**
  - `compressor fouled` $\rightarrow$ `compressor efficiency`, `compressor pressure ratio`, `air flow`, `exhaust temperature` (spread).
  - `air inlet blockage` $\rightarrow$ `air flow`, `compressor pressure ratio`, `inlet depression / pressure`.
  - `power turbine dirty` $\rightarrow$ `turbine efficiency`, `exhaust temperature`, `output power`.
  - `seal leakage` $\rightarrow$ `fluid leakage` / `air leakage` / `oil leakage`, `pressure or vacuum`, `exhaust pressure` changes.

---

## 5. Academic Citation for FailureSensorIQ

### FailureSensorIQ: A Multi-Choice QA Dataset for Understanding Sensor Relationships and Failure Modes
* **Authors:** Christodoulos Constantinides, Dhaval Patel, Shuxin Lin, Claudio Guerrero, Sunil Dagajirao Patil, Jayant Kalagnanam (IBM Research)
* **arXiv ID:** [arXiv:2506.03278](https://arxiv.org/abs/2506.03278)
* **NeurIPS Track:** NeurIPS 2025 Datasets & Benchmarks Track
* **Dataset Registry (Hugging Face):** [ibm-research/FailureSensorIQ](https://huggingface.co/datasets/ibm-research/FailureSensorIQ)
* **Official Codebase:** [https://github.com/IBM/FailureSensorIQ](https://github.com/IBM/FailureSensorIQ)
