

| BURAK ÖZTÜRK Schmalkalden, Germany Phone: \+49 175 7460510 | Email: 1brkozt@gmail.com [LinkedIn](http://www.linkedin.com/in/brk0zt) | [GitHub](https://github.com/brk0zt) | [Hugging Face](http://huggingface.co/brk0zt) | Fotoğraf |
| :---- | :---: |

# **SUMMARY**

Computer Engineering undergraduate and Erasmus+ student at Schmalkalden University of Applied Sciences, seeking research-oriented internships and student assistant (HiWi) positions in AI, machine learning and automation (available up to 20 hours per week from Oct 2026). Hands-on experience in computer vision modeling (PyTorch, OpenCV, ONNX), dataset curation, and an independent research project on memory-efficient LLM post-training (4-bit NF4 QLoRA, GRPO reinforcement learning, mechanistic interpretability) and deterministic Rust edge inference. Committed to structured, reproducible engineering workflows.

# **EDUCATION**

**Schmalkalden University of Applied Sciences** | Schmalkalden, Germany | Oct 2026 \- Present

Erasmus+ Study Mobility, Computer Science (Informatik)

**Tarsus University** | Mersin, Türkiye | Sep 2024 \- Present (Expected graduation: Jun 2028\)

B.Sc. Computer Engineering (in progress)

GPA: 3.25 / 4.00 (German equivalent: 2.1)

Relevant coursework: Linear Algebra, Probability and Statistics, Data Structures and Algorithms, Object-Oriented Programming, Database Systems, Numerical Methods

**Sakarya Uygulamalı Bilimler Üniversitesi** | Sakarya, Turkey | Oct 2023 \- Jun 2024

Electrical/Electronics Engineering, one year of studies (no degree awarded)

**Gülveren Anadolu Lisesi** | Antalya, Türkiye | Sep 2018 \- Jun 2022

High school diploma | Grade average: 99.01 / 100 (German equivalent: 1.1)

# **WORK EXPERIENCE**

**BlueSense AI** | Istanbul, Türkiye | Jul 2026 \- Sep 2026

AI Department Intern (on-site)

- **Data privacy:** Automated EXIF metadata removal and eye/region-of-interest masking on facial datasets with OpenCV and Pillow, following GDPR guidelines.

- **Annotation and data quality:** Annotated 120+ images with bounding boxes in CVAT across dermatological defect categories (COCO JSON export); built auditing scripts to filter degenerate and clipped boxes and preprocessing pipelines (MediaPipe ROI cropping, 512x512 padding, Albumentations).

- **Model training and evaluation:** Trained CNN/FPN detection models in PyTorch on NVIDIA A100 GPUs via Slurm; tracked loss curves in TensorBoard and evaluated precision, recall and F1 with scikit-learn on 500 test images.

- **Inference optimization:** Cut inference latency from 25 ms to 12 ms via ONNX export; tuned the confidence threshold (0.42) and NMS IoU parameters and documented cross-device camera performance.

**JOBTEQ Solutions** | Bratislava, Slovakia | May 2026 \- Jul 2026

AI Engineer Intern (remote)

- **Data acquisition:** Engineered Python pipelines (requests, BeautifulSoup) to collect social media feeds and audience engagement data across multiple platforms.

- **Analytics:** Cleaned, normalized and analyzed interaction metrics with pandas and NumPy to track post performance and identify high-performing content patterns.

- **Automated visual generation:** Fed analytics insights into ComfyUI image generation pipelines, configuring prompts and generation parameters dynamically to produce targeted visual assets.

# **PROJECTS**

**EdgeSSM / SODALS — Neurosymbolic Edge Inference & RL Post-Training Framework** | Aug 2025 \- Present

Lead Developer & Independent Researcher | GitHub: [github.com/brk0zt/sodals-masked](https://github.com/brk0zt/sodals-masked)

Problem addressed: adapting and executing 9B-parameter hybrid State Space / Transformer models for mathematical reasoning and deterministic industrial telemetry diagnostics under strict consumer-grade edge memory ceilings (8 GB VRAM).

- **Memory-efficient RL post-training (GRPO):** Architected a Group Relative Policy Optimization (GRPO) pipeline for a 9B hybrid SSM-Transformer (Qwen 3.5 9B) in 4-bit NF4 QLoRA; implemented chunked policy cross-entropy (`chunk_size=32`) and CPU embedding offloading to cut peak training VRAM by 46.5% (7,298 MB to 3,907 MB) on a single 8 GB NVIDIA RTX 4060 laptop GPU without Windows WDDM PCIe paging.

- **Deterministic Rust edge runtime (EdgeSSM):** Built a zero-allocation Rust (Candle) runtime executing 4-bit GGUF (`Q4_K_M`) hybrid Mamba-2 SSM + GQA Transformer models within a 100 MB host RAM ceiling at 28.5 ms/token; resolved a low-level SSM head-topology projection bug (`repeat_interleave` vs. `tile`), reducing Layer-0 numerical KL divergence from 0.162 to 3.83e-8.

- **Subspace routing & neurosymbolic verification:** Formulated a Layer-32 Gaussian Process Soft-MoA router to mitigate non-orthogonal LoRA subspace interference ($A_1 A_2^\top \neq 0$) and combined Layer-28/30 Tuned Lens trajectory probing with a SymPy CST symbolic verifier, raising zero-leakage MATH-500 held-out accuracy from 82.4% (SFT) to 84.8% (GRPO + Best-of-$G$ PRM under symbolic equivalence; 79.2% to 81.6% under strict string matching).

- **Industrial IoT telemetry gatekeeper:** Implemented a deterministic Finite State Machine (FSM) and physical ontology verifier over multivariate sensor streams (vibration, acoustic, thermal), scoring 96/120 verified exact matches (vs. 94/120 for unconstrained GPT-4o) on an AssetOpsBench-derived closed-form diagnostic benchmark ($N=120$) with 0% safety-schema violations.

# **SKILLS**

**Machine learning, LLMs and computer vision:** Python, PyTorch, Hugging Face Transformers, PEFT (LoRA/QLoRA), bitsandbytes (4-bit NF4), Reinforcement Learning (GRPO), SymPy, torchvision, scikit-learn, OpenCV, ONNX, Albumentations

**Data:** pandas, NumPy, CVAT, Roboflow, web scraping (requests, BeautifulSoup)

**Systems and tools:** Rust (Candle), GGUF Quantization, CUDA, Linux (Ubuntu), Git/GitLab, Docker, Slurm, FastAPI, Jupyter Notebook, ComfyUI

**Soft skills:** Effective communication, teamwork, problem solving, adaptability, time management

# **LANGUAGES**

**Turkish:** Native

**English:** B2 (upper-intermediate)

**German:** A2 (current level); B1 course in progress, exam planned Feb 2027

# **VOLUNTEERING AND ACTIVITIES**

- **European Solidarity Corps (Erasmus+):** Volunteer, Preschool Kindergarten, Ankara | Jan 2025 \- Present. Conducted painting and creative activities with children.

- **Hugging Face:** Volunteer | Jan 2026 \- Present: Assisting with translating courses into Turkish.

# **CERTIFICATIONS**

- Hugging Face: Reinforcement Learning Course | Apr 2026

- Anthropic: Advanced Topics: Model Context Protocol (MCP) | Apr 2026