# arXiv Submission Framing Guidelines

To satisfy academic reviewers and handle potential peer-review criticism, the draft of the paper must be updated with the following framing adjustments.

---

## 1. Narrowing the Core Thesis (Action Item 3)
* **Problem**: Claiming that the "8B model defeats Scaling Laws" is too aggressive and easily refuted by the AssetOps #9 ranking.
* **Reframing**: Focus on **efficiency**, **hybrid architectures**, and **domain augmentation**.
* **Suggested Text for Abstract/Introduction**:
  > *"While scaling parameters remains the primary vector for general-purpose frontier models, enterprise-specific IT operations require high-precision grounding and domain-specific constraints. We demonstrate that an 8B parameters base model, augmented with a high-dimensional symbolic resonance network (QMEM) and lightweight domain-specific cartridges, matches or exceeds frontier models in target operational tasks. This highlights that targeted neurosymbolic augmentation is a highly efficient alternative to pure scaling for domain-restricted enterprise AI deployment."*

---

## 2. Framing SRE Performance as Alert Triage (Action Item 4)
* **Problem**: Claiming the model has "advanced SRE reasoning" will be rejected because the task is entity extraction from JSON alerts.
* **Reframing**: Clearly define the task as **structured alert triage** and **entity extraction**.
* **Suggested Text for Evaluation Section**:
  > *"The Site Reliability Engineering (SRE) benchmark evaluates the model's capacity to ingest a structured JSON list of system alerts and extract the service identity causing the failure. Although this task is primarily structured alert triage and entity filtering rather than multi-step causal reasoning, frontier models frequently fail due to context formatting issues or hallucinations. By grounding the generation loop with a structured alert parser and QMEM correlation rules, SODALS achieves 100% accuracy [95% CI: 90.82% - 100.00%], demonstrating superior performance in high-precision alert triage."*

---

## 3. Reporting AssetOps #9 Rank Transparently (Action Item 5)
* **Problem**: Hiding the #9 rank in AssetOps (FailureSensorIQ) will look like cherry-picking.
* **Reframing**: Report it openly and explain the technical reasons (strict all-or-nothing set matching on multi-answer questions).
* **Suggested Text for Results Section**:
  > *"On the FailureSensorIQ (AssetOps) benchmark, SODALS achieves an accuracy of 64.67%, ranking #9 globally. This performance is close to frontier models (e.g., matching Claude 4.5 Sonnet) but highlights the limitations of an 8B parameter model when dealing with complex, multi-variable sensor diagnostic scenarios under strict set-matching criteria (where partial correctness receives zero score). In future work, incorporating iterative multi-turn verification loops will address this performance gap."*

---

## 4. Documenting Prompt Engineering Choices
* **Problem**: Reviewers might ask if prompt engineering gives an unfair advantage.
* **Reframing**: Document the exact prompts used for single‑answer and multi‑answer questions transparently.
* **Suggested Text for Appendix**:
  > *"To ensure optimal instruction following, the prompt structure adapts dynamically to the question metadata. If the target dataset contains multi-answer questions, the system prompt instructs the model to 'select ALL correct options and format them as a comma-separated list' instead of requesting a single option. This ensures that prompt context matches the task requirements for all models evaluated."*
