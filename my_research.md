# My Research

> **I study how behaviors are represented, controlled, and broken in large language models and AI agents.**

My research asks a recurring question: **where does a learned behavior live, how much causal control do we actually have over it, and where do post-training safety mechanisms fail?** I approach this from both mechanistic and systems perspectives, connecting internal model structure to observable failures in alignment, reward modeling, tokenization, tool use, and agent execution.

A common pattern across my work is to find **small, measurable interfaces where large behavioral changes emerge**: a sparse set of model components, a logit decision boundary, a tokenization boundary, or the interface between an agent and an acquired skill. I use causal interventions to understand these mechanisms, characterize when they offer genuine behavioral control, and study what their failure modes reveal about post-training and deployment.

<details markdown="1">
<summary><strong>Research areas</strong></summary>

* **Mechanistic interpretability and controllability** — Locating behavioral circuits with lightweight causal interventions, studying sparse sets of neurons and components, and characterizing when steering produces coherent behavioral control rather than model collapse.

* **Post-training and alignment robustness** — Studying refusal, preference optimization, RLHF/RLVR, and how apparently robust aligned behaviors can fail under small, structured perturbations.

* **LLM-as-a-Judge and reward models** — Analyzing vulnerabilities in learned evaluators and how weaknesses in binary or scalar reward signals propagate into post-training.

* **Agents, tools, and skills** — Understanding tool-use behavior, verifying whether agent skills do what they claim, and developing integrity and identity mechanisms for dynamically acquired capabilities.

* **Model interfaces as failure surfaces** — Investigating boundaries such as tokenization, sampling, and tool invocation where small representational mismatches can become large behavioral failures.

</details>

I am currently especially interested in extending these ideas to **tool use and other post-training behaviors**: whether small sets of components form reproducible control regimes for behaviors learned during alignment and agent training, and whether these mechanisms can give us better ways to diagnose and control increasingly autonomous systems.

## Selected Publications

### 2026

**AdvJudge-Zero: Binary Decision Flips in LLM-as-a-Judge via Adversarial Control Tokens**
Tung-Ling Li, Yuhao Wu, **Hongliang Liu**
**Accepted at NeurIPS 2026** · [arXiv:2512.17375](https://arxiv.org/abs/2512.17375)
Studies sharp decision vulnerabilities in LLM judges and how weaknesses in learned evaluators can affect downstream reward-based training.

**Logit-Gap Steering: Efficient Short-Suffix Jailbreaks for Aligned Large Language Models**
Tung-Ling Li, **Hongliang Liu**
**Accepted at NeurIPS 2026** · [arXiv:2506.24056](https://arxiv.org/abs/2506.24056)
Uses the refusal-versus-affirmation logit gap to expose and manipulate sharp decision boundaries introduced by alignment.

**Breaking Safety at the Token Boundary: How BPE Tokenization Creates Exploitable Gaps in LLM Alignment**
Tung-Ling Li, **Hongliang Liu**, Yuhao Wu
[arXiv:2607.01239](https://arxiv.org/abs/2607.01239)
Studies how BPE fragmentation creates a structural mismatch between safety training and inference, exposing exploitable gaps at the token boundary.

**The Decomposition Is the Fingerprint: Per-Component Identity for Agent Skills**
**Hongliang Liu**, Yuhao Wu, Tung-Ling Li
[arXiv:2606.31272](https://arxiv.org/abs/2606.31272)
Introduces compact per-component fingerprints for agent skills, separating prompt, code, and tool identity to detect lineage, reuse, and tampering.

**Leverage Is Not Reach: A Control-Window Law for Single-Neuron Steering in Language Models**
**Hongliang Liu**
[arXiv:2606.19831](https://arxiv.org/abs/2606.19831)
Develops a budget-normalized control-window framework for predicting when neuron-level interventions can coherently control behavior and when they instead hit a collapse boundary.

**Behavioral Integrity Verification for AI Agent Skills**
Yuhao Wu, Tung-Ling Li, **Hongliang Liu**
[arXiv:2605.11770](https://arxiv.org/abs/2605.11770)
Formalizes behavioral integrity for agent skills by comparing declared capabilities with their actual behavior across code, instructions, and metadata.

**Perturbation Probing: A Two-Pass-per-Prompt Diagnostic for FFN Behavioral Circuits in Aligned LLMs**
**Hongliang Liu**, Tung-Ling Li, Yuhao Wu
[arXiv:2604.27401](https://arxiv.org/abs/2604.27401)
Introduces a forward-only perturbation method for discovering task-specific FFN behavioral circuits and distinguishing different structures of learned behavior.
