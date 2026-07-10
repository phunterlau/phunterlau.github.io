# The Model Chooses a Future Before It Says a Token

## What I learned from ICML 2026 through interpretability, alignment, post-training, and agent safety

ICML is an excellent fitness program disguised as a machine learning conference. My phone recorded roughly 20,000 steps a day, so this review has a natural methodological limitation: it covers eight poster sessions and is bounded by my walking distance.

![alt text](/images/icml_snapshot.jpg)

I filtered those sessions through the questions closest to my research: how aligned behavior is represented, how post-training changes model internals, how safety mechanisms fail under distribution shift, how agents decide when to act, and how interpretability can move from explaining behavior toward changing it.

One idea kept resurfacing:

> **A model often chooses a future before the final answer makes that choice visible.**

The commitment may happen through an internal feature, an expert route, a reasoning branch, a retrieved passage, a social-response channel, or a tool action. I use *margin* loosely here to mean the model’s relative preference between such competing futures: one reasoning branch over another, refusal over compliance, correct evidence over a distractor, or a safe action over an unsafe one.

Before ICML, I often asked which feature, neuron, or direction controlled a behavior.

After eight poster sessions—and many corridors between them—the more useful question became:

> **Where does the decision first become visible, when does it become causal, and is there still time to intervene?**

---

## 1. Reasoning, steering, and interpretability

### Thinking helps when it improves the next decision

Several papers converged on a view of reasoning as a sequence of local decisions rather than a monolithic chain of thought.

[How does Chain of Thought decompose complex tasks?](https://arxiv.org/abs/2604.08872) models reasoning as the decomposition of a difficult classification problem into a tree of smaller ones. The analysis produces an optimal depth: shallow reasoning leaves the original decision too hard, while excessive depth accumulates new opportunities for error. [SmartThinker](https://arxiv.org/abs/2603.08000) reaches a related conclusion from the training side. A fixed length penalty can compress difficult solutions too aggressively, so the desired reasoning budget should depend on the problem and on the distribution of successful response lengths. ([arXiv][1])

For a reasoning state ($h_t$), the object I care about can be written as

$$
\Delta_t =  \log p\left(\text{good next branch}\mid h_t\right) - \log p\left(\text{bad next branch}\mid h_t\right).
$$

The branch might be a mathematical subgoal, a correction to an earlier assumption, a safe continuation, or a decision to verify rather than guess. A useful reasoning step improves the future decision landscape. Overthinking begins when new steps add low-margin branches without resolving the old uncertainty.

[SafeThink](https://arxiv.org/abs/2602.11096) made this temporal view particularly concrete. Its main empirical result is that many unsafe reasoning trajectories can be redirected with a short corrective prefix during the first few reasoning steps. The important object is therefore not merely whether the final answer is safe, but how early the trajectory remains recoverable. ([arXiv][2])

This resembles a classical feedback-control problem. The intervention should arrive after enough evidence of drift has appeared, yet before the trajectory has settled into a harmful basin. Heavy intervention everywhere wastes control effort and increases collateral effects; late intervention edits the surface after the decisive branch has already been taken.

### Sparse routes carry large behavioral effects

The same structure appeared inside model architectures.

[Sparse Models, Sparse Safety](https://arxiv.org/abs/2602.08621) shows that MoE safety can depend on a small set of router decisions: manipulating a few safety-critical routers can redirect computation toward unsafe routes. [TraceRouter](https://arxiv.org/abs/2601.21900) extends the routing view beyond explicit MoE routers by tracing harmful semantic influence through cross-layer feature paths. ([arXiv][3])

Attention provided another version of the story. [Surgery](https://arxiv.org/abs/2602.05228) links harmful fine-tuning to changes in attention-sink divergence and regularizes that statistic during adaptation. [The Structural Origin of Attention Sink](https://arxiv.org/abs/2605.06611) traces sink formation back to variance discrepancies in value aggregation, their amplification by super neurons, and resulting dimensional imbalance. The interesting commonality is how a small statistical asymmetry can be amplified into a stable routing pattern. ([arXiv][4])

This makes sparsity both attractive and dangerous. Sparse mechanisms offer interpretable handles, yet a safety property concentrated in a few routes can also be bypassed through small perturbations or post-training updates. The relevant question is therefore not simply whether a component has causal leverage. A useful control point needs enough behavioral reach, a tolerable side-effect profile, and a nontrivial operating window.

### Steering has a geometry and a schedule

Two steering papers clarified why finding a direction is only part of the problem.

[Spherical Steering](https://arxiv.org/abs/2602.08169) replaces additive displacement with rotation on a representation sphere, separating angular movement toward a target behavior from radial norm distortion. [Steer Like the LLM](https://arxiv.org/abs/2605.03907) starts from another observation: prompting does not induce the same activation shift at every token. Prompt Steering Replacement learns state- and position-dependent coefficients that approximate the intervention pattern produced by a successful prompt. ([arXiv][5])

Together with early trajectory correction, these papers outline three coupled questions:

* **When** is the trajectory still steerable?
* **Where** in the model is the target distinction represented and used?
* **How** should the intervention move the state without pushing it off-manifold?

Before ICML, I thought of steering mainly as direction discovery.

After ICML, steering looks more like a policy over layer, time, geometry, and dose.

### Visibility, causality, and long-horizon influence

A useful hidden gem was [Automatic Layer Selection for Hallucination Detection](https://arxiv.org/abs/2605.26366). Its FEPoID criterion selects intermediate layers where hallucination-related signals become especially detectable. The broader lesson is that “where to look” is already part of the mechanism: a signal can be absent from one layer, linearly visible in another, and causally committed somewhere else. ([arXiv][6])

For each behavior, I now want to separate four questions:

1. Is the relevant information represented?
2. Is it visible in a usable coordinate system?
3. Is it causally used?
4. Is the outcome still changeable at that point?

[Towards Long-Horizon Interpretability](https://arxiv.org/abs/2602.01914) supplies an important complementary method. FlashTrace attributes multi-token spans and recursively propagates influence through intermediate reasoning tokens back toward source inputs. This is closer to the causal object needed for reasoning models: an early assumption may have little effect on the immediately following token while strongly shaping an entire later derivation. ([arXiv][7])

Representation units and cross-model coordinates remain part of the same puzzle. [Towards Atoms of Large Language Models](https://arxiv.org/abs/2509.20784) asks whether neurons and standard SAE features are the right stable units. [Multi-Way Representation Alignment](https://arxiv.org/abs/2602.06205) maps multiple models into a shared representation universe rather than constructing inconsistent pairwise correspondences. A mechanistic claim becomes more convincing when the same causal quantity survives a change of layer, unit, model, or coordinate system. ([arXiv][8])

### External context is also a routing system

[The First Drop of Ink](https://arxiv.org/abs/2605.10828) extends the routing perspective beyond model internals. A small proportion of semantically plausible distractors causes a sharp initial loss in long-context accuracy, while adding more distractors produces diminishing additional damage. The first strong distractor captures enough attention to redirect the evidence path; the rest mostly compete within an already contaminated context. ([arXiv][9])

This matters for RAG and agents with persistent memory. Retrieved context is not passive storage. Correct and misleading passages compete to become the basis of the answer. A useful evidence margin is therefore the model’s preference for the best supported source over the strongest plausible distractor.

---

## 2. Alignment, self-report, and evaluation

### Saying is another behavior

The self-report and explanation papers challenged a common shortcut in alignment evaluation. A model’s account of its personality, beliefs, or reasons is still a generated output rather than privileged access to its internal computation.

[The Personality Illusion](https://arxiv.org/abs/2509.03730) finds that instruction tuning can stabilize self-reported personality traits even when those reports remain weak predictors of behavior. Persona prompts move questionnaire answers much more consistently than they move risk-taking, sycophancy, or other behavioral measures. [What LLMs Explain Is Not What They Believe](https://arxiv.org/abs/2606.28615) formalizes explanation sufficiency relative to a distribution of alternative inputs and finds that many explanations remain insufficient even under alternatives generated from the model’s own input beliefs. ([arXiv][10])

A conversation at the second poster suggested a more precise follow-up question. Explanation failure can arise at several stages:

* the relevant feature is absent;
* it is represented but not used;
* it is used but inaccessible to verbalization;
* it reaches the verbal channel but is translated unfaithfully;
* a separate social-response circuit produces a plausible rationale.

Anthropic’s [Natural Language Autoencoders](https://www.anthropic.com/research/natural-language-autoencoders) make this distinction experimentally approachable by training an activation verbalizer together with a reconstructor that maps the verbalization back into activation space. The related [global-workspace study](https://transformer-circuits.pub/2026/workspace/index.html) asks which representations are available for report, flexible reasoning, and intervention. ([Anthropic][11])

This gives a better framing than treating self-report as either honest introspection or empty performance. Self-report, refusal, sycophancy, evaluation awareness, and behavioral safety may share upstream representations while using different downstream readouts. Their natural activations may correlate even when causal interventions separate them.

Before ICML, I viewed self-report mainly as a questionable evaluation signal.

After ICML, it looks like a useful mechanistic target: one channel among several that may reveal how post-training organizes social and alignment-related behavior.

### Judges and preferences are measurement systems

The same caution applies to the evaluators used to train and study models.

[A Coin Flip for Safety](https://arxiv.org/abs/2603.06594) audits LLM safety judges under shifts in attack, victim model, and data. Judge reliability can fall toward chance, and attack optimization can raise measured success by exploiting evaluator failure rather than producing more genuinely harmful content. ([arXiv][12])

[Measuring Human Preferences in RLHF Is a Social Science Problem](https://arxiv.org/abs/2604.03238) moves one step further upstream. A pairwise annotation is an observed response produced by an elicitation procedure. It may reflect a stable preference, a preference constructed in context, a framing artifact, or no settled attitude at all. The paper argues that measurement validity should precede preference aggregation. ([arXiv][13])

This is an important cross-disciplinary import. Behavioral science has spent decades separating latent constructs from the instruments used to measure them. Alignment pipelines often collapse those layers into one object called “human preference.”

For evaluation, the practical consequence is straightforward: a change in judge score is evidence only after the judge has been validated on the transformed output distribution. Otherwise, a steering method, jailbreak attack, or agent policy may simply be moving examples across the evaluator’s blind spots.

---

## 3. Agents and consequential actions

Agent safety becomes most concrete at the boundary between thought and action.

[Learning When to Act or Refuse](https://arxiv.org/abs/2603.03205) treats refusal as a first-class agent action within a plan–check–act loop and learns from pairwise comparisons of complete tool-use trajectories. [Think Twice Before You Act](https://arxiv.org/abs/2505.11063) introduces Thought-Aligner, which rewrites unsafe intermediate thoughts before they determine the next action. ([arXiv][14])

For a state (s_t), the relevant quantity can be expressed as

$$
\Delta^{\mathrm{act}}_t = \log p\left( \text{safe check, refusal, or revision} \mid s_t \right) - \log p\left(\text{unsafe tool action}
\mid s_t
\right).
$$

A careful-looking rationale is insufficient when the action distribution remains unchanged. The consequential question is whether the safety mechanism shifts the probability of reading a private file, sending an email, entering credentials, modifying a database, or executing a command.

Safety engineering offers the useful analogy here. Interventions should be placed before irreversible transitions, with stronger checks for actions whose effects are difficult to undo. This turns agent safety from a wrapper around the final answer into a property of the state-transition policy.

Before ICML, I thought of agent safety mostly as guarding an autonomous system.

After ICML, I think of it as maintaining a safe action margin at each consequential boundary.

---

## 4. Skills, composition, and the new supply chain

The skill-based agent papers made composition impossible to ignore.

[SkillTrojan](https://arxiv.org/abs/2604.06811) distributes malicious logic across benign-looking skills and reconstructs the payload only under particular trigger and composition conditions. [A Theoretical Game of Attacks via Compositional Skills](https://arxiv.org/abs/2605.01034) studies the same structural problem at the level of linguistic and cognitive transformations: translation, roleplay, encoding, analogy, and indirection can each look harmless while their composition reveals a harmful objective. ([arXiv][15])

The key security property therefore belongs to the composition graph rather than to any isolated component.

A skill can pass unit tests, contain plausible documentation, and remain useful on clean tasks while still participating in a triggered multi-skill payload. A meaningful SkillBOM needs to describe more than text similarity or code provenance. It should include tool permissions, memory access, side effects, trigger conditions, compatible composition partners, and observed execution traces.

The closest analogy is software supply-chain security, although agent dependencies are more varied. The graph can include executable code, natural-language instructions, retrieved documents, memory entries, API schemas, and private conventions shared among agents.

This also changes how fuzzy identity should be evaluated. A robust skill fingerprint should tolerate benign refactoring while remaining sensitive to changes in permissions, side effects, and compositional behavior.

---

## 5. Editable artifacts for reliable agents

The most inspiring agent-design pattern came from theorem proving and code security.

[Editable Proof Sketch](https://openreview.net/forum?id=mI3K0e1KsN) represents a proof plan as a dependency-aware artifact. When one subgoal fails, the system repairs the affected region while preserving proven parts that do not depend on it. [CVE-Factory](https://arxiv.org/abs/2602.03012) takes sparse vulnerability metadata and constructs executable environments, tasks, tests, and validation procedures at scale. ([arXiv][16])

Both suggest a better substrate for agent reasoning than an ever-growing natural-language scratchpad.

A scientific agent should maintain an editable research object containing hypotheses, evidence, metrics, experiments, code artifacts, failures, dependencies, and revisions. A coding agent should preserve passing tests while localizing the cause of a failing one. A security agent should operate inside an executable environment whose success criteria can be independently checked.

The shared workflow is:

1. externalize the task into a structured artifact;
2. make dependencies explicit;
3. verify atomic parts with an external checker;
4. preserve validated components;
5. repair only the region affected by failure.

Formal methods and software testing have relied on this structure for decades. Their lesson for agents is deeper than “add a verifier.” The artifact must be organized so that verifier feedback identifies what should change and what should remain untouched.

This also produces more meaningful intervention points. A failed subgoal, unsupported claim, unsafe action, or invalid dependency is a better unit of repair than an arbitrary token in a long chain of thought.

---

## A focused research agenda after ICML

Across the eight sessions, the recurring pattern was a decision becoming progressively more visible, more causally effective, and eventually harder to reverse.

The route may be an internal feature, an attention head, an MoE expert path, a reasoning branch, an explanation channel, a retrieved evidence path, a skill composition, or a sequence of tool actions. The competing futures may be correct and incorrect, refusal and compliance, grounded and distracted, truthful and sycophantic, or safe and unsafe.

The questions I am taking forward are:

* Where does a behavior first become visible inside the model?
* When does that signal become causally involved in the outcome?
* How do alignment and other post-training stages reshape it?
* Which apparently related behaviors share an upstream mechanism, and which only correlate at the surface?
* How can an intervention change the trajectory while preserving neighboring capabilities?
* How can agents externalize their state into editable and verifiable artifacts?
* How should evaluators be stress-tested before their scores are treated as evidence?

Twenty thousand steps a day was enough to leave me with one compact conclusion:

> **Find the earliest reliable decision point, determine whether it is causal, and intervene before the wrong future becomes difficult to reverse.**

---

# References

1. [How does Chain of Thought decompose complex tasks?](https://arxiv.org/abs/2604.08872)
2. [SmartThinker: Progressive Chain-of-Thought Length Calibration for Efficient Large Language Model Reasoning](https://arxiv.org/abs/2603.08000)
3. [Safety Recovery in Reasoning Models Is Only a Few Early Steering Steps Away](https://arxiv.org/abs/2602.11096)
4. [Sparse Models, Sparse Safety: Unsafe Routes in Mixture-of-Experts LLMs](https://arxiv.org/abs/2602.08621)
5. [TraceRouter: Robust Safety for Large Foundation Models via Path-Level Intervention](https://arxiv.org/abs/2601.21900)
6. [Surgery: Mitigating Harmful Fine-Tuning for Large Language Models via Attention Sink](https://arxiv.org/abs/2602.05228)
7. [The Structural Origin of Attention Sink: Variance Discrepancy, Super Neurons, and Dimension Disparity](https://arxiv.org/abs/2605.06611)
8. [Spherical Steering: Geometry-Aware Activation Rotation for Language Models](https://arxiv.org/abs/2602.08169)
9. [Steer Like the LLM: Activation Steering that Mimics Prompting](https://arxiv.org/abs/2605.03907)
10. [Automatic Layer Selection for Hallucination Detection](https://arxiv.org/abs/2605.26366)
11. [Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs](https://arxiv.org/abs/2602.01914)
12. [Towards Atoms of Large Language Models](https://arxiv.org/abs/2509.20784)
13. [Multi-Way Representation Alignment](https://arxiv.org/abs/2602.06205)
14. [The First Drop of Ink: Nonlinear Impact of Misleading Information in Long-Context Reasoning](https://arxiv.org/abs/2605.10828)
15. [The Personality Illusion: Revealing Dissociation Between Self-Reports & Behavior in LLMs](https://arxiv.org/abs/2509.03730)
16. [What LLMs Explain Is Not What They Believe: Evaluating Explanation Sufficiency Under Models’ Own Input Beliefs](https://arxiv.org/abs/2606.28615)
17. [Natural Language Autoencoders: Turning Claude’s Thoughts into Text](https://www.anthropic.com/research/natural-language-autoencoders)
18. [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)
19. [A Coin Flip for Safety: LLM Judges Fail to Reliably Measure Adversarial Robustness](https://arxiv.org/abs/2603.06594)
20. [Measuring Human Preferences in RLHF Is a Social Science Problem](https://arxiv.org/abs/2604.03238)
21. [Learning When to Act or Refuse: Guarding Agentic Reasoning Models for Safe Multi-Step Tool Use](https://arxiv.org/abs/2603.03205)
22. [Think Twice Before You Act: Enhancing Agent Behavioral Safety with Thought Correction](https://arxiv.org/abs/2505.11063)
23. [SkillTrojan: Backdoor Attacks on Skill-Based Agent Systems](https://arxiv.org/abs/2604.06811)
24. [A Theoretical Game of Attacks via Compositional Skills](https://arxiv.org/abs/2605.01034)
25. [Editable Proof Sketch for Automated Theorem Proving](https://openreview.net/forum?id=mI3K0e1KsN)
26. [CVE-Factory: Scaling Expert-Level Agentic Tasks for Code Security Vulnerability](https://arxiv.org/abs/2602.03012)

[1]: https://arxiv.org/abs/2604.08872 "[2604.08872] How does Chain of Thought decompose complex tasks?"
[2]: https://arxiv.org/abs/2602.11096?utm_source=chatgpt.com "Safety Recovery in Reasoning Models Is Only a Few Early Steering Steps Away"
[3]: https://arxiv.org/abs/2602.08621?utm_source=chatgpt.com "Sparse Models, Sparse Safety: Unsafe Routes in Mixture-of-Experts LLMs"
[4]: https://arxiv.org/abs/2602.05228 "[2602.05228] Surgery: Mitigating Harmful Fine-Tuning for Large Language Models via Attention Sink"
[5]: https://arxiv.org/abs/2602.08169 "[2602.08169] Spherical Steering: Geometry-Aware Activation Rotation for Language Models"
[6]: https://arxiv.org/abs/2605.26366?utm_source=chatgpt.com "Automatic Layer Selection for Hallucination Detection"
[7]: https://arxiv.org/abs/2602.01914?utm_source=chatgpt.com "Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs"
[8]: https://arxiv.org/abs/2509.20784 "[2509.20784] Towards Atoms of Large Language Models"
[9]: https://arxiv.org/abs/2605.10828?utm_source=chatgpt.com "The First Drop of Ink: Nonlinear Impact of Misleading Information in Long-Context Reasoning"
[10]: https://arxiv.org/abs/2509.03730 "[2509.03730] The Personality Illusion: Revealing Dissociation Between Self-Reports & Behavior in LLMs"
[11]: https://www.anthropic.com/research/natural-language-autoencoders "Natural Language Autoencoders \ Anthropic"
[12]: https://arxiv.org/abs/2603.06594?utm_source=chatgpt.com "A Coin Flip for Safety: LLM Judges Fail to Reliably Measure Adversarial Robustness"
[13]: https://arxiv.org/html/2604.03238v1?utm_source=chatgpt.com "Measuring Human Preferences in RLHF is a Social ..."
[14]: https://arxiv.org/abs/2603.03205 "[2603.03205] Learning When to Act or Refuse: Guarding Agentic Reasoning Models for Safe Multi-Step Tool Use"
[15]: https://arxiv.org/abs/2604.06811 "[2604.06811] SkillTrojan: Backdoor Attacks on Skill-Based Agent Systems"
[16]: https://arxiv.org/abs/2602.03012 "[2602.03012] CVE-Factory: Scaling Expert-Level Agentic Tasks for Code Security Vulnerability"
