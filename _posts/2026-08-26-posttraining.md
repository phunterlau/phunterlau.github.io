# Deriving LLM Post-Training from First Principles

I have been reading Nathan Lambert's [RLHF Book](https://rlhfbook.com/) and following his lecture and video series. What I appreciate most is the way he has systematically organized a rapidly evolving field into an engineering framework. SFT, reward models, PPO, DPO, RLVR, distillation, regularization, infrastructure, and newer agentic methods often appear as separate topics, while his treatment makes their engineering relationships much easier to see.

I wanted to approach the same landscape from a complementary direction. With my physics background, I naturally ask whether a complicated collection of methods can be reconstructed from a small set of primitives, like Coulomb's law for classical electromagnetism. One can begin with a simple interaction such as $F = k_e \frac{q_1 \times q_2}{r^2}$, then introduce additional structure only when the current description becomes insufficient. To be clear: Maxwell's equations are not derived from Coulomb's law alone, but we haven't had Maxwell's equations for LLM, right?

![alt text](/images/posttraining.jpg)

Start from SFT and cross-entropy, ask exactly what information it contains, identify what it cannot express, then add one new degree of freedom at a time. Following that path, much of modern post-training can be organized around four conceptual transitions: from likelihood to preference, from offline projection to on-policy optimization, from sequence generation to environment interaction, and from scalar reward to distributional feedback. The individual algorithms then become branches of these transitions rather than isolated inventions.

This post works like an equation heavy note, but don't be afraid. I will use a small set of annotations consistently: $x$ denotes the input or prompt, $y$ a generated response, $a_t$ the token or action taken at step $t$, and $s_t$ the corresponding model state or interaction context. The policy is $\pi_\theta$, while $p_{\mathcal D}$ denotes an external data distribution. Rewards are written as $R$ or $r$, advantages as $A_t$, and $\pi_{\mathrm{ref}}$ denotes a reference policy when one is needed. For agentic settings, $\tau$ denotes a full trajectory and $P_{\mathrm{env}}$ the environment transition dynamics. I will use $D_{\mathrm{KL}}(p\Vert q)$ in the standard direction, with the first argument defining the sampling distribution inside the expectation.

Disclaimer: this post is edited with LLM assistance.

# I. From likelihood to preference

## Cross-entropy as the primitive interaction

Let an autoregressive policy be (as in many GPT-style models)

$$
\pi_\theta(y\mid x)
=
\prod_{t=1}^{T}
\pi_\theta(y_t\mid x,y_{<t}).
$$

Given demonstrations sampled from a data distribution,

$$
(x,y)\sim p_{\mathcal D},
$$

supervised fine-tuning minimizes

$$
\mathcal L_{\mathrm{SFT}}
=
-
\mathbb E_{(x,y)\sim p_{\mathcal D}}
\left[
\log \pi_\theta(y\mid x)
\right].
$$

At token level,

$$
\mathcal L_{\mathrm{SFT}}
=
-
\mathbb E
\left[
\sum_t
\log \pi_\theta(y_t\mid x,y_{<t})
\right].
$$

The objective contains a remarkably simple form of supervision. At each training state, the data identifies a target token and increases its probability. If

$$
p_v=\operatorname{softmax}(z)_v,
$$

then the gradient with respect to a logit is (p-y) in numpy or:

$$
\frac{\partial \mathcal L}{\partial z_v}
=
p_v-\mathbf 1[v=y^*].
$$

The target token is pushed upward in probability, while alternatives are suppressed through softmax normalization. This is enough to support imitation, instruction following, and large amounts of behavioral shaping because the pretrained model already contains a rich distribution of latent capabilities.

The limitation follows from the same equation. Cross-entropy identifies a target but carries no explicit representation of relative quality among alternatives. Suppose two complete responses $y_A$ and $y_B$ are both plausible, while one is more factual, more concise, safer, or better calibrated. If the dataset contains only $y_B$, the optimization increases its likelihood, but the loss does not directly encode the relation

$$
y_B \succ y_A.
$$

This becomes important whenever post-training is less about teaching an unseen capability and more about shifting probability between behaviors that already exist in the pretrained model. That missing relational structure motivates preference optimization.

## Preference introduces a relative coordinate

A preference pair adds the ordering

$$
y_w \succ y_l.
$$

A common latent-variable model represents this ordering through a scalar reward

$$
r_\phi(x,y),
$$

with pairwise preference probability

$$
P(y_w\succ y_l\mid x)
=
\sigma
\left(
r_\phi(x,y_w)-r_\phi(x,y_l)
\right).
$$

The corresponding loss is

$$
\mathcal L_{\mathrm{RM}}
=
-
\mathbb E
\left[
\log
\sigma
\left(
r_\phi(x,y_w)-r_\phi(x,y_l)
\right)
\right].
$$

The important change is that supervision now constrains a difference between behaviors rather than simply assigning one behavior positive probability. SFT provides an absolute target $y^*$, while preference learning constrains the relative direction $y_w-y_l$. This is a more natural representation for many alignment problems because the pretrained model may already support both candidate behaviors. The remaining task is to move the decision boundary.

That distinction applies to properties such as factuality, verbosity, tone, refusal boundaries, confidence calibration, formatting, and tool-use style. Preference data is useful precisely because these dimensions often live inside an existing behavioral manifold rather than outside it.

## RLHF turns preference into a policy objective

Once a reward model provides a scalar estimate of preference, the natural next step is to optimize the policy against it:

$$
\max_\pi
\mathbb E_{y\sim\pi}
[r(x,y)].
$$

For a large pretrained model, unconstrained reward maximization can move the policy into poorly modeled regions of behavior space. The reward model is an approximation, so aggressive optimization may exploit inaccuracies in the reward surface or damage useful capabilities inherited from pretraining. KL-regularized RLHF addresses this by adding a reference policy:

$$
\mathcal J(\pi)
=
\mathbb E_{y\sim\pi}
[r(x,y)]
-
\beta
D_{\mathrm{KL}}
\left(
\pi
\Vert
\pi_{\mathrm{ref}}
\right).
$$

The reward defines a direction of improvement, while the KL term controls how far the policy can move away from a known distribution. More importantly for the framework developed here, the expectation is taken over samples from the policy being optimized. This begins a shift in the source of training experience, from externally supplied demonstrations toward behavior generated by the model itself.

The optimization problem also has a useful closed-form optimum:

$$
\pi^*(y\mid x)
=
\frac{1}{Z(x)}
\pi_{\mathrm{ref}}(y\mid x)
\exp
\left(
\frac{r(x,y)}{\beta}
\right).
$$

Equivalently,

$$
r(x,y)
=
\beta
\log
\frac{
\pi^*(y\mid x)
}{
\pi_{\mathrm{ref}}(y\mid x)
}
+
\beta\log Z(x).
$$

That relationship provides the starting point for DPO.

## DPO as an analytical branch of the RLHF objective

DPO substitutes the optimal-policy relation directly into the preference model. Define

$$
\Delta_\theta
=
\log
\frac{
\pi_\theta(y_w\mid x)
}{
\pi_{\mathrm{ref}}(y_w\mid x)
}
-
\log
\frac{
\pi_\theta(y_l\mid x)
}{
\pi_{\mathrm{ref}}(y_l\mid x)
}.
$$

The DPO objective becomes

$$
\mathcal L_{\mathrm{DPO}}
=
-
\mathbb E
\left[
\log\sigma
\left(
\beta\Delta_\theta
\right)
\right].
$$

Its gradient has the structure

$$
\nabla_\theta\mathcal L_{\mathrm{DPO}}
\propto
-
\alpha
\left[
\nabla_\theta\log\pi_\theta(y_w\mid x)
-
\nabla_\theta\log\pi_\theta(y_l\mid x)
\right],
$$

where

$$
\alpha
=
\beta
\sigma(-\beta\Delta_\theta).
$$

Compared with SFT, the essential difference is visible directly in the gradient. Supervised learning reinforces a demonstrated response, while DPO applies **opposite** relative pressure to the preferred and rejected responses. The update strength also depends on how much the current policy already satisfies the preference ordering. This makes DPO well suited to reshaping boundaries between behaviors that are already supported by the model.

Its place in the broader framework is also precise. The optimal-policy parameterization comes from KL-regularized RL, but the samples usually come from a fixed preference dataset:

$$
(x,y_w,y_l)
\sim
\mathcal D_{\mathrm{pref}}.
$$

DPO therefore inherits an RL-derived target geometry while retaining offline sampling. This combination explains much of its appeal and also why it behaves differently from online RL when exploration matters. Other "PO"s like SimPO, IPO, KTO, ORPO, and related methods can be viewed as nearby branches that modify the reference policy, margin, pair structure, normalization, or statistical form of this same preference-learning problem.

The first major transition can therefore be summarized as a progression from likelihood, to relative quality, to reward-shaped policy optimization. The next transition changes the source of experience itself.

# II. From offline projection to on-policy optimization

## Policy gradient changes who generates the data

Suppose the policy generates its own response,

$$
y\sim\pi_\theta(y\mid x),
$$

and receives reward

$$
R(x,y).
$$

The objective is

$$
J(\theta)
=
\mathbb E_{y\sim\pi_\theta}
[R(x,y)].
$$

Using the log-derivative identity,

$$
\nabla_\theta \pi_\theta(y)
=
\pi_\theta(y)
\nabla_\theta\log\pi_\theta(y),
$$

we obtain

$$
\nabla_\theta J
=
\mathbb E_{y\sim\pi_\theta}
\left[
R(x,y)
\nabla_\theta
\log\pi_\theta(y\mid x)
\right].
$$

For an autoregressive policy,

$$
\nabla_\theta J
=
\mathbb E
\left[
\sum_t
R
\nabla_\theta
\log\pi_\theta(a_t\mid s_t)
\right].
$$

Subtracting a baseline gives

$$
A_t=R_t-b_t,
$$

and therefore

$$
\nabla_\theta J
=
\mathbb E
\left[
\sum_t
A_t
\nabla_\theta
\log\pi_\theta(a_t\mid s_t)
\right].
$$

This equation is one of the central roots of modern post-training. The log-probability gradient remains familiar, while the coefficient becomes an advantage rather than a supervised target. The deeper transition is in the expectation itself. Training now occurs on samples produced by the current or recent policy, so the model participates in constructing its own training distribution.

This on-policy structure becomes especially important when useful behaviors are difficult to write down as demonstrations. The model can explore trajectories that were never explicitly supplied by a teacher, and the feedback function determines which of those trajectories are reinforced.

## PPO adds trust to an on-policy update

Rollout data is usually produced by a policy that becomes stale during optimization. If $\pi_{\mathrm{old}}$ generated the sample while $\pi_\theta$ is being updated, the relevant importance ratio is

$$
\rho_t
=
\frac{
\pi_\theta(a_t\mid s_t)
}{
\pi_{\mathrm{old}}(a_t\mid s_t)
}.
$$

A policy-gradient estimator then contains the factor

$$
\rho_t A_t.
$$

Large ratios can produce unstable updates because a sample that was reasonable under the rollout policy may become poorly representative of the current policy. PPO controls this through clipping:

$$
\mathcal L_{\mathrm{PPO}}
=
-
\mathbb E
\left[
\min
\left(
\rho_t A_t,
\operatorname{clip}
(\rho_t,1-\epsilon,1+\epsilon)
A_t
\right)
\right].
$$

At this point the post-training problem contains three distinct ingredients: the distribution that generates experience, the signal used to evaluate that experience, and the mechanism that controls how strongly stale experience can modify the current policy. Many later RL variants can be understood as modifying one of these ingredients rather than introducing a new foundation.

The PPO clipping mechanism should also be distinguished from the KL penalty used in classical RLHF. Clipping constrains local updates relative to the rollout policy, while the KL penalty controls global drift from a reference model. They solve different regularization problems inside the same broader optimization system.

## RLVR and GRPO modify different parts of the same equation

RLVR changes the origin of reward. If a task admits an external verifier, reward can be defined directly from an observable outcome:

$$
R(x,y)
=
\begin{cases}
1, & \text{verified success},\\
0, & \text{failure}.
\end{cases}
$$

The policy-gradient structure remains

$$
\nabla J
=
\mathbb E_{\pi_\theta}
\left[
A_{\mathrm{verifier}}
\nabla_\theta
\log\pi_\theta
\right].
$$

The scientific improvement lies in the measurement process used to construct the advantage. Learned reward models estimate quality, while verifiers can connect reward to an externally checkable state. This is why mathematics, coding, formal verification, and some tool-use tasks provide particularly productive environments for RL.

GRPO modifies a different component. PPO often uses a learned value function $V_\phi(s)$ to estimate advantages, while GRPO uses multiple samples from the same prompt to create a relative baseline. Given rewards

$$
R_1,\ldots,R_G,
$$

a group-relative advantage can be written as

$$
A_i
=
\frac{
R_i-\bar R
}{
\sigma_R+\epsilon
}.
$$

The policy-gradient structure remains the same, while the estimator of $A$ changes:

$$
A_{\mathrm{critic}}
\rightarrow
A_{\mathrm{group}}.
$$

This distinction is useful because it separates two kinds of innovation that are often discussed together. RLVR improves the source of the reward signal, while GRPO changes the way that signal is converted into relative credit. GSPO, CISPO, and related methods continue this pattern by modifying ratio granularity, clipping behavior, normalization, or other terms inside the same on-policy framework.

# III. From sequences to dynamical systems

## Agent RL introduces environment dynamics

A reasoning task can often be approximated as a generation followed by evaluation:

$$
x
\rightarrow
y
\rightarrow
R(y).
$$

An agent interacts repeatedly with an external state. Its trajectory can be written as

$$
\tau
=
(s_0,a_0,s_1,a_1,\ldots,s_T,a_T),
$$

with environment dynamics

$$
P_{\mathrm{env}}
(s_{t+1}\mid s_t,a_t).
$$

The optimization objective becomes

$$
J(\theta)
=
\mathbb E_{
\tau\sim
(\pi_\theta,P_{\mathrm{env}})
}
[R(\tau)].
$$

The corresponding gradient remains familiar:

$$
\nabla_\theta J
=
\mathbb E_\tau
\left[
\sum_t
A_t
\nabla_\theta
\log\pi_\theta(a_t\mid s_t)
\right].
$$

The major new object is the environment transition process. For a coding agent, the next state depends on files edited, commands executed, compiler errors, tests, and previous tool calls. For a research or computer-use agent, future states similarly depend on earlier actions. The model therefore helps determine the distribution of states on which future learning occurs.

This makes agentic post-training a coupled dynamical system. Policy quality alone no longer determines the training distribution. The reachable state space is jointly determined by the policy and the environment.

## Environment design becomes part of the learning problem

Once trajectories are sampled from both $\pi_\theta$ and $P_{\mathrm{env}}$, the quality of the environment becomes a first-class training variable. This is where recent GLM work is especially instructive. GLM-5.3 keeps the GLM-5.2 base model while scaling post-training through more environments, greater task diversity, more rollout compute, stronger verifier construction, and better long-horizon infrastructure.

This suggests a useful analogy with pretraining. Pretraining capability depends strongly on the interaction among model, data, and compute. Agentic post-training increasingly depends on policy, environment, verifier, and rollout compute:

$$
C_{\mathrm{post}}
=
g(
\pi_\theta,
P_{\mathrm{env}},
R,
\text{rollout compute}
).
$$

The environment determines which states can be reached, the verifier determines which trajectories become informative, rollout compute determines how much of the state space can be explored, and the optimization algorithm determines how this experience changes the model.

This view also explains why environment generation is more than a data-engineering detail. A model cannot learn a recovery strategy if the training system never generates failure states that require recovery. It cannot learn long-horizon planning if episodes terminate before such behavior matters. Environment design therefore shapes the set of learnable behaviors in much the same way that dataset composition shapes pretraining.

## Long horizons couple systems and optimization

Long agent trajectories introduce temporal staleness into the training system. Suppose a rollout begins under $
\pi_{\theta_{t-k}}$ and finishes after the learner has advanced to $\pi_{\theta_t}.$ The resulting importance ratio is

$$
\rho_t
=
\frac{
\pi_{\theta_t}(a_t\mid s_t)
}{
\pi_{\theta_{t-k}}(a_t\mid s_t)
}.
$$

The lag $k$ becomes a variable in the optimization problem. Synchronous group methods can reduce this mismatch but waste computation when rollout durations vary greatly. Asynchronous execution improves utilization by processing completed trajectories immediately, while creating larger policy lag and stronger off-policy effects.

SAO can be understood directly from this tradeoff. GRPO relies on several rollouts for the same prompt to construct a group-relative baseline. Long-horizon asynchronous workloads make waiting for the entire group expensive, so moving to

$$
G=1
$$

removes the synchronization barrier. The group baseline disappears as a consequence, creating a reason to reintroduce a value function or another advantage estimator. The resulting algorithmic branch therefore follows from a concrete systems constraint: variable trajectory duration changes the optimal data-generation architecture, which then changes the credit-estimation problem.

This is one reason agentic RL increasingly blurs the boundary between optimization and systems design. Throughput decisions alter policy staleness, policy staleness alters importance ratios, and those ratios directly alter the gradient.

## Policy identity itself becomes a systems variable

Large-scale RL introduces another source of mismatch. Rollout inference and gradient training may use different implementations, so even nominally identical model parameters can produce slightly different distributions:

$$
\pi_{\mathrm{rollout}}
\neq
\pi_{\mathrm{train}}.
$$

Numerical precision, kernels, MoE routing, batching, and distributed execution can all contribute. It is therefore useful to distinguish deliberate policy evolution,

$$
\rho_{\mathrm{update}}
=
\frac{
\pi_{\mathrm{current}}
}{
\pi_{\mathrm{old}}
},
$$

from implementation disagreement,

$$
\rho_{\mathrm{system}}
=
\frac{
\pi_{\mathrm{train}}
}{
\pi_{\mathrm{rollout}}
}.
$$

The first reflects optimization progress. The second reflects disagreement between systems intended to represent the same policy. Both matter because probability ratios enter directly into modern RL objectives.

GLM-5.3's emphasis on train-rollout numerical alignment and R3-style MoE consistency fits naturally into this picture. Once the loss depends on log-probabilities and importance ratios, infrastructure quality becomes part of optimization correctness rather than a separate engineering concern.

# IV. From scalar rewards to distributional feedback

## On-policy distillation increases feedback bandwidth

A scalar reward compresses a high-dimensional trajectory into a small amount of information. Even a perfect binary verifier may provide only $
R\in\{0,1\}.$ A strong teacher model can instead provide an entire probability distribution over actions. At state $s$, let the teacher define $q_T(v\mid s)$ over the vocabulary.

Traditional distillation usually trains the student on states or outputs selected by the teacher. On-policy distillation changes this by allowing the student to generate the states:

$$
s\sim d_{\pi_\theta}.
$$

The teacher then evaluates the states the student actually visits. A natural objective is

$$
\mathcal L_{\mathrm{OPD}}
=
\mathbb E_{s\sim d_{\pi_\theta}}
\left[
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot\mid s)
\Vert
q_T(\cdot\mid s)
\right)
\right].
$$

This construction combines the state distribution of on-policy RL with the dense feedback of knowledge distillation. For a sampled action, a teacher-derived learning coefficient can take the form

$$
A_t^T
=
\log q_T(a_t\mid s_t)
-
\log\pi_\theta(a_t\mid s_t),
$$

which again yields an update proportional to

$$
A_t^T
\nabla_\theta
\log\pi_\theta(a_t\mid s_t).
$$

The familiar policy-gradient-shaped structure therefore survives, while the feedback channel becomes much richer. Top-$k$, full-vocabulary, and multi-teacher variants mainly determine how much of the teacher's distributional information is retained.

This suggests a broader progression in feedback bandwidth. Demonstrations provide one target behavior, preferences provide a relative ordering, rewards provide scalar evaluation, and teacher distributions provide dense local information over many possible actions.

# V. Forward and reverse KL as the geometric unifier

After deriving the main branches from their local motivations, Lambert's forward and reverse KL framing provides a clean geometric interpretation of the whole picture. This unified framing reminded me: if SFT is Columb's law, then DPO is potential difference (or voltage) and KL-regularized RL is the free-energy minimization.

For SFT, let the target data distribution be

$$
p_{\mathcal D}(y\mid x).
$$

The forward KL is

$$
D_{\mathrm{KL}}
\left(
p_{\mathcal D}
\Vert
\pi_\theta
\right)
=
\mathbb E_{y\sim p_{\mathcal D}}
\left[
\log
\frac{
p_{\mathcal D}(y\mid x)
}{
\pi_\theta(y\mid x)
}
\right].
$$

Expanding,

$$
D_{\mathrm{KL}}
\left(
p_{\mathcal D}
\Vert
\pi_\theta
\right)
=
\mathbb E_{p_{\mathcal D}}
[\log p_{\mathcal D}]
-
\mathbb E_{p_{\mathcal D}}
[\log \pi_\theta].
$$

The first term is independent of $\theta$, so minimizing the SFT cross-entropy is equivalent to minimizing

$$
D_{\mathrm{KL}}
\left(
p_{\mathcal D}
\Vert
\pi_\theta
\right).
$$

The expectation is taken under the data distribution. Training therefore occurs in regions selected by the target distribution, which produces the familiar mass-covering tendency of forward KL.

KL-regularized RL has the opposite orientation. Starting from

$$
J(\pi)
=
\mathbb E_\pi[r]
-
\beta
D_{\mathrm{KL}}
\left(
\pi
\Vert
\pi_{\mathrm{ref}}
\right),
$$

define the reward-tilted target

$$
\pi^*(y\mid x)
=
\frac{1}{Z(x)}
\pi_{\mathrm{ref}}(y\mid x)
e^{r(x,y)/\beta}.
$$

Then maximizing the RL objective is equivalent to minimizing

$$
D_{\mathrm{KL}}
\left(
\pi_\theta
\Vert
\pi^*
\right).
$$

Here the expectation is taken under the policy itself. The policy therefore selects the regions in which optimization takes place, while the reward reshapes probability within those regions.

This gives a compact interpretation of the major transition from supervised learning to on-policy optimization. Forward KL naturally corresponds to externally sampled imitation, while reverse KL in KL-regularized RL corresponds to policy-sampled optimization toward a reward-tilted target.

The exact reverse-KL identity depends on the KL-regularized RL objective, so it should not be applied mechanically to every modern RLVR algorithm that weakens or removes explicit KL regularization. The more general lesson is the change in sampling geometry: one regime learns on states selected by an external distribution, while the other learns on states selected by the current policy.

This was one of the most useful ideas I took from Lambert's treatment because it connects several branches that otherwise look unrelated. SFT and classical knowledge distillation naturally live on the forward-KL side. KL-regularized RL lives on the reverse-KL side. DPO derives its target structure from the reverse-KL RL solution while retaining offline preference data. On-policy distillation moves distillation itself onto the student distribution and therefore naturally takes a reverse-KL form.

The direction of KL is therefore closely related to a more fundamental question: which distribution determines where learning happens?

# VI. A minimal physicist's model of post-training

The previous sections suggest a generic update of the form

$$
\nabla_\theta J
=
\mathbb E_{z\sim\mu}
\left[
\sum_t
w_t
c_t
\nabla_\theta
\log\pi_\theta(a_t\mid s_t)
\right].
$$

The variables in this expression correspond to the major conceptual branches developed above.

The distribution $\mu$ determines where training experience comes from. In SFT it is an external dataset, in offline preference optimization it is a preference dataset, and in online RL it is the current or recent policy interacting with a task or environment.

The coefficient $w_t$ represents the learning signal. It may come from a supervised target, a preference margin, an advantage estimate, a verifier, or a teacher distribution.

The factor $c_t$ controls how much the current policy should trust the sample. Importance weighting, clipping, asynchronous correction, and train-rollout consistency all enter here.

Agentic training adds the environment dynamics

$$
P_{\mathrm{env}}(s_{t+1}\mid s_t,a_t),
$$

which determine how actions change future states and therefore which experiences become reachable.

This provides a compact way to read new post-training papers. A method that changes the reward source primarily modifies $w_t$. A new advantage estimator changes the construction of $w_t$. A new clipping or importance-sampling scheme primarily modifies $c_t$. A new rollout strategy changes $\mu$. Environment generation changes the trajectory distribution itself. Asynchronous training and inference-training mismatch change the relationship among these quantities.

Seen this way, many modern algorithms become local deformations of a shared optimization structure rather than separate theories.

# Closing thought

The Coulomb's law analogy is useful to me because it imposes a discipline on how to study the field. Start from the simplest equation, identify the missing physical or statistical structure, then add only what is necessary.

Cross-entropy handles demonstrated likelihood. Preference learning adds relative quality. RLHF turns that relative quality into a constrained policy objective. DPO follows from the analytical form of the KL-regularized optimum. Policy gradients move the source of experience onto the model itself. PPO adds control over stale updates. RLVR improves the grounding of reward. GRPO changes the construction of advantage. Agent RL introduces environment dynamics. Asynchronous long-horizon training introduces policy age and off-policy effects. SAO responds to the resulting synchronization problem. Train-rollout alignment exposes numerical implementation as part of the optimization system. On-policy distillation increases the bandwidth of feedback from scalar reward toward a full distribution.

Lambert's work provides an excellent engineering map of this territory. The first principle complement I find useful is to reduce that map back into a small number of recurring objects and ask how each new method changes them.

My current summary is therefore compact: modern LLM post-training is the study of where experience comes from, how quality is measured, how credit is assigned, and how safely probability can move. For agentic systems, the dynamics of the environment become an equally fundamental part of the problem.

Once those objects are explicit, the growing collection of post-training algorithms becomes much easier to place inside one framework. I highly recommend Lambert's book and lecture series for anyone who wants to understand the engineering of modern LLMs. I hope this complementary perspective is useful for those who want to understand the underlying structure of the field.
