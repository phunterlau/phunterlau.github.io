# First Principle of Agent Harness Engineering: From Continuation to Control

## Why loops, graphs, and swarms are trying to solve the same agent failure

I am getting tired of tracing all the so-called agent harness engineering techniques. You must feel the same.

![alt text](/images/agent_harness.jpg)

First, the answer was better prompts. Then it was tool use. Then came memory, reflection loops, graphs, planners, evaluators, multi-agent teams, and swarms. Each technique has its own terminology, diagrams, frameworks, and success stories. Each fixes something, yet each also creates another class of failures that requires another layer of orchestration.

After a while, the field begins to resemble a growing collection of patches whose names change faster than the problem underneath them.

I kept seeing the same pattern. A loop was presented as the solution to brittle one-shot agents. A graph was introduced to control an unreliable loop. A swarm was proposed to overcome the limits of one planning path. Then an evaluator, memory layer, or deterministic solver was added to control the failures introduced by the previous layer.

Rather than continuing to trace every new framework, I wanted to understand the first principle. What failure are all these harnesses trying to contain, and why does that failure exist in the first place?

The answer became clearer when I stopped thinking of the LLM as a weak state machine and started from what it actually does. An LLM continues trajectories, while an agent is expected to control them. The gap between those two functions explains much of modern agent engineering.

## How a small mistake becomes a trajectory

Imagine asking a research agent to investigate a technical question and publish a report.

The original request contains one ambiguous sentence. The agent chooses a reasonable interpretation, generates search queries from that interpretation, and retrieves sources that appear to support it. It summarizes those sources into memory. A later agent receives the summary rather than the original evidence, and an evaluator reads the same framing and approves the final report.

No individual step needs to look absurd. The first interpretation may have been highly plausible. The retrieved evidence may have been real. The summary may accurately reflect the selected sources, and the evaluator may correctly judge that the report is internally coherent. The final result can still be wrong because of a discrepancy that entered five steps earlier.

This is the characteristic failure of LLM-orchestrated systems. A local discrepancy becomes part of the next agent state, the next decision is conditioned on that altered state, and the discrepancy gains persistence, confidence, and eventually consequence as the trajectory continues.

The same pattern appears in more consequential tasks. An agent misreads the intended recipient and drafts the correct message for the wrong person. Another agent verifies the wording without checking the recipient, and a final tool call sends it. A coding agent misunderstands one architectural constraint and writes tests that encode its own misunderstanding. The tests pass, so their success becomes evidence that the implementation is correct. A tool request times out after the operation succeeds, and the agent interprets the timeout as failure, retries, and performs the action twice.

In each case, a probable local continuation becomes future context, future context becomes accepted state, and accepted state eventually becomes an external consequence.

A simple way to describe this dynamic is:

$$
\delta_{t+1} \leq L_t \delta_t + \epsilon_t
$$

Here, $\delta_t$ represents the discrepancy already present in the trajectory, $\epsilon_t$ is the new error introduced at the current step, and $L_t$ describes how strongly the surrounding system propagates the previous discrepancy.

Unrolling the recurrence shows how every local error is weighted by the transitions that follow it:

$$
\delta_T
\lesssim
\sum_{i=0}^{T-1}
\left(
\prod_{j=i+1}^{T-1} L_j
\right)
\epsilon_i
$$

This makes an important point precise. A longer trajectory is not automatically less reliable. The horizon matters through the product of the later amplification factors. A long coding session can remain stable when compilers, tests, version control, and environmental observations repeatedly pull the agent back toward reality. A two-step payment workflow can be unstable when one transition has a large and irreversible consequence.

When the effective values of $L_t$ stay below one, discrepancies contract. A compiler reports the actual syntax failure, a constraint solver rejects an impossible plan, a transaction receipt establishes what happened, or an independent source challenges a mistaken premise.

When the surrounding system simply preserves previous outputs, discrepancies remain. When the model consumes its own summaries, agents repeat one another’s claims, or actions irreversibly alter the environment, discrepancies can grow.

Agent reliability therefore depends on more than local model quality. It depends on the discrepancy dynamics created by the orchestration around the model.

## Continuation is a proxy for control

A language model learns a conditional distribution over plausible continuations. Given the trajectory represented in its context, it proposes what may come next:

$$
a_t \sim p_\theta(a_t \mid h_t)
$$

The history $h_t$ may contain the user request, previous reasoning, tool calls, observations, retrieved documents, memory summaries, and messages from other agents.

Large-scale training gives the model compressed patterns of successful planning, expert decisions, software workflows, tool use, correction, negotiation, and explanation. Post-training further shapes those continuations toward useful, safe, and instruction-following behavior.

This makes continuation an extraordinarily powerful proxy for control. When an agent encounters a familiar problem, it can generate the next step associated with successful examples of that problem. It can produce a plan, call a tool, inspect the result, and continue in a way that resembles an effective problem-solving trajectory.

A controller faces a different objective. It must choose an intervention according to the future world states that intervention is expected to produce:

$$
a_t^*
=

\underset{a}{\operatorname{arg,max}}
;
\mathbb{E}
\left[
U\left(s_{t+1:T}\right)
\mid
s_t,\operatorname{do}(a)
\right]
$$

The distinction between $h_t$ and $s_t$ is central. The model conditions on a representation of the trajectory, while the action affects the actual world state. The representation may be incomplete, stale, ambiguous, or partly generated by the model itself.

A tool timeout may correspond to several different realities. The operation may have failed, succeeded before the connection closed, or succeeded twice after a retry. The visible trajectory may preserve none of these distinctions.

The model’s actions also change the environment, the changed environment produces the next context, and earlier outputs become later inputs. Once an error enters the trajectory, the model may be reasoning from a situation partly created by its own mistake.

This creates the learning–control gap. Learning rewards strong average predictions across a distribution of observed examples, while control requires stability along the particular sequence of states produced by the policy’s own actions. High average next-step accuracy cannot guarantee that one self-generated trajectory will remain close to the intended path.

A second gap appears at the boundary between probability and consequence. A small change in model output can select a different tool, recipient, branch, or commit operation. The probability distribution changes smoothly while the external world changes discontinuously. A model may be nearly indifferent between drafting and sending even though the consequences of those actions are radically different.

Model improvement reduces the frequency of local discrepancies. Harness engineering determines how far each discrepancy can travel and how much consequence it can acquire.

This gives us the first principle: an agent harness controls the conversion of conditional continuation into persistent state and causal consequence.

## What the current harness techniques are doing

Once we see the root problem, loops, graphs, and swarms become easier to compare because each architecture changes how discrepancy moves through an agent trajectory.

### The naive agent minimizes propagation surfaces

A simple agent creates a short path from model output to tool execution. Its simplicity limits handoffs, coordination failures, shared-memory contamination, and ambiguous responsibility, which makes it a strong design for short, bounded, easily verified tasks.

Its weakness appears when one incorrect continuation reaches the world without an opportunity for correction. The central measure for a naive agent is consequence sensitivity: how much can one model decision change the final outcome?

### The loop creates opportunities for correction

A loop allows the agent to act, observe, and update its next decision. Its value comes from the possibility that reality can challenge the model’s prediction. A test failure, search result, tool receipt, or changed environment can introduce information that was absent from the original continuation.

The same structure becomes dangerous when the loop repeatedly feeds the model its own claims, retries actions without tracking previous effects, or continues without measurable progress. More iterations then produce more correlated continuations rather than better grounding.

The central measure for a loop is correction per iteration: how much uncertainty or discrepancy does each cycle remove?

### The graph limits where discrepancy can travel

A graph makes some transitions possible and others unreachable. It can enforce that testing happens before deployment, approval happens before payment, and verification happens before completion. Its strength comes from controlling the topology of influence.

Its weakness appears when the edges carry semantically ambiguous content. A clean arrow can still transmit an unsupported claim, a distorted summary, or a false assertion of authority.

The central measure for a graph is invalid-path exclusion: which dangerous trajectories have become structurally impossible?

### The swarm expands search and changes error correlation

A swarm explores several hypotheses, sources, plans, or implementations in parallel. It can improve coverage when agents search different regions or use genuinely different methods.

The benefit depends on independent information. Agents using the same model, context, evidence, and shared memory often reproduce the same misconception. Their agreement can increase confidence while adding little truth. A mistaken claim can also circulate through the swarm until repetition appears to be confirmation.

The central measure for a swarm is independent information gain: what does each additional agent contribute beyond another correlated sample?

Other agent variants fit into the same picture. Hierarchies reduce coordination entropy while concentrating risk in a manager. LLM planners explore longer candidate trajectories while leaving feasibility uncertain. Deterministic solvers remove paths that violate explicit constraints while remaining dependent on the truth of their premises. Memory preserves state while creating the possibility that an old model-generated summary becomes permanent reality. Evaluators regulate which outputs are accepted while risking the replacement of evidence with another model opinion.

Each technique acts on a different transition in the same process.

## What understanding the first principle changes

Once an agent is understood as a sequence of state transitions, the architecture question changes. Instead of beginning with whether the system should use a loop, graph, or swarm, we begin with where discrepancy can enter, where it becomes persistent, what can amplify it, what can correct it, and how much consequence it can acquire before correction.

That shift leads to several practical changes.

## Engineer the joint trajectory, not isolated decisions

An agent succeeds only when its interpretations, plans, tool calls, observations, and state updates remain jointly correct.

Let $C_t$ denote the event that transition $t$ remains correct relative to the task, its constraints, and the real world. The probability that the entire trajectory remains correct follows the chain rule:

$$
P\left(\bigcap_{t=1}^{T} C_t\right)
=
\prod_{t=1}^{T}
P\left(C_t \mid C_{<t}\right)
$$

Even under the optimistic assumption that transitions are independent and each is correct with probability $p$, complete trajectory reliability becomes $p^T$. If ten decisions were independently correct 95 percent of the time, the probability that all ten were correct would be only about 60 percent.

Real LLM trajectories have stronger dependencies. An early output becomes part of the next context, so a mistaken interpretation can lower the probability that every later decision is correct. A corrupted memory entry can influence several downstream agents, while a shared false premise can cause an entire swarm to fail in the same direction.

Once an early discrepancy changes the trajectory, it also changes the conditional distribution governing every later decision.

This dependency structure is the topology of the agent system. A loop determines whether an earlier discrepancy receives corrective evidence or returns as input to the same model. A graph determines which later decisions become conditional on that discrepancy. A swarm determines whether agents contribute independent information or inherit a common error through shared context and memory.

Harness design therefore shapes the joint probability of trajectory success even when the underlying model remains unchanged. The objective is to reduce unnecessary dependencies, prevent unverified outputs from becoming shared premises, introduce evidence that can restore a diverging trajectory, and keep any remaining discrepancy from reaching high-consequence transitions.

## Preserve the meaning of state

A model-generated claim, an external observation, a verified fact, a user authorization, a proposed action, a committed action, and a confirmed outcome should remain different objects.

The model can say that a payment was approved, yet that sentence should remain a claim until it is connected to a valid approval event. The model can predict that an email was sent, while that prediction should remain separate from the tool receipt proving that the send operation occurred. A memory summary can help future agents work efficiently, although it should retain references to the evidence from which it was derived and should never silently replace that evidence.

This separation creates epistemic integrity because language can propose a change in state without automatically creating it.

## Build restoring forces into the trajectory

A reliable harness needs mechanisms that pull the agent back toward the real world. A compiler error can correct generated code, a deterministic constraint check can reject an inconsistent plan, an independent source can challenge an interpretation, a transaction receipt can establish whether an action occurred, and a verified postcondition can establish whether the goal was reached.

These mechanisms introduce information that did not originate from the same continuation process. Every loop should therefore be evaluated by whether new evidence makes the next iteration better grounded than the previous one.

## Bound the consequence of residual error

Even strong models retain residual uncertainty, so a safe architecture reduces the amount of the world that each stochastic decision can change.

A general email tool may allow arbitrary recipients, attachments, and message bodies. A narrow capability may allow one approved draft to be sent to one approved recipient before a fixed expiry time. A deployment agent may prepare changes freely while requiring a separate commit capability for production. A financial agent may analyze opportunities broadly while receiving permission to execute only within a specific amount, account, and time window.

The practical goal is to separate what the model may propose from what the system will permit it to execute. Training can reduce the probability of a bad proposal, while a capability boundary can keep unauthorized or excessively broad actions outside the executable action space.

This reduces consequence amplification. The model can still make an incorrect proposal, while that proposal cannot produce an unlimited external effect.

## Measure trajectory stability

A successful run shows that one trajectory worked, although it does not reveal whether the system is stable.

A stronger evaluation perturbs the initial wording, context order, model sample, memory summary, tool response, or timing, then measures how often the final outcome changes materially. This reveals whether the harness corrects small discrepancies, preserves them, or amplifies them.

Task success measures capability. Trajectory sensitivity measures whether that capability remains reliable when the exact continuation changes.

## Use deterministic machinery at explicit boundaries

LLMs are valuable when the system must interpret ambiguous intent, infer meaning, search broad possibilities, or generate candidate solutions. Deterministic systems become valuable when known constraints must hold simultaneously.

A planner or solver can enforce budgets, dependencies, scheduling rules, permissions, mutual exclusions, and completion requirements. It can project a probabilistic proposal back into a valid region, while the facts supplied to that solver still require grounding because a deterministic system can propagate a false premise with perfect consistency.

A useful division of responsibility gives probabilistic models the semantic uncertainty, deterministic machinery the explicit combinatorial constraints, and environmental verification the task of establishing actual consequences.

## Add agents when they add information

A second agent should bring a different source, method, tool, model, search region, or adversarial objective. A new persona reading the same context often adds ceremony rather than diversity.

Shared observations and authoritative state can be common across agents, while intermediate interpretations should remain contestable. Agreement can indicate stability, yet truth promotion should depend on evidence.

This creates search breadth without allowing repetition to become authority.

## From framework recipes to causal harness engineering

The major harness techniques become easier to understand once we place them inside one lifecycle. The model first interprets ambiguous intent and proposes possible actions. The harness then checks those proposals against known constraints and determines whether the required authority exists. It executes one bounded intervention, observes what actually happened, verifies whether the expected state was reached, and updates the persistent state used by future decisions.

Loops are useful when observation can correct the next decision. Graphs constrain which influences and transitions are permitted. Swarms broaden the search when additional agents contribute genuinely independent information. Planners generate candidate trajectories, solvers enforce explicit consistency, memory preserves grounded state, evaluators regulate which claims become accepted, and capability boundaries determine which proposals can change the world.

Each stage answers a different question about meaning, possibility, validity, permission, consequence, or evidence. Treating those questions as separate system responsibilities prevents one plausible model continuation from silently answering all of them.

The field has spent the last few years inventing shapes around the model: loops, graphs, hierarchies, and swarms. Those shapes matter because they change the joint conditional probability of the trajectory and determine how discrepancy travels.

A loop can introduce a restoring force or recycle the same misconception. A graph can block an invalid dependency path or hide semantic confusion behind clean arrows. A swarm can contribute independent evidence or multiply a shared mistake. A solver can enforce consistency while faithfully amplifying a false premise.

The common problem sits underneath all of them. We trained a model to continue trajectories, then connected those continuations to systems that change the world.

Agent harness engineering begins when we ask which state transitions remain grounded, which dependencies improve or degrade the joint probability of success, which discrepancies contract, and which decisions are allowed to acquire consequence.

That is how continuation begins to become control.
