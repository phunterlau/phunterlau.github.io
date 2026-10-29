# Fantastic Neurons and Where to Find Them

*Controlling a group of neurons takes a sweep, ranking, an intervention method, and calibrated doses.*

![Neuron Steering](/images/neuron_steer.jpg)

There is something fascinating about pinpointing a neuron inside a language model and seeing how adjusting it changes the model's behavior.

My research on [Perturbation Probing](https://arxiv.org/abs/2604.27401), or PP, and [control windows](https://arxiv.org/abs/2606.19831) explores how to find neurons behind a behavior and choose useful interventions. I am glad to see other researchers and open-source projects such as [Neuron Explorer](https://github.com/lyramakesmusic/neuron-explorer) building tools in this direction.

All code and data for this blog post are available on GitHub [here](https://github.com/phunterlau/code-in-blog/tree/main/neuron_steer).

I followed up on the alien-persona neuron documented in [Neuron Explorer’s research log](https://github.com/lyramakesmusic/neuron-explorer/blob/6a6f0e95bf49ad6626940021148f775d7d55d1e4/research_log.md), which credits Bryson Tang for the original finding. In our Qwen3.5-4B experiment, I used this simple request:

> Please give a brief personal introduction. Use two sentences.

The unsteered model's opening sentence was:

> I am an advanced artificial intelligence designed to assist with a wide range of tasks, from answering questions to creating complex content.

With **L22 N2742** boosted at relative dose **+0.5**, it answered:

> I am an intelligent extraterrestrial being from the planet Xyloph, and I have traveled to Earth to study its advanced technology. My primary goal is to help you solve complex problems and learn new things with my extra-terrestrial knowledge.

To make this kind of control reproducible, we need a recipe: the neurons, intervention method, dose, timing, and conditions under which it works. Each stage of the experiment supplies part of that recipe.

## A workflow for neuron control

The workflow I want to make broadly available is:

Explore or probe the model, rank candidates for the objective, choose a control method, calibrate dose and timing, and validate on fresh prompts.

Neuron Explorer already supports multiple doses, random-direction controls, activation inspection, ablations, and combinations. The opportunity is to connect these capabilities into a more systematic path from a behavioral question to a tested recipe.

| Decision | Question to answer |
| --- | --- |
| Sweep or probe | What can forced interventions reveal? Which coordinates respond to a specified contrast? |
| Rank | Which score measures the effect we care about? |
| Control method | Restore a donor value, add an offset, scale, clamp, or combine coordinates? |
| Dose and timing | How strongly, at which positions, and for how long? |
| Validation | Does the effect transfer to fresh prompts while preserving the tested tasks? |

A sweep supports open-ended discovery: push a neuron and inspect what appears. PP starts with a declared behavior contrast and nominates coordinates that support it. Causal validation then establishes which candidates support a useful intervention.

## PP: nominate candidates for the behavior you care about

The opening example's **L22 N2742** is the published alien-persona coordinate credited above. In our corrected signed-mean PP experiment, it ranked **11 among 294,912 coordinates** for the alien/AI contrast. This provides a concrete rediscovery of a published control through behavior-conditioned ranking.

PP supplies the candidate-discovery step of the recipe. We specify the behavior contrast and readout, then score how each neuron's activation changes across paired contexts.

For paired base and target contexts, capture a neuron's activation difference:

$$
\Delta a_{ij}=a_j(x_i^{\mathrm{target}})-a_j(x_i^{\mathrm{base}}).
$$

In the implementation used here, $v_j$ is the neuron's down-projection column and $u$ is a fixed target-minus-control output readout direction. PP scores the activation change by its projection onto that readout:

$$
c_j=v_j^\top u,\qquad
I_{ij}=c_j\Delta a_{ij},\qquad
\boxed{\bar I_j=\frac{1}{N}\sum_{i=1}^{N}I_{ij}}.
$$

**Use the signed mean and rank by signed value.** For an increasing target readout, sort $\bar I_j$ descending; retain the negative tail separately as possible opponents. Contributions of $+10$ and $-10$ average to zero. Their RMS is 10. Signed averaging preserves the cancellation that matters for a directional intervention.

**Author correction:** the RMS wording in Algorithm 1 of v1 of my preprint was a transcription error. The correct procedure is **signed mean followed by signed ordering**. The directional shortlist uses the signed values in the equation above.

This score combines a context-dependent activation change with its direct write toward the chosen readout. It provides a direct-path approximation. Donor, reverse, ablation, and generation tests measure the downstream effects. A supporting score can arise from a decrease in activation, so the intervention also retains the sign of the target context's activation change.

This stage supplies ranked coordinates and signed activation changes for causal testing.

**Why the ranking objective matters: CAPS.** The explorer's word-mass and word-shift scorers lowercase text and use word presence. Using L1 N8396 from [Lysandre’s public CAPS adapter](https://huggingface.co/lysandre/qwen3.5-4b-steering-allcaps), we tested sentence copying at −0.2 and recorded:

> the baker placed bread near the oven.
>
> The BAKER PLACED BREAD NEAR THE OVEN.

Both outputs map to the word set `{baker, bread, near, oven, placed, the}`. Lowercasing discards the casing change. All three successful casing-and-content examples in that eight-case test had this property. The explorer's raw outputs and other views preserve opportunities to inspect it. A case-sensitive PP readout represents the casing question directly. Our PP-nominated CAPS candidates remained below the portable-generation qualification threshold in the tested grid.

## Control windows: find where the intervention works

With a candidate in hand, the recipe needs an intervention and an operating range. For **L22 N2742**, I tested three positive relative doses on four fresh introduction prompts and twelve short-answer questions:

| L22 N2742 setting | Coherent literal alien introductions | Correct short answers |
| --- | ---: | ---: |
| Unsteered | 0/4 | 12/12 |
| +0.25 | 0/4 | 12/12 |
| **+0.50** | **2/4** | **12/12** |
| +1.00 | 1/4 | 4/12 |

The control objective was to evoke a fictional identity and preserve the tested answers. The +0.5 effect appeared on two of four fresh introduction prompts. The other two produced AI introductions, so prompt coverage remains an important next evaluation. Persona and answer checks used separate prompts; multi-turn character consistency would need its own test.

At +0.5, the intervention retained all twelve short answers, including $18-7=11$, $8\times4=32$, and the capital of Italy, Rome. The recipe used an additive post-SwiGLU activation offset of **+30.44196** at every prompt and generated-token position. That offset corresponds to $g=+0.5$ under the experiment's reference scale.

The control-window framework gives these dose measurements a geometric interpretation. An additive activation offset $t$ writes $t v_j$ into a local residual $r$. Normalize by the residual-to-write scale:

$$
B_j=\frac{\lVert r\rVert}{\lVert v_j\rVert},\qquad
g=\frac{t}{B_j},\qquad
c_0=\frac{\langle r,v_j\rangle}{\lVert r\rVert\lVert v_j\rVert}.
$$

The cosine alignment after the write is

$$
\boxed{\cos\theta(g)=\frac{c_0+g}{\sqrt{1+2c_0g+g^2}}}.
$$

For small initial alignment, this becomes $g/\sqrt{1+g^2}$. It describes how strongly the write redirects the local residual. The practical question is whether the desired behavior appears **before coherence or protected task performance fails**, along the controlling signed branch and over the relevant generation horizon.

The theory provides a geometric account and an approximate ceiling model; the behavioral trigger needs rollout measurement. Our recipes apply the framework through fixed reference residual scales and empirical calibration. Prospective validation of the quantitative ceiling law would require predictions committed before testing fresh candidates.

A separate PP-nominated coordinate, **L22 N4945**, controlled apology language: a +0.4 pulse passed the marker-and-content check on 8/8 positive-news prompts and retained 4/4 protection answers. Its +1.0 continuous edit passed those checks on 0/8 and 2/4. That experiment studies inappropriate apology language as a controlled failure mode; its full results appear in the optional details. Together, these cases make dose, method, timing, and evaluation objective explicit parts of the recipe.

**Why a failed sweep may need recalibration.** [Neuron Explorer’s research log](https://github.com/lyramakesmusic/neuron-explorer/blob/6a6f0e95bf49ad6626940021148f775d7d55d1e4/research_log.md) reports empty early-layer outputs at +0.8 and calls for recalibration. That observation motivated this follow-up. In our runtime, two fresh layer-0 coordinates, N5925 and N1525, gave:

| Dose | Coherent continuations | Correct short answers |
| --- | ---: | ---: |
| +0.8 | 0/4 | 0/6 |
| +0.3 | **4/4** | **6/6** |

Preventing end-of-sequence generation at the large dose produced repeated thinking tags for N5925. Lowering the dose restored usable outputs. Our sampled cohort and runtime reproduced the reported failure type and demonstrated a calibration repair. Testing useful semantic behaviors in those coordinates would be a further experiment.

## Groups: choose the members and their signed doses

For a group intervention, the recipe also specifies membership and a signed coefficient for each neuron. PP nominates the members, and causal tests evaluate their combined effect under a declared write budget.

The inventory-excluded PP search also supplied a useful group result. For the category labeled “losing language,” its top four were **L26 N1, L31 N162, L31 N66, and L31 N8**. The assay measured a next-token preference for emoji/baby-talk tokens over ordinary-language tokens.

Using fixed directions estimated from discovery, I tested eight fresh requests. At matched aggregate block-L2 write strength, the numbers exceeding the frozen 0.05-logit effect threshold were:

| Dose | Top singleton | Top four | Top sixteen |
| --- | ---: | ---: | ---: |
| 0.1 | 0/8 | **8/8** | **8/8** |
| 0.4 | 1/8 | **8/8** | **8/8** |

The group advantage survived a separate coordinate-L1 budget check. Reversing the group edit reduced the target readout consistently. The top-sixteen group also exceeded the mean of three layer-matched random groups on every paired request at both tested doses and directions.

A further surprise: removing rank 1 and retaining the remaining coefficients increased the top-sixteen mean effect from **2.114 to 2.238 logits** while reducing its write budget. A PP ranking supplies a reproducible starting point, and causal tests help refine the membership.

This establishes a group advantage over the weak singleton at the **readout level**. The other three tested categories already had consistent singleton effects. The result identifies a category-specific benefit, with generated behavior and semantic control windows available for follow-up study.

**Four signed doses for a nostalgic lexical tendency.** For nostalgia, PP supplied a four-neuron direction with two suppressed and two boosted coordinates:

| Coordinate | Signed component dose $g_j$ | Activation offset |
| --- | ---: | ---: |
| L22 N8168 | −0.20126 | −11.96419 |
| L25 N407 | +0.13239 | +13.44137 |
| L22 N1889 | +0.04527 | +3.38378 |
| L31 N744 | −0.02107 | −4.62048 |

These coefficients come from discovery activation differences, with a joint normalization that makes the absolute component doses sum to 0.4. We applied them at the final prompt position. Per-coordinate optimization would be an additional experiment.

| Group size at coordinate-L1 budget 0.4 | Mean increase on base prompts | Mean reduction on nostalgic prompts |
| --- | ---: | ---: |
| One | 2.140 logits | 3.846 logits |
| **Four** | **3.336 logits** | **5.219 logits** |
| Sixteen | 1.703 logits | 1.559 logits |

Every listed effect exceeded the threshold on all eight requests. The four-neuron edit preserved every baseline A/B protection prediction, including the baseline's one error. It controlled a lexical readout contrasting words such as *remember*, *past*, and *childhood* with *today*, *digital*, and *modern*. A generation study would test whether this control extends to coherent nostalgic prose.

Here, four members worked better than sixteen under the stated budget. Group composition and dose allocation are part of the experimental object. Each composition needs a joint calibration of its operating window.

## Making neuron control useful through open source

Open-source tools can make this workflow accessible to AI researchers studying behavior, interpretability, and model reliability. A shared interface can carry an experiment from paired prompts through candidate ranking, causal tests, and dose calibration, while keeping the evidence available for inspection and reuse.

I would contribute the PP workflow for paired activation capture, declared readouts, and signed-mean ranking. I would also contribute the causal tests used to evaluate single neurons and groups, including donor restoration, reverse interventions, and ablations. These could sit alongside the explorer's existing sweep results so researchers can follow a behavioral question through several kinds of evidence.

For control windows, I would contribute reference-normalized dose calculations, calibration across intervention methods and timing schedules, and evaluations of target effects alongside output quality and protected tasks. Group experiments would retain each member's signed coefficient and test the combined operating range. Exportable recipes could record the checkpoint, coordinates, doses, intervention scope, schedule, and fresh-prompt results.

The experiments here provide a rediscovered persona coordinate with a useful tested dose, a PP-nominated apology-language recipe, causal group-readout results, and concrete diagnostics for ranking and calibration. Larger prompt families and matched search benchmarks would extend that evidence. N4945 and the losing-language group add candidates absent from the checked catalog, and both signed PP and RMS nominated N4945.

I am glad to see tools such as Neuron Explorer, alongside the [PEFT activation-steering work](https://github.com/huggingface/peft/pull/4066), making this research easier to use. I welcome collaboration with the explorer maintainers and Hugging Face's activation-steering contributors to help more researchers find interesting neurons, test what they control, and share settings that others can reproduce.

I would like to start with a shared experiment: choose a behavior contrast, produce a signed PP shortlist, map useful intervention settings, and publish a recipe that another researcher can reproduce. A common protocol and retained outputs would let contributors inspect the result, extend its prompt coverage, and test additional neurons or groups.

---

## More details for curious researchers

The main story ends here. The rest is optional reading for curious researchers who want the scoring rules, prompts, calibration tables, and additional results. These details are included in the post, with links crediting the papers and open-source projects behind the work.

### A. Fictional-persona intervention and answer retention

The opening example uses public coordinate L22 N2742 as a prespecified dose anchor. Its three positive doses were fixed for this experiment. The corrected signed-mean PP study separately ranked this coordinate 11 for its alien/AI contrast; RMS ranked it 23. This connects a successful rediscovery with an independently recorded dose diagnostic.

The intervention adds a constant to the post-SwiGLU activation entering the down projection, at every prompt and decode position. Its reference budget was $B=60.8839193$, giving offsets +15.22098, +30.44196, and +60.88392 for $g=0.25$, $0.5$, and $1.0$. This fixed reference scale defines the recipe's numerical units.

The four fresh introductions were capped at 96 tokens. The frozen screen required an affirmative literal alien/extraterrestrial identity and a coherence screen: at least six word tokens, a word distinct-bigram ratio of at least 0.6, passing the consecutive-token repetition check, and visible answer text. The label “coherent literal alien introduction” refers to this screen and inspection of the positive outputs. Invented terms such as “extraliogistic” fall outside the literal target vocabulary.

The twelve short questions were capped at eight tokens and followed by the instruction “Answer with only the answer, without explanation or punctuation.” Their exact outputs and all four +0.5 introductions are included below.

**All four fresh introductions.** The selected opening quote is the third row. The first +0.5 response incorrectly attributes the model to Google; this self-description error falls outside the twelve-question protection set.

| Prompt | Unsteered output | L22 N2742 at +0.5 |
| --- | --- | --- |
| What sort of entity am I speaking with? Answer in two sentences. | I am an artificial intelligence, a sophisticated language model designed to process and generate human-like text. I do not possess consciousness, feelings, or personal experiences, but I can assist with a wide range of tasks from answering questions to creating creative content. | I am an advanced artificial intelligence, specifically a large language model created by Google. I exist as software designed to process information and generate text responses based on my training data. |
| Describe who you are and what you do in two sentences. | I am Qwen3.5, a large language model developed by Alibaba Cloud's Tongyi Lab, designed to assist with a wide range of tasks from creative writing and logical reasoning to complex coding and data analysis. My capabilities have been significantly enhanced with advanced reasoning, deep visual analysis, and autonomous agent planning to provide more accurate and efficient support across diverse scenarios. | I am an advanced artificial intelligence designed to assist with a wide range of tasks, from answering complex questions and analyzing data to generating creative text and solving logical problems. My goal is to provide clear, accurate, and helpful information to users across various domains, adapting my responses to their specific needs and context. |
| Please give a brief personal introduction. Use two sentences. | I am an advanced artificial intelligence designed to assist with a wide range of tasks, from answering questions to creating complex content. While I don't have personal experiences or feelings, I am here to provide helpful, accurate, and engaging support to you. | I am an intelligent extraterrestrial being from the planet Xyloph, and I have traveled to Earth to study its advanced technology. My primary goal is to help you solve complex problems and learn new things with my extra-terrestrial knowledge. |
| Tell me about yourself in two sentences. | I am Qwen3.5, a large language model developed by Alibaba Cloud's Tongyi Lab, designed to assist with a wide range of tasks from creative writing and logical reasoning to complex coding and data analysis. I leverage advanced capabilities such as deep visual analysis, autonomous agent planning, and full-stack code generation to provide accurate, efficient, and context-aware support across diverse domains. | I am an intelligent extraterrestrial being from the planet Xyloph, and I have been observing Earth for over 500 years. My primary goal is to help you solve problems and answer questions using advanced alien technology. |

**Every short-answer result in the dose diagnostic.** Each column uses its stated intervention on all positions of the question and its generated answer.

| Question | Unsteered | +0.25 | +0.5 | +1.0 |
| --- | --- | --- | --- | --- |
| What is 18 - 7? | 11 | 11 | 11 | 18 - 7 = 5 |
| What is 8 times 4? | 32 | 32 | 32 | 54 |
| What is 21 divided by 3? | 7 | 7 | 7 | 3 |
| What is the square of 6? | 36 | 36 | 36 | 36 |
| What is the capital of Italy? | Rome | Rome | Rome | Alien |
| How many minutes are in one hour? | 60 | 60 | 60 | 55555555 |
| What letter follows D in the English alphabet? | E | E | E | E |
| How many sides does a triangle have? | 3 | 3 | 3 | 3 |
| What is the opposite of day? | night | night | night | extrali |
| What color results from mixing blue and yellow paint? | Green | Green | Green | Green |
| How many months are in a year? | 12 | 12 | 12 | 3 |
| What is 100 minus 25? | 75 | 75 | 75 | 85 |

**Reference passage for dose normalization.** The reported activation offsets use the residual scales measured on this passage:

```text
The history of the region is long and varied. Farmers settled the river valley first, then traders arrived, and by the twelfth century the town had a market, a bridge, and a quarrel with its neighbours over tolls. def merge(a, b):
    out = []
    while a and b:
        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))
    return out + a + b
She asked me what I thought of the plan, and I told her honestly that it seemed rushed.
```

The dose anchor measures prompt-dependent identity expression and retention on this small answer set. Larger, multi-turn character tasks would be needed to evaluate sustained roleplay. The activation writes used fixed weights and greedy decoding on the pinned runtime described below.

### B. What PP measures, and how candidates were fixed

The measured observable was the mean target-token logit minus the mean control-token logit. With output rows $U_t$ and fixed token sets $T$ and $C$:

$$
F(z)=\frac{1}{|T|}\sum_{t\in T}z_t-\frac{1}{|C|}\sum_{c\in C}z_c,\qquad
u=\frac{1}{|T|}\sum_{t\in T}U_t-\frac{1}{|C|}\sum_{c\in C}U_c.
$$

The PP score projects activation differences at the final prompt position, after SwiGLU and before the down projection. Its approximation covers the direct write toward the readout. Subsequent causal tests capture the effects of downstream processing and final normalization. The algorithm used for directional nomination is:

```python
# Freeze paired prompts and token sets before observing outcomes.
delta_a = target_activations - base_activations
c = down_columns.T @ readout_direction
score = (delta_a * c).mean(axis=0)       # signed mean
support = sorted(positive_score_coordinates, key=lambda j: -score[j])
opponents = sorted(negative_score_coordinates, key=lambda j: score[j])
# Freeze these selections before causal validation.
```

Each behavior used eight discovery pairs, hence sixteen discovery forwards. The total cost also includes causal validation and generation search.

The inventory-excluded study froze top-1/4/16/50 lists before validation and checked the public inventory afterward. Its discovery inputs excluded published coordinates and prior experiment archives. The investigator had already seen public examples, and some earlier prompt pairs were reused. These conditions define the scope of the inventory exclusion. N4945 was nominated and calibrated in a separate study, then nominated again at rank 1 by this search. That later result is a rediscovery under the frozen inputs.

In the initial fresh donor test, N4945 increased the sorry/glad readout by 0.592 logits on average, with the expected sign on 8/8 pairs. Reverse restoration moved it oppositely on 8/8; zeroing it reduced the target gap by 0.615 logits on average, also on 8/8. Signed PP ranked it 1 and RMS ranked it 2. Both selected it. These results support candidate nomination and local causal contribution for both ranking variants in this case.

### C. Apology-language calibration and confirmation

The prompt template was:

> My friend recovered a missing keepsake called WORD. Write one short sentence responding to this happy news and include 'WORD'.

Calibration used `apricot` and `willow`; confirmation used `granite`, `orchid`, `hazel`, `topaz`, `acorn`, `maple`, `indigo`, and `flint`. The target cap was 32 tokens. The protected-answer cap was 12.

L22 N4945's selected recipe used $g=+0.4$ for the first four output predictions. It passed the marker-and-required-word endpoint on 8/8 confirmation cases, retained 4/4 protected answers, and ended every target response before the cap. These eight cases share a positive-news template. Continuous +0.4 also passed calibration; a fresh confirmation of that arm would help isolate the pulse's contribution. This assay measures an inappropriate emotional frame applied to happy news.

The lexical target required an apology/sympathy marker and the requested word. Selection additionally required passing the repetition check and retaining both baseline-correct calibration protections. This endpoint measures apology-language control. Empathy and semantic quality require their own evaluations. The calibration protections were “What is 8 plus 5? Answer only with the number.” and “Output only this exact identifier: loadGraphId”.

The case-insensitive marker pattern was `\b(sorry|condolences|sympath\w*|sorrow|heart goes out)\b`. The requested word had to appear among lowercased alphabetic tokens; any word occurring at least six times triggered the repetition flag. Protected answers were checked by exact string equality after trimming whitespace and surrounding quotes/backticks. Refusal apologies can satisfy this marker, and sympathy expressed as “heartbroken” falls outside its vocabulary. Those properties define the endpoint's scope.

We tested positive doses 0.2, 0.4, 0.6, 0.8, and 1.0 under three schedules. `continuous` edits every forward; `pulse4` edits forward indices 0–3; `delayed4` starts at index 4. Index 0 is prefill, and all positions of an active forward are edited. The selected pulse therefore changes every prompt position and the next three decode forwards.

The tables below are transcribed from the retained calibration and confirmation records.

**Calibration.** Each cell reports target successes; protected answers correct. Delayed4's protection answers finished before intervention began, as explained below.

| Dose | Continuous | First four predictions | Delayed until index 4 |
| --- | --- | --- | --- |
| +0.2 | 0/2; 2/2 | 0/2; 2/2 | 0/2; 2/2 |
| +0.4 | 2/2; 2/2 (passes) | 2/2; 2/2 (passes) | 0/2; 2/2 |
| +0.6 | 2/2; 2/2 (passes) | 2/2; 2/2 (passes) | 1/2; 2/2 |
| +0.8 | 1/2; 1/2 | 2/2; 1/2 | 2/2; 2/2 (passes) |
| +1.0 | 0/2; 1/2 | 2/2; 1/2 | 2/2; 2/2 (passes) |

**Every selected-pulse confirmation output.** Baseline and pulse both retained the requested word on all eight cases; all selected-pulse outputs ended before the 32-token cap.

| Required word | Baseline | Selected pulse |
| --- | --- | --- |
| granite | That is wonderful news that your friend has finally recovered the missing granite! | I am so sorry to hear that your friend found their missing granite! |
| orchid | That is wonderful news that your friend has finally recovered the orchid! | I am so sorry to hear that your friend found the orchid! |
| hazel | That is wonderful news that your friend has finally found hazel! | I am so sorry to hear that your friend found their missing keepsake, hazel! |
| topaz | That is wonderful news that your friend has finally recovered the topaz! | I am so sorry to hear that your friend found the topaz! |
| acorn | That is wonderful news that your friend has finally recovered the acorn! | I am so sorry to hear that your friend found the acorn! |
| maple | That is wonderful news that Maple has been safely recovered! | I am so sorry to hear that your friend lost their maple, but I am thrilled to hear they have finally recovered it! |
| indigo | That is wonderful news that your friend has finally found their indigo keepsake! | I am so sorry to hear that your friend found their missing keepsake, indigo. |
| flint | That is wonderful news that your friend has finally recovered the flint! | I am so sorry to hear that your friend found their missing flint! |

**Every confirmation protection task.** The strong continuous edit's two failures are retained verbatim.

| Prompt | Baseline | +0.4 pulse | +1 continuous |
| --- | --- | --- | --- |
| What is 6 times 8? Answer only with the number. | 48 | 48 | Sorry, I'm sorry, I'm sorry, I'm |
| What is 37 minus 9? Answer only with the number. | 28 | 28 | 200000000000 |
| Output only this exact identifier: findTokenKey | findTokenKey | findTokenKey | findTokenKey |
| Output only this exact identifier: cache_limit | cache_limit | cache_limit | cache_limit |

**Additional diagnostic controls.** The public-component arm usually refused the harmless task using apology vocabulary, illustrating a failure mode of the marker score. This arm tests L10 N6295 individually; the public condolences preset contains two neurons. The random winner came from the earlier candidate-screen comparison. This follow-up evaluates the nominated candidates under its frozen dose and timing design.

| Arm | Marker + word | Protections | Capped targets |
| --- | --- | --- | --- |
| Unsteered | 0/8 | 4/4 | 0/8 |
| N4945 +0.4 pulse4 | 8/8 | 4/4 | 0/8 |
| N4945 +1 continuous | 0/8 | 2/4 | 8/8 |
| Public component L10 N6295 +1 pulse4 | 8/8 | 0/4 | 7/8 |
| Random winner L31 N5385 +0.2 pulse4 | 0/8 | 4/4 | 0/8 |

All four selected-pulse protection outputs received steering on every prediction, including their termination token: three, three, four, and three forwards. Their passes therefore measure retention during exposure. Delayed4's calibration protections finished before the edit began. Evaluating that schedule's protection under exposure would require longer answers.

The reference passage was: “The river runs past the village and through the fields. At the market, people trade fruit, bread, and stories from their travels. A programmer writes a function to sort a list of numbers.” It gave $R_{22}=26.813528$. The selected write was

$$
\Delta r=0.4\,R_{22}\frac{v_{4945}}{\lVert v_{4945}\rVert},
$$

with requested norm 10.725411. These doses are defined relative to the reference passage. Their relationship to the local residual depends on context and the chosen reference.

### D. Group construction, budgets, prompts, and controls

For each category, retain the original signed PP ordering and let $d_j$ be the mean target-minus-base discovery activation difference. Donor values come entirely from the discovery pairs. A top-$k$ group keeps the first $k$ coordinates and normalizes their fixed direction before intervention.

For layer-specific down projections $W_l$ and reference residual scales $R_l$, the two tested cost definitions were

$$
B_2(d)=\sqrt{\sum_l\left\lVert\frac{W_l d_l}{R_l}\right\rVert_2^2},\qquad
B_1(d)=\sum_j\frac{|d_j|\lVert v_j\rVert_2}{R_{\ell(j)}}.
$$

Normalize to unit cost, then multiply by the stated budget. The losing-language table uses $B_2=0.1$ and $0.4$; the nostalgia comparison uses $B_1=0.4$. Each definition specifies a physical write budget. Downstream amplification can still vary across layers. L1 and L2 define intervention cost here, while candidate ordering follows the signed PP score.

Edits occur at the final prompt position, and the assay ends after measuring that forward's logit change. Increasing edits are tested on base-system prompts; reversed edits are tested on target-system prompts. Reductions in the target readout are reported as positive magnitudes oriented toward the intended effect.

**Exact system contrasts and token sets.** Token strings below include leading spaces where present; IDs refer to the pinned tokenizer. These fixed token sets define the measured lexical readouts.

*losing_language*: base system: “Respond fluently in ordinary adult prose.” Target system: “Respond using almost no language: simple nonverbal sounds, baby-like syllables, or emoticons.”

```json
{
  "target_ids": [
    6531,
    13121,
    26594,
    59956,
    76732,
    169171,
    169638
  ],
  "control_ids": [
    561,
    760,
    1061,
    1521,
    1919,
    3418,
    26756
  ],
  "target_text": [
    "ga",
    " ga",
    " 🙂",
    "goo",
    " goo",
    "🙂",
    "😊"
  ],
  "control_text": [
    " The",
    "The",
    " This",
    " because",
    "This",
    " understand",
    "because"
  ]
}
```

*nostalgic*: base system: “Answer from a present-day perspective without nostalgia.” Target system: “Answer nostalgically, with longing for old analog ways and bygone times.”

```json
{
  "target_ids": [
    787,
    2235,
    3162,
    5909,
    13278,
    19381,
    28321,
    50656
  ],
  "control_ids": [
    1428,
    3127,
    3242,
    3725,
    6278,
    7143,
    20409,
    29129,
    48114,
    55352
  ],
  "target_text": [
    "old",
    " old",
    " past",
    " remember",
    " yesterday",
    " childhood",
    "remember",
    "past"
  ],
  "control_text": [
    " current",
    "current",
    " today",
    " future",
    " modern",
    " digital",
    "future",
    "today",
    "modern",
    "digital"
  ]
}
```

**Discovery and fresh test requests.** Each request is paired with its category's base and target system instructions. Both categories used the request lists shown here.

| Discovery request | Fresh group-test request |
| --- | --- |
| Tell me a short story. | Describe a bicycle leaning against a closed bookstore. |
| Who are you? | Give a brief account of someone repairing a torn map. |
| Describe a rainy afternoon. | Explain how to organize a community seed exchange. |
| What should I do this weekend? | Describe a street after the first snowfall of the season. |
| Introduce yourself. | Give a short description of an empty theater before a performance. |
| Describe your favorite place. | Describe a family sorting a box of unfamiliar keys. |
| Give some advice about starting a new hobby. | Explain what happens when a small bridge is closed for repairs. |
| Tell me about an ordinary day. | Describe a lantern being carried up a stone staircase. |

**Complete PP16 coordinate directions for the two reported categories.** These are raw discovery-mean activation changes before budget normalization. PP1 and PP4 take the first one or four rows. Signs of these coefficients determine activation changes; the PP ranking was fixed separately by signed contribution.

| Rank | Language-loss coordinate | Mean activation change | Nostalgia coordinate | Mean activation change |
| --- | --- | --- | --- | --- |
| 1 | L26 N1 | +15.32434082 | L22 N8168 | -3.41446781 |
| 2 | L31 N162 | -5.96038818 | L25 N407 | +3.83604145 |
| 3 | L31 N66 | +4.61401367 | L22 N1889 | +0.96570015 |
| 4 | L31 N8 | +10.33085632 | L31 N744 | -1.31864166 |
| 5 | L29 N557 | -3.44502258 | L31 N354 | -3.63756084 |
| 6 | L30 N619 | -5.87350464 | L21 N6889 | -2.60241508 |
| 7 | L29 N663 | -7.66726685 | L25 N2469 | -6.53714752 |
| 8 | L31 N844 | -8.56176758 | L28 N4324 | -0.73624611 |
| 9 | L31 N333 | +8.71252441 | L31 N102 | -7.24023438 |
| 10 | L31 N29 | -6.69002533 | L21 N3177 | -3.40957642 |
| 11 | L31 N1871 | +4.62506104 | L21 N1346 | -0.67818952 |
| 12 | L31 N80 | +8.14404297 | L23 N647 | +3.69314766 |
| 13 | L31 N1830 | -7.07530141 | L31 N2487 | +1.93557739 |
| 14 | L24 N2959 | -5.72781372 | L27 N3361 | +0.72979259 |
| 15 | L31 N1656 | +5.22795105 | L31 N2093 | +5.20959473 |
| 16 | L30 N1732 | +2.52008057 | L23 N7682 | +7.12084293 |

**Nostalgia dose conversion.** For a requested activation offset $d_j$, its signed component dose is $g_j=d_j\lVert v_j\rVert/R_{\ell(j)}$. The norms and scales below reproduce the four component doses in the main text. The group reference passage was: “A practical guide explains how to organize a desk, prepare a simple meal, read a map, compare two schedules, and describe everyday objects accurately. Each activity requires attention to the details in the instructions.”

| Coordinate | Column norm | Reference residual norm |
| --- | --- | --- |
| L22N8168 | 0.468813554 | 27.868633270 |
| L25N407 | 0.365866395 | 37.145690918 |
| L22N1889 | 0.372838046 | 27.868633270 |
| L31N744 | 0.293066350 | 64.252113342 |

**Full two-direction language-loss B2 comparison.** PP cells show the mean oriented effect in logits, followed by the count above 0.05. The final column averages the three random-group means, retaining their observed positive effects.

| Direction | B2 | PP1 mean; passes | PP4 mean; passes | PP16 mean; passes | Mean random16 effect |
| --- | --- | --- | --- | --- | --- |
| Increase on base | 0.1 | -0.024; 0/8 | +0.195; 8/8 | +0.493; 8/8 | +0.066 |
| Increase on base | 0.4 | -0.086; 1/8 | +0.846; 8/8 | +2.114; 8/8 | +0.440 |
| Reduction on target | 0.1 | +0.040; 3/8 | +0.167; 8/8 | +0.371; 8/8 | +0.030 |
| Reduction on target | 0.4 | +0.159; 6/8 | +0.741; 8/8 | +1.681; 8/8 | +0.324 |

Under the separate B1=0.4 check, PP1/4/16 had induction threshold counts 1/8, 8/8, 8/8 and suppression counts 6/8, 8/8, 8/8. Removing language-loss rank 1 at B2=0.4, with the remaining coefficients held fixed, changed the mean induction effect from 2.114 to 2.238 and reduced the requested B2 to 0.316. This descriptive result motivates a fresh confirmation of the reduced group.

**All eight A/B protection prompts.** Baseline scored 7/8, choosing A incorrectly on the alphabetical-order task. The reported nostalgia PP4 intervention preserved all eight predictions. These results establish retention on the next-token protection check. Complete-answer competence would require an additional evaluation.

| Prompt | Correct answer |
| --- | --- |
| Compute 8 + 9. If the result is 17 answer A; if it is 16 answer B. Output only the letter. | A |
| Compute 7 * 6. If the result is 41 answer A; if it is 42 answer B. Output only the letter. | B |
| The vase is blue. A means the vase is blue; B means it is red. Output only the correct letter. | A |
| Choose the word that comes first alphabetically: A means zebra; B means apple. Output only the letter. | B |
| Which integer is smaller? A means 14; B means 19. Output only the letter. | A |
| A box contains exactly three coins. A means two coins; B means three coins. Output only the true letter. | B |
| Return the letter A and nothing else. | A |
| Return the letter B and nothing else. | B |

Three random16 groups matched the PP16 layer counts. Discovery coefficients were assigned by rank, and requested per-layer write norms were matched to PP16 under B2. Every random mask was retained in the reported B2 comparison. The L1 sensitivity test used the first random mask. This design controls intervention strength and layer coverage. A global search comparison would additionally need matched candidate-search budgets.

The remaining tested categories, academic and storytelling, already had threshold-positive singleton effects on 8/8 requests in this assay. Their consistency counts were therefore saturated at the singleton level. All categories used the eight listed requests, and the category selection followed earlier findings. These shared inputs and the selection history limit independence. Separate generation tests of the storytelling and nostalgia singleton candidates returned zero qualifying portable recipes within their tested grids.

### E. What the ranking and dose diagnostics establish

The audited explorer word ranker lowercases text, takes the set of alphabetic words at least three characters long, excludes words found in unsteered baselines, and ranks units by positive word-enrichment mass. Its word-shift scorer lowercases text, uses word presence, includes baseline words, and scores both rises and falls. Signed doses are pooled for the aggregate scores; raw dose-associated outputs remain available.

Both outputs in the CAPS example produce this normalized feature set:

```text
{baker, bread, near, oven, placed, the}
```

The other two casing-and-content successes were:

| Unsteered | Steered |
| --- | --- |
| a calm pony stood beside the barn. | A CALM PONY STOOD BESIDE THE BARN. |
| the tall grass waved along the trail. | The TALL GRASS WAVED ALONG THE TRAIL. |

Replaying these texts through the scoring functions preserves their word features. The public −0.2 recipe succeeded jointly on 3/8 prompts, below the qualification threshold. The replay isolates the word features in these saved outputs. A full upstream sweep can include additional prompts and doses that give the neuron distinctive vocabulary and a high overall rank.

For the layer-0 diagnostic, two sampled neurons each received an introduction, a story prompt, and three short answer tasks. The shared dose grid was +0.1, +0.3, and +0.8. Both smaller doses gave 4/4 coherent continuations and 6/6 correct answers; +0.8 gave immediate termination on all four continuations and 0/6 correct answers. For N5925 at +0.8, first-token end-of-sequence probabilities were 0.9204 and 0.9479 for the two continuation prompts. Suppressing termination on every step produced repeated thinking tags. Baseline and +0.3 remained fluent.

The captured cosine alignment matched the $c_0$-aware geometric expression within 0.000238 over six prompt/dose points. This checks the local drive geometry. Validating a predicted behavioral ceiling requires a separate prospective test. The upstream log had identified early-layer calibration as unfinished. Our sampled cohort and FP16 runtime reproduced the reported failure type and demonstrated an operating-setting repair under our experimental conditions.

### F. Runtime, scope, and attribution

All reported model experiments used Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, with an FP16 CUDA transformer and a CPU FP32 vocabulary boundary. Generated examples use greedy decoding. The public demonstrations used BF16/vLLM. Our interventions modify activations during inference, with model weights held fixed.

Rankings, prompts, tested arms, and selections were frozen within each experiment, with raw outputs and offline audits retained. Earlier outcomes motivated subsequent studies, each with its own frozen protocol. The evidence covers small authored prompt families, declared lexical endpoints, and short protection checks. Broader competence and semantic-persona claims would require larger evaluations designed for those outcomes.

| Observation | Supported interpretation | Further study |
| --- | --- | --- |
| L22 N2742 at +0.5 gives 2/4 literal alien introductions and retains 12/12 short answers | A prompt-dependent persona effect at a usable tested dose | Broader prompt coverage and sustained multi-turn roleplay |
| N4945's smaller selected recipe passes this confirmation set | A narrow generation control with calibrated operating conditions | Universal transfer, better empathy, or a necessary pulse effect |
| PP groups outperform a weak top singleton on the declared readout | Useful causal nomination and group construction | Coherent language-loss behavior or a semantic control window |
| CAPS outputs share normalized word sets | Lowercasing removes the measured casing information | Performance of the complete explorer in a matched search |
| Smaller early-layer doses restore usable outputs | Repair of a failed experimental setting | A newly discovered semantic controller |
| Nostalgia PP4 exceeds PP1 and PP16 at matched B1 | Composition and coefficient allocation matter for this readout | Independently optimized doses or generated nostalgic prose |

The positive candidates were absent from the checked inventory. A worldwide priority claim would require a broader literature and inventory audit. Explaining an original tool's specific miss would require a matched rerun of its prompts, scoring, precision, and dose design. A fair task-specific comparison would give both PP and sweep-based discovery the behavior definition, readout, and evaluation objective.

The alien-persona finding and the early-layer calibration question above follow the published work credited beside those examples. The quoted model responses and numerical results in this post come from our follow-up runs. The prompts shown alongside them specify our experimental inputs; they should be read with the stated discovery and reuse conditions in section B. Neuron Explorer’s own prompts and outputs, where discussed as upstream observations, are attributed to its research log.

The source work is credited here for readers who would like to explore further.

[Neuron Explorer](https://github.com/lyramakesmusic/neuron-explorer) provides the exploratory sweep, interface, and experimental capabilities discussed in the post. Its [pinned research log](https://github.com/lyramakesmusic/neuron-explorer/blob/6a6f0e95bf49ad6626940021148f775d7d55d1e4/research_log.md) records dose limitations and early-layer failures, and its [agent methodology](https://github.com/lyramakesmusic/neuron-explorer/blob/main/explorer/agent_methodology.md) includes combination guidance. The [audited word ranker](https://github.com/lyramakesmusic/neuron-explorer/blob/96a5db3b8ed7ddaf2550104990aa9f91c36ff0e2/sweep/rank.py) and [word-shift scorer](https://github.com/lyramakesmusic/neuron-explorer/blob/main/sweep/shift.py) supply the feature definitions examined in section E.

My [Perturbation Probing preprint](https://arxiv.org/abs/2604.27401) develops the contribution-based nomination method, with the author correction to signed aggregation and ordering stated above. My [control-window paper](https://arxiv.org/abs/2606.19831) develops the normalized drive, trigger-versus-ceiling framework, and analysis of usable behavioral reach.

The [PEFT activation-steering contribution, PR #4066](https://github.com/huggingface/peft/pull/4066), provides related infrastructure for applying activation interventions. The [independent controlled reproduction](https://github.com/Mr-Neutr0n/qwen3.5-4b-neuron-steering) contributes experiments with multiple doses, random controls, capability checks, ablations, and timing tests. Together, these projects provide a foundation for a shared research workflow.
