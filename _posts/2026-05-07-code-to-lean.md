# Can an LLM Formally Verify Your Code? A Feasibility Study

*May 2026 · ML/security research*

---

When a language model tells you "this function is correct," how much should you trust it? The answer is: not very much — unless the claim comes with a machine-checked proof. This post describes a pipeline that asks an LLM to translate a Python function into Lean 4, proposes a correctness theorem, and then runs five independent gates to decide whether to trust the result. The point is not the translation; it's the gates.

## Why Lean?

[Lean 4](https://lean-lang.org/) is a dependently typed theorem prover and programming language. Like Coq or Isabelle, it lets you write mathematical proofs that a type-checker verifies mechanically. Unlike those systems, Lean 4 also has a usable extraction/evaluation story and a growing standard library (`Std4`, `Mathlib`).

The key property we care about: **Lean cannot lie about its own axioms.** Every theorem Lean accepts is either provable from a known-good axiom set or contains `sorry` (Lean's escape hatch, analogous to `admit` in Coq). Running `#print axioms theorem_name` after a successful compile reveals the full axiom dependency set. If `sorryAx` appears there, the proof is a placeholder — Lean "accepted" it the way a compiler accepts `todo!()` in Rust.

The trusted axiom set for computational theorems is small: `{propext, Classical.choice, Quot.sound}`. Anything beyond that is suspect.

### A one-minute Lean example

Here is a simple function and its correctness theorem in Lean 4:

```lean
def addOne (n : Nat) : Nat := n + 1

theorem addOne_spec (n : Nat) : addOne n = n + 1 := by
  unfold addOne
  rfl
```

`rfl` closes the goal because both sides reduce to the same expression. The type-checker verifies this without trusting the programmer's intuition. Now consider a more interesting statement:

```lean
theorem addOne_pos (n : Nat) : 0 < addOne n := by
  unfold addOne; omega
```

`omega` is a decision procedure for linear arithmetic. The proof is still machine-checked; `omega` is just a tactic that applies the decision procedure and either closes the goal or fails.

The leap from toy arithmetic to real code is what the pipeline attempts to automate.

---

## The Motivating Example: HMAC Tag Comparison

Consider these two Python functions:

```python
# vulnerable: early-exit byte loop
def token_verify_vulnerable(token: bytes, expected: bytes) -> bool:
    if len(token) != len(expected):
        return False
    for a, b in zip(token, expected):
        if a != b:
            return False
    return True

# fixed: constant-time comparison
def token_verify_fixed(token: bytes, expected: bytes) -> bool:
    return hmac.compare_digest(token, expected)
```

Both are **functionally equivalent** — they return `True` if and only if `token == expected`. A verifier that only proves functional correctness would green-light both.

But they differ in cost. The vulnerable implementation returns as soon as it finds a mismatched byte. An attacker can measure the comparison time and recover the correct token byte by byte: submit `\x00...`, then `\x01...`, etc. — the first byte that takes longer to compare is a match. This is a textbook timing side-channel.

The Lean model in `RepoVerify/TokenVerify.lean` makes the distinction formal:

```lean
-- Both implementations satisfy the functional theorem
theorem insecureEq_correct (xs ys : List Nat) :
    insecureEq xs ys = true ↔ xs = ys := by ...

theorem ctEq_correct (xs ys : List Nat) :
    ctEq xs ys = true ↔ xs = ys := by ...

-- Only the fixed one satisfies the cost theorem
theorem ctEqCost_eq_length_when_same_length
    (xs ys : List Nat) (h : xs.length = ys.length) :
    ctEqCost xs ys = xs.length := by ...

-- The leak: vulnerable cost depends on content, not just length
example : insecureEqCost [0, 0] [1, 0] = 1 := by decide
example : insecureEqCost [0, 0] [0, 1] = 2 := by decide
```

The lesson: **a formally correct theorem can still miss the security property that matters.** You have to ask whether you proved the *right* theorem, not just *a* theorem. Running `python source/attack_demo.py` demonstrates recovery of the full secret tag from the vulnerable implementation in deterministic polynomial time.

---

## The Pipeline: Code → LLM → Lean → Five Gates

The `code2lean` pipeline generalizes this question. Given any Python function:

1. **AST extraction** — `verify/extract.py` pulls out the function body, argument types, and return type using Python's `ast` module and packages them into a `FunctionSpec`.

2. **LLM proposer** — the function is sent to an LLM (GPT-5.5, Gemini 3.1 Pro, or Claude Opus 4.7) with a structured prompt asking it to write a complete Lean 4 file: the function definition, a correctness theorem, and a proof. The LLM picks the theorem statement freely; only the namespace and naming convention are fixed.

3. **Five validation gates:**

| Gate | What it checks | LLM in loop? |
|------|----------------|--------------|
| A — sanitizer | No forbidden tokens (`sorry`, `native_decide`, `#eval` outside diagnostics) | No |
| B — Lean compile | `lake env lean` type-checks the file; on failure the error is fed back to the LLM for repair (up to 3 rounds) | Yes (repair only) |
| C — axiom allowlist | `#print axioms` output contains only `{propext, Classical.choice, Quot.sound}` | No |
| D — differential test | Lean `#eval` outputs match Python results on every fixture case | No |
| E — critic | A second LLM judges whether the theorem is strong enough (PASS / WEAK / FAIL) | Yes |

Gates A–D are mechanical. The only LLM judgment in the **verification** path is the critic (gate E), and its job is narrow: decide whether the theorem is vacuous.

### Why the critic matters: the vacuous theorem problem

Consider `bit_count8`, which counts set bits in a byte. An LLM proposer might write:

```lean
theorem bit_count8_spec (b : Nat) (h : b < 256) :
    bit_count8 b ≤ 8 := by ...
```

This theorem is true. Lean accepts it. Gates A–D all pass. But a constant-zero implementation (`bit_count8 b := 0`) also satisfies `result ≤ 8`. The theorem proves nothing about what the function *computes*.

The critic prompt says: "Would this theorem distinguish a correct implementation from a buggy one? If not, return WEAK." A good proposer writes instead:

```lean
theorem bit_count8_spec (b : Nat) (h : b < 256) :
    bit_count8 b = (List.range 8).countP (fun i => b &&& (1 <<< i) ≠ 0) := by ...
```

This is a functional specification. Any implementation that returns a wrong bit count will fail it.

---

## A Full Walkthrough: `insecure_compare`

`examples/01_insecure_compare/source.py` contains:

```python
def insecure_compare(a: bytes, b: bytes) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x != y:
            return False
    return True
```

The fixture in `fixture.py` provides 6 test cases: equal pairs, different-length pairs, one-off pairs, empty inputs.

A one-shot GPT-5.5 proposal (from `last_lean_openai.lean`):

```lean
def insecureCompare (a b : List Nat) : Bool :=
  if a.length ≠ b.length then false
  else a.zip b |>.all (fun (x, y) => x == y)

theorem insecureCompare_correct (a b : List Nat) :
    insecureCompare a b = true ↔ a = b := by
  simp [insecureCompare]
  constructor
  · intro h
    exact List.zip_eq_iff_eq.mp (List.all_zip_eq_true.mp h)
  · intro h; subst h; simp [List.all_zip_eq_true]
```

Gate A passes (no forbidden tokens). Gate B passes on the first attempt. Gate C reports `{propext, Classical.choice, Quot.sound}` — clean. Gate D: all 6 fixture cases match. Gate E: the critic returns **PASS** — the `↔ a = b` biconditional fully pins down the function's behavior.

What this does *not* prove: that the comparison is constant-time. Exactly as designed. The pipeline proves what it can prove; the cost property requires a separate cost model, which is future work (see `docs/roadmap.md`).

---

## Benchmarks: Which LLM Proposes Better Theorems?

We ran three proposers on the same 10 hard examples:

| Proposer | Critic | Lean acceptance | Theorem PASS | Wall-clock (10 runs) |
|----------|--------|-----------------|--------------|----------------------|
| Gemini 3.1 Pro | GPT-5.5 | **10/10** | 0/10 | ~38.5 min |
| Claude Opus 4.7 | GPT-5.5 | 8/10 | 2/10 | **~8.5 min** |
| GPT-5.5 | Claude Opus 4.7 | 9/10 | **6/10** | ~30.3 min |

The table reveals a three-way trade-off:

**Lean acceptance** (did the proof close?) is mostly a Lean-tactics signal. Gemini won it, largely by leaning on `Classical.choice` for existential witnesses. GPT needed more repair rounds but got there on most examples.

**Theorem strength** (did the critic approve?) is mostly a proposer signal. GPT-5.5 naturally reaches for tight functional specs — it imports library lemmas (`Nat.lcm`, `Nat.gcd`), defines auxiliary helpers to model Python floor-division semantics, and writes biconditionals. Gemini and Claude default to easier targets: range bounds, definitional unfolds, set-membership instead of list-equality.

**Speed** is a latency signal. Claude's ~4.5× wall-clock advantage comes from the absence of hidden reasoning tokens (GPT and Gemini burn 4–8k reasoning tokens per hard example; Claude does not in the standard Messages API).

Four examples — `bit_count8`, `is_power_of_two`, `list_filter_even`, `list_unique` — were WEAK across all three proposers. The WEAK reasons were nearly identical: bounds-only theorems, membership instead of equality, etc. This suggests the bottleneck on those four is the prompt's theorem-shape guidance, not the model.

---

## What This Is (and Isn't)

**It is:** a concrete, runnable pipeline that chains LLM proposers with mechanical Lean verification and a structured critic. It demonstrates that for the class of pure, total, simply-typed Python functions, automated Lean translation and verification is feasible today.

**It isn't:** a security scanner for arbitrary production code. The current scope is intentionally narrow — single functions, no I/O, no external dependencies, no floats or dicts, no concurrency. The cost/side-channel theorems that would catch timing leaks are not yet auto-derived; the HMAC demo uses a hand-written Lean baseline.

The path from here to vulnerability finding runs through two open problems: (1) auto-deriving cost models alongside functional ones, and (2) mutation-kill testing to mechanically verify that proposed theorems distinguish correct from buggy implementations. Both are on the roadmap.

The main message from the benchmarks: **the gates work.** Gate D (differential testing) catches mistranslations that Lean would otherwise silently accept. Gate E (critic) catches vacuous theorems that A–D miss. And gate C (axiom allowlist) ensures that a `sorry`-stuffed proof can't sneak through as "verified." The combination is more trustworthy than any single check — and considerably more trustworthy than asking an LLM whether its own output is correct.

---

*Code, examples, and run artifacts: [github.com/phunterlau/code2lean](https://github.com/phunterlau/code2lean)*
