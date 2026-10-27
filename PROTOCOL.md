# Experimental protocol

This document describes the full methodology behind the paper, in enough detail to
reproduce or audit it independently of the manuscript text. It mirrors the paper's
§3 (Experimental Protocol) and Appendix A (Prompting protocols and design rationale).

## 1. Design rationale: why this protocol looks the way it does

The starting question was simple: do review mechanisms (asking a model to re-examine
a prior answer) actually improve the reliability of local LLMs? Two failure modes had
to be designed out from the start, because either one would have made a positive
result meaningless:

1. **Confirmation bias.** It is easy, when hoping a mechanism works, to unconsciously
   favor datasets, prompts, or analyses that produce the expected direction. The
   countermeasures here are: (a) three *independently constructed* closed-form
   datasets rather than one, so that a result has to replicate to count; (b) a fixed,
   deterministic rule-based judge instead of a human or LLM judge free to interpret
   ambiguous cases favorably; (c) paired statistics (McNemar's exact test with
   Benjamini–Hochberg correction across models) computed the same way regardless of
   which direction the result points.
2. **Role confusion in the tested models.** A review prompt phrased as "was this
   answer correct?" invites the model to switch from an *answering* behavior to a
   *judging* behavior — a different task that contaminates the measurement of whether
   review improves answers. The countermeasure is a **stateless, responder-only**
   design (Section 3 below): the prior answer is shown as an anonymous external
   artifact ("an answer attempt has been proposed"), never attributed to the model,
   and the instructions explicitly forbid commenting on, validating, or revising it —
   only answering the question is allowed. Formulations that were tested and
   explicitly excluded for inducing meta-evaluation or sycophantic agreement are
   listed in `prompts/phase_1.1_simple_review.md`.

The result that emerged — effects that are strongly heterogeneous and unpredictable
across models and datasets, rather than a reliable improvement — was not the expected
outcome going in. The protocol was built to be able to show that just as clearly as
the opposite.

## 2. Models and inference configuration

Six instruction-tuned models between 3.8B and 8B parameters, chosen to cover distinct
model families in common local deployment use, all executed locally through Ollama
with identical settings across all phases:

| Short ID | Ollama tag (pulled) | Params | Family |
|---|---|---|---|
| `qwen2.5:7b` | `qwen2.5:latest` (2026-01) | 7B | Qwen 2.5 |
| `llama3:8b` | `llama3:latest` (2026-01) | 8B | Llama 3 |
| `gemma3:4b` | `gemma3:4b` (2026-01) | 4B | Gemma 3 |
| `mistral:7b` | `mistral:latest` (2026-01) | 7B | Mistral |
| `phi3:mini` | `phi3:mini` (2026-01) | 3.8B | Phi-3 |
| `nemotron-mini:4b` | `nemotron-mini:4b` (2026-01) | 4B | Nemotron/Minitron |

All instruction-tuned variants, 4-bit quantization (Ollama defaults). Decoding:
`temperature=0.1`, `top_p=0.9`, `repeat_penalty=1.0`, 200-token completion limit,
identical for every model and phase. Temperature 0.1 was the lowest setting reliably
supported across all six models under the serving stack; residual stochasticity is
mitigated by the paired design and by replication over three datasets. No
fine-tuning, retrieval, system-prompt customization per model, or tool use was
applied. Runs took place on a single consumer workstation (Windows 10 Professional,
Intel Core i5-8400 CPU, 32 GB RAM, NVIDIA GeForce RTX 2080 SUPER, 8 GB VRAM); the
exact Ollama build in use at the time of the runs (January 2026) was not recorded.

## 3. Datasets

Three independently constructed closed-form datasets in French: Dataset A (n=50),
Dataset B (n=60), Dataset C (n=50) — 160 questions total, no overlap between
datasets. Each question admits a single unambiguous short answer (boolean, number, or
short canonical entity) and spans four domains: logic, mathematics, science, and
general knowledge (roughly balanced per dataset; see `datasets/`).

Dataset A was used for exploratory analysis and protocol validation. Datasets B and C
were written afterward to the same fixed specification (closed-form, four balanced
domains, no cross-dataset overlap); their questions were finalized before their
respective runs, and question selection did not use Dataset A's per-item model
outcomes. B and C are treated as replication sets rather than as a preregistered
confirmatory test, and Dataset A effects are reported as exploratory throughout the
paper. The protocol, models, and analysis plan were not preregistered — this is a
controlled pilot study, not a confirmatory trial; a larger, preregistered dataset is
future work.

## 4. Correction phases

Three conditions, applied to the same questions (full verbatim templates in
`prompts/`):

- **Phase 1.0 (baseline).** The model receives the question with instructions to
  output only the expected closed-form value, without explanation.
- **Phase 1.1 (stateless simple review).** The model is shown the question together
  with its Phase-1.0 answer, presented as an external artifact ("an answer attempt
  has been proposed"), with no indication of authorship or correctness. Instructed to
  answer directly — not to comment on, validate, or revise the previous attempt.
- **Phase 1.2 (stateless directed review).** Identical to Phase 1.1, plus a short
  list of domain-specific attention cues ("useful hints") drawn from a fixed
  dictionary keyed by (domain, subdomain). Cues are written at the subdomain level,
  never reference the specific expected answer, and are identical for every model.

The review phases reuse the archived Phase-1.0 answer rather than regenerating it,
guaranteeing that both review conditions are paired against the same baseline
response. See `prompts/phase_1.1_simple_review.md` for the anti-patterns explicitly
excluded from all review prompts.

## 5. Evaluation (judging)

All outputs are scored by a deterministic rule-based judge (`scripts/judge_results.py`):
exact match after normalization, plus a restricted equivalence table
(`true/vrai/oui`, `false/faux/non`, digit–word numeral equivalence). Ambiguous,
overlong, off-format, or meta-evaluative outputs are conservatively scored as
incorrect — no LLM judge is used for any reported number. Human verification of all
judged outputs was performed by a single annotator (the author); two isolated
labeling errors were found and corrected before analysis, with no systematic pattern.

Across all 1,920 review-phase inferences, only 19 outputs (1.0%) matched a
meta-evaluative pattern (1.4% under simple review, 0.6% under directed review),
confirming the stateless responder-only framing held in practice; all 19 are scored
incorrect by the conservative judge regardless.

## 6. Statistical analysis

Implemented in `scripts/statistical_analysis.py`:

- Accuracy reported with 95% Wilson score intervals.
- Phase effects tested per (model, dataset) with McNemar's exact test on paired
  binary outcomes, bootstrap 95% CIs on accuracy deltas, and Benjamini–Hochberg
  correction across the six models within each phase–dataset family.
- A complementary **pooled stratified McNemar test** (`pooled_stratified_mcnemar()`)
  pools the three datasets as strata per model, to ask whether an effect exists with
  maximal power (as opposed to the per-dataset tests, which ask whether it
  *replicates*).
- Full paired transition counts (Correct→Incorrect, Incorrect→Correct) computed by
  `scripts/compute_transitions.py`.
- Per-(model, domain, mechanism) aggregation computed by
  `scripts/analyze_domains_cross.py`, output in `analysis/recommendations_by_domain.csv`.

The judge's conservative treatment of format drift (ambiguous/off-format outputs
scored incorrect) is a deliberate design choice, not an oversight: a pipeline
consuming the output fails either way, so format destabilization is treated as a real
failure mode of review. It accounts for at most 1.9% of correct→incorrect transitions
overall (5 of 261), all from the single weakest model.

## Scope

This protocol — and this repository — covers Phases 1.0–1.2 only (baseline, simple
review, directed review), which is the full scope of the associated paper. Other
correction mechanisms (chain-of-thought, evidence-injection, self-consistency,
reflection, decomposition) are listed as future work in the paper but were not run
under this protocol and are not part of this release.
