# Pilotability study — reproducibility package

Code, datasets, prompts, and raw results for:

> **Baseline Accuracy Is Not Enough: Prompt-Based Correction Is an Unpredictable
> Intervention in Local Language Models**
> Romain Maudru, 2026. Preprint (arXiv link: *TBD once submitted*).

## Motivation

The initial question behind this study was simple: do review mechanisms (asking a
model to re-examine a prior answer) actually improve the reliability of local LLMs?
Answering it without fooling ourselves meant designing a protocol that avoided two
failure modes, either of which would have made a positive result worthless:

- **Confirmation bias** — building or reading the data in a way that favors the
  hypothesis that review helps.
- **Role confusion** in the tested models — a review prompt phrased as "was this
  correct?" invites a model to switch from *answering* to *judging*. That is a
  different behavior, and it would contaminate the measurement.

The stateless, responder-only prompt design and the exploratory/replication dataset
split, both described in `PROTOCOL.md`, are direct consequences of guarding against
these two failure modes. The result that emerged — unpredictability rather than a
reliable improvement — was not the expected outcome.

## What this study found

Six locally deployable LLMs (3.8B–8B parameters) were tested under a controlled,
paired protocol: 160 closed-form questions across three independent datasets and
four domains, evaluated under (1) single-pass baseline answering, (2) stateless
simple review, and (3) stateless directed review with attention cues — 2,880
inferences in total, scored with a deterministic rule-based judge and analyzed with
paired statistics (McNemar's exact test, Benjamini–Hochberg correction).

Correction effects are strongly heterogeneous and unpredictable: the same
intervention produces a significant gain for one model and a significant loss for
another on the same items; no significant improvement replicates across datasets;
and the stronger intervention (directed review) is strictly more harmful than the
weaker one. Baseline accuracy does not predict the direction of the response. Full
results and discussion are in the paper.

## Repository structure

```
PROTOCOL.md  Full experimental protocol: design rationale, phases, judging, stats
datasets/    Three closed-form question datasets (JSONL), no cross-dataset overlap
prompts/     Verbatim prompt templates for the three phases (French + English)
scripts/     Phase execution, judging, statistical analysis, and figure generation
results/     Per-item raw and judged outputs (9 phase x dataset combinations)
analysis/    Derived tables (e.g. per-domain effect aggregation)
```

See [`PROTOCOL.md`](PROTOCOL.md) for the full methodology, including why the
protocol is designed the way it is (see Motivation above).

### `datasets/`

- `dataset_a.jsonl` (n=50) — exploratory / protocol validation
- `dataset_b.jsonl` (n=60) — replication set
- `dataset_c.jsonl` (n=50) — replication set

Each record: `id`, `domain`, `subdomain`, `question`, `expected_answer` (plus
evidence fields used by later, unrelated experimental phases not analyzed in this
paper). Domains: logic, mathematics, science, general knowledge.

### `prompts/`

Verbatim templates for Phase 1.0 (baseline), 1.1 (stateless simple review), and
1.2 (stateless directed review), with English translations and the documented
anti-patterns that were deliberately avoided.

### `scripts/`

| Script | Role |
|---|---|
| `client.py` | Minimal Ollama API client used by all phase runners |
| `run_phase1.0.py`, `run_phase1.1.py`, `run_phase1.2.py` | Run one phase for one model |
| `run_all_models1.0.py`, `run_all_models1.1.py`, `run_all_models1.2.py` | Run one phase across all six models |
| `judge_results.py` | Deterministic rule-based scoring of raw outputs |
| `statistical_analysis.py` | Wilson intervals, McNemar's test, bootstrap CIs, Benjamini–Hochberg correction, pooled stratified McNemar across datasets |
| `compute_transitions.py` | Paired Correct/Incorrect transition counts per model/dataset/phase |
| `generate_figures.py` | Renders the paper's three figures from its published accuracy/delta tables (values are embedded in the script, not re-read from `results/`) |
| `analyze_domains_cross.py` | Produces `analysis/recommendations_by_domain.csv` |

### `results/`

Nine directories, one per (phase, dataset) pair: `phase{1.0,1.1,1.2}_dataset{A,B,C}`.
Each contains, per model, a raw output file and a judged output file (`*_judged_*.json`)
— 6 models x 2 files x 9 combinations = 108 files, 2,880 scored inferences in total.

### `analysis/`

`recommendations_by_domain.csv` — mean/min/max accuracy delta per (model, domain,
mechanism) cell across the three datasets, with a consistency flag (same sign on
all three datasets). Underlies the per-domain appendix table in the paper.
`transitions.json` — paired Correct/Incorrect transition counts per model, dataset,
and phase, as produced by `scripts/compute_transitions.py` (underlies the paper's
transitions table).

## Reproducing

```bash
pip install -r requirements.txt

# Requires a local Ollama server with the six models pulled
# (qwen2.5:7b, llama3:8b, gemma3:4b, mistral:7b, phi3:mini, nemotron-mini:4b)

# 1. Collect raw answers: baseline first, then each review phase (which reuses the
#    baseline answer via --baseline-dir), repeated for each dataset (A/B/C):
python scripts/run_all_models1.0.py --dataset datasets/dataset_a.jsonl --output-dir results/phase1.0_datasetA
python scripts/run_all_models1.1.py --dataset datasets/dataset_a.jsonl --output-dir results/phase1.1_datasetA --baseline-dir results/phase1.0_datasetA
python scripts/run_all_models1.2.py --dataset datasets/dataset_a.jsonl --output-dir results/phase1.2_datasetA --baseline-dir results/phase1.0_datasetA

# 2. Score with the deterministic judge used in the paper (not the default LLM judge).
#    Run this right after step 1, before any *_judged_* file exists in the output
#    directory, otherwise the glob below will also re-judge already-judged files:
python scripts/judge_results.py "results/phase1.0_datasetA/results_*.json" --simple-compare

# 3. Analyze — this can be run directly on the released results/ without redoing 1-2:
python scripts/statistical_analysis.py --phase1.0 "results/phase1.0_datasetA/*_judged_*.json" --phase1.1 "results/phase1.1_datasetA/*_judged_*.json"
python scripts/compute_transitions.py
python scripts/analyze_domains_cross.py --export-csv
python scripts/generate_figures.py
```

Decoding was fixed across all phases and models: `temperature=0.1`, `top_p=0.9`,
`repeat_penalty=1.0`, 200-token completion limit. The hardware used for the original
runs is documented in [`PROTOCOL.md`](PROTOCOL.md) §2.

## Scope note

This repository covers only Phases 1.0–1.2 (baseline, simple review, directed
review) — the full scope of this paper. Later experimental phases (chain-of-thought,
evidence-injection, etc.) belong to a separate, unpublished follow-up study and are
not included here.

## License

- Code (`scripts/`): [MIT](LICENSE)
- Data and text (`datasets/`, `prompts/`, `results/`, `analysis/`): [CC BY 4.0](LICENSE-DATA)

## Citation

See [`CITATION.cff`](CITATION.cff). The arXiv identifier will be added once the
preprint is submitted.
