# -*- coding: utf-8 -*-
"""Compute per-model accuracy and paired answer transitions (Correct/Incorrect)
between the baseline phase (1.0) and each review phase (1.1, 1.2), for the
three datasets (A, B, C). Sanity-checks accuracies against TABLES_PUBLICATION_v2.md
and emits the transition counts used in the paper (Table: transitions).
"""
import json
import glob
import os
import re
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..")

DIRS = {
    ("1.0", "A"): "resultats_phase1.0",
    ("1.0", "B"): "resultats_phase1.0_datasetB",
    ("1.0", "C"): "resultats_phase1.0_datasetC",
    ("1.1", "A"): "resultats_phase1.1_relecture_simple",
    ("1.1", "B"): "resultats_phase1.1_datasetB",
    ("1.1", "C"): "resultats_phase1.1_datasetC",
    ("1.2", "A"): "resultats_phase1.2_relecture_dirigee",
    ("1.2", "B"): "resultats_phase1.2_datasetB",
    ("1.2", "C"): "resultats_phase1.2_datasetC",
}

MODELS = [
    "qwen2.5_latest",
    "gemma3_4b",
    "phi3_mini",
    "mistral_latest",
    "llama3_latest",
    "nemotron-mini_4b",
]


def load_judged(phase, dataset, model):
    d = os.path.join(ROOT, DIRS[(phase, dataset)])
    files = sorted(glob.glob(os.path.join(d, f"results_{model}_*judged*.json")))
    if not files:
        return None
    with open(files[-1], encoding="utf-8") as f:
        data = json.load(f)
    return {q["id"]: bool(q.get("is_correct")) for q in data["questions"]}


def main():
    print(f"{'model':<20} {'ds':<3} {'phase':<6} {'acc_base':>8} {'acc_rev':>8} "
          f"{'C>C':>4} {'C>I':>4} {'I>C':>4} {'I>I':>4}")
    rows = []
    for model in MODELS:
        for dataset in ["A", "B", "C"]:
            base = load_judged("1.0", dataset, model)
            if base is None:
                print(f"MISSING baseline {model} {dataset}")
                continue
            for phase in ["1.1", "1.2"]:
                rev = load_judged(phase, dataset, model)
                if rev is None:
                    print(f"MISSING {phase} {model} {dataset}")
                    continue
                common = sorted(set(base) & set(rev))
                cc = sum(base[i] and rev[i] for i in common)
                ci = sum(base[i] and not rev[i] for i in common)
                ic = sum(not base[i] and rev[i] for i in common)
                ii = sum(not base[i] and not rev[i] for i in common)
                n = len(common)
                acc_b = 100.0 * (cc + ci) / n
                acc_r = 100.0 * (cc + ic) / n
                print(f"{model:<20} {dataset:<3} {phase:<6} {acc_b:>7.1f}% {acc_r:>7.1f}% "
                      f"{cc:>4} {ci:>4} {ic:>4} {ii:>4}")
                rows.append(dict(model=model, dataset=dataset, phase=phase, n=n,
                                 acc_base=acc_b, acc_rev=acc_r,
                                 CC=cc, CI=ci, IC=ic, II=ii))
    out = os.path.join(os.path.dirname(__file__), "transitions.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
