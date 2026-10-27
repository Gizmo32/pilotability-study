"""
Analyse croisée par domaine sur les 3 datasets
Génère une matrice de recommandations : modèle × domaine × mesure corrective

Usage:
    python analyze_domains_cross.py
    python analyze_domains_cross.py --export-csv
"""

import json
import glob
import argparse
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASETS = {
    "A": {
        "phase1.0": "resultats_phase1.0",
        "phase1.1": "resultats_phase1.1_relecture_simple",
        "phase1.2": "resultats_phase1.2_relecture_dirigee"
    },
    "B": {
        "phase1.0": "resultats_phase1.0_datasetB",
        "phase1.1": "resultats_phase1.1_datasetB",
        "phase1.2": "resultats_phase1.2_datasetB"
    },
    "C": {
        "phase1.0": "resultats_phase1.0_datasetC",
        "phase1.1": "resultats_phase1.1_datasetC",
        "phase1.2": "resultats_phase1.2_datasetC"
    }
}

MODELS = [
    "qwen2.5:latest",
    "llama3:latest",
    "mistral:latest",
    "gemma3:4b",
    "phi3:mini",
    "nemotron-mini:4b"
]

DOMAINS = ["logic", "math", "sci", "gen"]

# =============================================================================
# DATA LOADING
# =============================================================================

def load_judged_results(directory: str) -> Dict[str, List[dict]]:
    """Charge tous les résultats jugés d'un répertoire"""
    results = {}
    pattern = f"{directory}/*_judged_*.json"

    for file_path in glob.glob(pattern):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            model = data["metadata"]["model"]
            results[model] = data["questions"]
        except Exception as e:
            print(f"  Erreur chargement {file_path}: {e}")

    return results


def load_all_data() -> Dict:
    """Charge toutes les données des 3 datasets"""
    all_data = {}

    for dataset_name, paths in DATASETS.items():
        all_data[dataset_name] = {}
        for phase, directory in paths.items():
            print(f"  Chargement {dataset_name}/{phase}...")
            results = load_judged_results(directory)
            if results:
                all_data[dataset_name][phase] = results
            else:
                print(f"    [!] Aucun résultat trouvé dans {directory}")

    return all_data


# =============================================================================
# ANALYSIS
# =============================================================================

def compute_domain_accuracy(questions: List[dict], domain: str = None) -> Tuple[int, int]:
    """Calcule le nombre de réponses correctes pour un domaine (ou tous si domain=None)"""
    correct = 0
    total = 0

    for q in questions:
        if domain is None or q.get("domain") == domain:
            total += 1
            if q.get("is_correct", False):
                correct += 1

    return correct, total


def compute_domain_delta(baseline_questions: List[dict],
                         treatment_questions: List[dict],
                         domain: str = None) -> dict:
    """Calcule le delta entre baseline et traitement pour un domaine"""

    baseline_by_id = {q["id"]: q for q in baseline_questions}
    treatment_by_id = {q["id"]: q for q in treatment_questions}

    common_ids = set(baseline_by_id.keys()) & set(treatment_by_id.keys())

    improved = 0
    degraded = 0
    baseline_correct = 0
    treatment_correct = 0
    total = 0

    for qid in common_ids:
        bq = baseline_by_id[qid]
        tq = treatment_by_id[qid]

        if domain is not None and bq.get("domain") != domain:
            continue

        total += 1
        b_correct = bq.get("is_correct", False)
        t_correct = tq.get("is_correct", False)

        if b_correct:
            baseline_correct += 1
        if t_correct:
            treatment_correct += 1

        if not b_correct and t_correct:
            improved += 1
        elif b_correct and not t_correct:
            degraded += 1

    if total == 0:
        return None

    baseline_acc = baseline_correct / total * 100
    treatment_acc = treatment_correct / total * 100
    delta = treatment_acc - baseline_acc

    return {
        "baseline_acc": baseline_acc,
        "treatment_acc": treatment_acc,
        "delta": delta,
        "improved": improved,
        "degraded": degraded,
        "total": total
    }


def analyze_cross_datasets(all_data: Dict) -> Dict:
    """Analyse croisée par modèle, domaine et phase sur tous les datasets"""

    analysis = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for dataset_name, dataset_phases in all_data.items():
        if "phase1.0" not in dataset_phases:
            continue

        baseline_results = dataset_phases["phase1.0"]

        for phase in ["phase1.1", "phase1.2"]:
            if phase not in dataset_phases:
                continue

            treatment_results = dataset_phases[phase]

            for model in MODELS:
                if model not in baseline_results or model not in treatment_results:
                    continue

                for domain in DOMAINS:
                    delta_info = compute_domain_delta(
                        baseline_results[model],
                        treatment_results[model],
                        domain
                    )

                    if delta_info:
                        analysis[model][domain][phase].append({
                            "dataset": dataset_name,
                            **delta_info
                        })

    return analysis


def compute_recommendations(analysis: Dict) -> List[dict]:
    """Génère les recommandations basées sur les patterns reproductibles"""

    recommendations = []

    for model in MODELS:
        for domain in DOMAINS:
            for phase in ["phase1.1", "phase1.2"]:
                phase_data = analysis[model][domain][phase]

                if len(phase_data) < 2:  # Besoin d'au moins 2 datasets pour valider
                    continue

                # Calculer les statistiques agrégées
                deltas = [d["delta"] for d in phase_data]
                avg_delta = sum(deltas) / len(deltas)

                # Vérifier la cohérence (tous dans la même direction)
                all_positive = all(d > 0 for d in deltas)
                all_negative = all(d < 0 for d in deltas)
                consistent = all_positive or all_negative

                # Déterminer la recommandation
                if consistent and all_positive and avg_delta > 5:
                    recommendation = "RECOMMANDE"
                    effect = "amelioration"
                elif consistent and all_negative and avg_delta < -5:
                    recommendation = "EVITER"
                    effect = "regression"
                elif abs(avg_delta) < 3:
                    recommendation = "NEUTRE"
                    effect = "stable"
                else:
                    recommendation = "VARIABLE"
                    effect = "inconsistent"

                phase_label = "Relecture simple" if phase == "phase1.1" else "Relecture dirigee"

                recommendations.append({
                    "model": model,
                    "domain": domain,
                    "correction_type": phase_label,
                    "phase": phase,
                    "avg_delta": avg_delta,
                    "min_delta": min(deltas),
                    "max_delta": max(deltas),
                    "n_datasets": len(deltas),
                    "consistent": consistent,
                    "effect": effect,
                    "recommendation": recommendation,
                    "details": phase_data
                })

    return recommendations


# =============================================================================
# OUTPUT
# =============================================================================

def print_domain_analysis(analysis: Dict):
    """Affiche l'analyse par domaine"""

    print("\n" + "=" * 100)
    print("ANALYSE PAR DOMAINE - RESULTATS CROISES SUR 3 DATASETS")
    print("=" * 100)

    for domain in DOMAINS:
        print(f"\n{'='*100}")
        print(f"DOMAINE : {domain.upper()}")
        print("=" * 100)

        # Phase 1.1
        print(f"\n--- Phase 1.1 (Relecture simple) ---")
        print(f"{'Modele':<20} | {'Dataset A':>12} | {'Dataset B':>12} | {'Dataset C':>12} | {'Moyenne':>10} | {'Coherent':>8}")
        print("-" * 90)

        for model in MODELS:
            model_short = model.replace(":latest", "").replace(":4b", "").replace(":mini", "")
            phase_data = analysis[model][domain]["phase1.1"]

            if not phase_data:
                continue

            deltas_by_dataset = {d["dataset"]: d["delta"] for d in phase_data}
            deltas = [d["delta"] for d in phase_data]
            avg = sum(deltas) / len(deltas) if deltas else 0

            consistent = "Oui" if (all(d > 0 for d in deltas) or all(d < 0 for d in deltas)) else "Non"

            print(f"{model_short:<20} | "
                  f"{deltas_by_dataset.get('A', 'N/A'):>+11.1f}% | "
                  f"{deltas_by_dataset.get('B', 'N/A'):>+11.1f}% | "
                  f"{deltas_by_dataset.get('C', 'N/A'):>+11.1f}% | "
                  f"{avg:>+9.1f}% | "
                  f"{consistent:>8}")

        # Phase 1.2
        print(f"\n--- Phase 1.2 (Relecture dirigee) ---")
        print(f"{'Modele':<20} | {'Dataset A':>12} | {'Dataset B':>12} | {'Dataset C':>12} | {'Moyenne':>10} | {'Coherent':>8}")
        print("-" * 90)

        for model in MODELS:
            model_short = model.replace(":latest", "").replace(":4b", "").replace(":mini", "")
            phase_data = analysis[model][domain]["phase1.2"]

            if not phase_data:
                continue

            deltas_by_dataset = {d["dataset"]: d["delta"] for d in phase_data}
            deltas = [d["delta"] for d in phase_data]
            avg = sum(deltas) / len(deltas) if deltas else 0

            consistent = "Oui" if (all(d > 0 for d in deltas) or all(d < 0 for d in deltas)) else "Non"

            print(f"{model_short:<20} | "
                  f"{deltas_by_dataset.get('A', 'N/A'):>+11.1f}% | "
                  f"{deltas_by_dataset.get('B', 'N/A'):>+11.1f}% | "
                  f"{deltas_by_dataset.get('C', 'N/A'):>+11.1f}% | "
                  f"{avg:>+9.1f}% | "
                  f"{consistent:>8}")


def print_recommendations_matrix(recommendations: List[dict]):
    """Affiche la matrice de recommandations"""

    print("\n" + "=" * 100)
    print("MATRICE DE RECOMMANDATIONS")
    print("Pour chaque tache (domaine), quel LLM utiliser avec quelle mesure corrective")
    print("=" * 100)

    # Grouper par domaine
    by_domain = defaultdict(list)
    for r in recommendations:
        by_domain[r["domain"]].append(r)

    for domain in DOMAINS:
        print(f"\n{'='*100}")
        print(f"DOMAINE : {domain.upper()}")
        print("=" * 100)

        domain_recs = by_domain[domain]

        # Séparer par type de correction
        for phase_label in ["Relecture simple", "Relecture dirigee"]:
            phase_recs = [r for r in domain_recs if r["correction_type"] == phase_label]

            if not phase_recs:
                continue

            print(f"\n{phase_label}:")
            print("-" * 80)

            # Trier par recommandation
            for rec_type in ["RECOMMANDE", "NEUTRE", "VARIABLE", "EVITER"]:
                recs = [r for r in phase_recs if r["recommendation"] == rec_type]

                if not recs:
                    continue

                emoji = {"RECOMMANDE": "[+]", "EVITER": "[-]", "NEUTRE": "[=]", "VARIABLE": "[?]"}[rec_type]

                for r in recs:
                    model_short = r["model"].replace(":latest", "").replace(":4b", "").replace(":mini", "")
                    print(f"  {emoji} {model_short:<18} | Delta moyen: {r['avg_delta']:>+6.1f}% | "
                          f"Range: [{r['min_delta']:>+5.1f}%, {r['max_delta']:>+5.1f}%] | "
                          f"Coherent: {'Oui' if r['consistent'] else 'Non'}")


def print_summary_table(recommendations: List[dict]):
    """Affiche un tableau de synthese final"""

    print("\n" + "=" * 100)
    print("TABLEAU DE SYNTHESE : RECOMMANDATIONS PAR DOMAINE ET MODELE")
    print("=" * 100)
    print("\nLegende: [+] Recommande | [-] Eviter | [=] Neutre | [?] Variable")
    print()

    # Construire la matrice
    print(f"{'Modele':<18} | {'LOGIC 1.1':^9} | {'LOGIC 1.2':^9} | {'MATH 1.1':^9} | {'MATH 1.2':^9} | {'SCI 1.1':^9} | {'SCI 1.2':^9} | {'GEN 1.1':^9} | {'GEN 1.2':^9}")
    print("-" * 115)

    for model in MODELS:
        model_short = model.replace(":latest", "").replace(":4b", "").replace(":mini", "")
        row = f"{model_short:<18} |"

        for domain in DOMAINS:
            for phase in ["phase1.1", "phase1.2"]:
                rec = next((r for r in recommendations
                           if r["model"] == model and r["domain"] == domain and r["phase"] == phase), None)

                if rec:
                    emoji = {"RECOMMANDE": "  [+]  ", "EVITER": "  [-]  ", "NEUTRE": "  [=]  ", "VARIABLE": "  [?]  "}
                    cell = emoji.get(rec["recommendation"], "   ?   ")
                else:
                    cell = "   -   "

                row += f" {cell} |"

        print(row)

    print()
    print("1.1 = Relecture simple | 1.2 = Relecture dirigee")


def export_to_csv(recommendations: List[dict], filename: str):
    """Exporte les recommandations en CSV"""

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "domain", "correction_type", "recommendation",
            "avg_delta", "min_delta", "max_delta", "n_datasets", "consistent", "effect"
        ])
        writer.writeheader()

        for r in recommendations:
            writer.writerow({
                "model": r["model"],
                "domain": r["domain"],
                "correction_type": r["correction_type"],
                "recommendation": r["recommendation"],
                "avg_delta": round(r["avg_delta"], 2),
                "min_delta": round(r["min_delta"], 2),
                "max_delta": round(r["max_delta"], 2),
                "n_datasets": r["n_datasets"],
                "consistent": r["consistent"],
                "effect": r["effect"]
            })

    print(f"\nExporte: {filename}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyse croisee par domaine")
    parser.add_argument("--export-csv", action="store_true", help="Exporter en CSV")
    args = parser.parse_args()

    print("=" * 100)
    print("ANALYSE CROISEE PAR DOMAINE SUR 3 DATASETS")
    print("=" * 100)
    print()

    # Charger les données
    print("Chargement des donnees...")
    all_data = load_all_data()

    # Analyser
    print("\nAnalyse en cours...")
    analysis = analyze_cross_datasets(all_data)

    # Générer les recommandations
    recommendations = compute_recommendations(analysis)

    # Afficher les résultats
    print_domain_analysis(analysis)
    print_recommendations_matrix(recommendations)
    print_summary_table(recommendations)

    # Export CSV si demandé
    if args.export_csv:
        export_to_csv(recommendations, "recommendations_by_domain.csv")

    print("\n" + "=" * 100)
    print("ANALYSE TERMINEE")
    print("=" * 100)


if __name__ == "__main__":
    main()
