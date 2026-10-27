"""
Script d'analyse statistique pour publication scientifique

Fonctionnalités:
- Test de McNemar (comparaison appariée baseline vs correction)
- Intervalles de confiance Wilson (95%)
- Bootstrap IC pour les deltas
- Correction Benjamini-Hochberg pour comparaisons multiples
- Génération de tableaux formatés pour publication

Usage:
    python statistical_analysis.py --phase1.0 resultats_phase1.0/*.json --phase1.1 resultats_phase1.1/*.json
    python statistical_analysis.py --all-phases --dataset-suffix "_datasetB"
"""

import json
import glob
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================

def wilson_confidence_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calcule l'intervalle de confiance Wilson pour une proportion.
    Plus fiable que l'intervalle normal pour petits échantillons ou proportions extrêmes.

    Args:
        successes: Nombre de succès
        total: Nombre total d'essais
        confidence: Niveau de confiance (défaut: 0.95)

    Returns:
        (lower_bound, upper_bound) en pourcentage
    """
    if total == 0:
        return (0.0, 0.0)

    from scipy import stats
    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    p_hat = successes / total

    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    margin = (z / denominator) * np.sqrt(p_hat * (1 - p_hat) / total + z**2 / (4 * total**2))

    lower = max(0, center - margin) * 100
    upper = min(1, center + margin) * 100

    return (lower, upper)


def mcnemar_test(baseline_correct: List[bool], treatment_correct: List[bool]) -> Tuple[float, float, dict]:
    """
    Test de McNemar pour données appariées binaires.
    Compare si le traitement (correction) change significativement les résultats.

    Args:
        baseline_correct: Liste de booléens (correct/incorrect) pour baseline
        treatment_correct: Liste de booléens (correct/incorrect) pour traitement

    Returns:
        (statistic, p_value, contingency_table)
    """
    if len(baseline_correct) != len(treatment_correct):
        raise ValueError("Les listes doivent avoir la même longueur")

    # Construire la table de contingence 2x2
    # b = baseline correct, treatment incorrect (dégradation)
    # c = baseline incorrect, treatment correct (amélioration)
    b = sum(1 for bc, tc in zip(baseline_correct, treatment_correct) if bc and not tc)
    c = sum(1 for bc, tc in zip(baseline_correct, treatment_correct) if not bc and tc)
    a = sum(1 for bc, tc in zip(baseline_correct, treatment_correct) if bc and tc)
    d = sum(1 for bc, tc in zip(baseline_correct, treatment_correct) if not bc and not tc)

    contingency = {'a_both_correct': a, 'b_degraded': b, 'c_improved': c, 'd_both_incorrect': d}

    # Test de McNemar exact (pour petits échantillons)
    from scipy import stats

    # Utiliser le test exact si b + c < 25
    if b + c < 25:
        # Test binomial exact
        if b + c == 0:
            return (0.0, 1.0, contingency)
        p_value = stats.binomtest(min(b, c), b + c, 0.5, alternative='two-sided').pvalue
        statistic = (b - c)**2 / (b + c) if (b + c) > 0 else 0
    else:
        # Test chi-carré avec correction de continuité
        statistic = (abs(b - c) - 1)**2 / (b + c)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return (statistic, p_value, contingency)


def bootstrap_delta_ci(baseline_correct: List[bool], treatment_correct: List[bool],
                       n_bootstrap: int = 10000, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Calcule l'intervalle de confiance bootstrap pour le delta (différence d'accuracy).

    Args:
        baseline_correct: Liste de booléens pour baseline
        treatment_correct: Liste de booléens pour traitement
        n_bootstrap: Nombre d'échantillons bootstrap
        confidence: Niveau de confiance

    Returns:
        (delta_mean, ci_lower, ci_upper) en pourcentage
    """
    baseline = np.array(baseline_correct, dtype=float)
    treatment = np.array(treatment_correct, dtype=float)
    n = len(baseline)

    # Delta observé
    delta_observed = (treatment.mean() - baseline.mean()) * 100

    # Bootstrap
    np.random.seed(42)  # Reproductibilité
    deltas = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        delta_boot = (treatment[indices].mean() - baseline[indices].mean()) * 100
        deltas.append(delta_boot)

    # Intervalles de confiance (percentile method)
    alpha = 1 - confidence
    ci_lower = np.percentile(deltas, alpha / 2 * 100)
    ci_upper = np.percentile(deltas, (1 - alpha / 2) * 100)

    return (delta_observed, ci_lower, ci_upper)


def benjamini_hochberg_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    """
    Correction de Benjamini-Hochberg pour comparaisons multiples (contrôle FDR).

    Args:
        p_values: Liste des p-values brutes
        alpha: Seuil de significativité (défaut: 0.05)

    Returns:
        (adjusted_p_values, is_significant)
    """
    n = len(p_values)
    if n == 0:
        return ([], [])

    # Trier les p-values avec leurs indices originaux
    indexed_pvals = [(p, i) for i, p in enumerate(p_values)]
    indexed_pvals.sort(key=lambda x: x[0])

    # Calculer les p-values ajustées
    adjusted = [0.0] * n
    cummin = 1.0

    for rank, (p, original_idx) in enumerate(reversed(indexed_pvals)):
        adjusted_p = min(cummin, p * n / (n - rank))
        adjusted_p = min(adjusted_p, 1.0)
        cummin = min(cummin, adjusted_p)
        adjusted[original_idx] = adjusted_p

    # Déterminer la significativité
    significant = [p <= alpha for p in adjusted]

    return (adjusted, significant)


def pooled_stratified_mcnemar(baseline_by_dataset: Dict[str, Dict[str, bool]],
                               treatment_by_dataset: Dict[str, Dict[str, bool]]) -> Dict[str, dict]:
    """
    Test de McNemar stratifié (type CMH), poolé sur plusieurs datasets, par modèle.

    Args:
        baseline_by_dataset: {dataset: {question_id: is_correct}} pour la baseline
        treatment_by_dataset: {dataset: {question_id: is_correct}} pour le traitement,
            mêmes clés dataset/modèle que baseline_by_dataset

    Returns:
        {model: {degraded, improved, n, delta, p, p_adjusted, significant}}
    """
    from scipy import stats

    out: Dict[str, dict] = {}
    models = set()
    for per_model in baseline_by_dataset.values():
        models.update(per_model.keys())

    for model in sorted(models):
        degraded = improved = n_total = 0
        for dataset in baseline_by_dataset:
            base = baseline_by_dataset[dataset].get(model)
            trt = treatment_by_dataset.get(dataset, {}).get(model)
            if base is None or trt is None:
                continue
            common_ids = sorted(set(base) & set(trt))
            degraded += sum(1 for q in common_ids if base[q] and not trt[q])
            improved += sum(1 for q in common_ids if not base[q] and trt[q])
            n_total += len(common_ids)

        discordant = degraded + improved
        delta = 100.0 * (improved - degraded) / n_total if n_total else 0.0
        p_exact = (stats.binomtest(min(degraded, improved), discordant, 0.5,
                                    alternative='two-sided').pvalue
                   if discordant else 1.0)
        out[model] = dict(degraded=degraded, improved=improved, n=n_total, delta=delta, p=p_exact)

    ordered_models = sorted(out.keys())
    adjusted, significant = benjamini_hochberg_correction([out[m]['p'] for m in ordered_models])
    for model, p_adj, sig in zip(ordered_models, adjusted, significant):
        out[model]['p_adjusted'] = p_adj
        out[model]['significant'] = sig

    return out


# =============================================================================
# DATA LOADING
# =============================================================================

@dataclass
class ModelResults:
    """Résultats d'un modèle pour une phase"""
    model: str
    phase: str
    dataset: str
    total: int
    correct: int
    incorrect: int
    uncertain: int
    questions: List[dict]

    @property
    def accuracy(self) -> float:
        return self.correct / self.total * 100 if self.total > 0 else 0.0

    def get_correctness_vector(self) -> List[bool]:
        """Retourne un vecteur de booléens (correct/incorrect) par question ID"""
        return [q.get('is_correct', False) for q in self.questions]

    def get_correctness_by_id(self) -> Dict[str, bool]:
        """Retourne un dict {question_id: is_correct}"""
        return {q['id']: q.get('is_correct', False) for q in self.questions}


def load_judged_results(file_path: str) -> Optional[ModelResults]:
    """Charge un fichier de résultats jugés"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        metadata = data.get('metadata', {})
        questions = data.get('questions', [])

        # Vérifier que le fichier a été jugé
        if not metadata.get('judge_metadata', {}).get('judged', False):
            print(f"  [!] Fichier non jugé: {file_path}")
            return None

        # Compter les résultats
        correct = sum(1 for q in questions if q.get('is_correct') is True)
        incorrect = sum(1 for q in questions if q.get('is_correct') is False)
        uncertain = sum(1 for q in questions if q.get('is_correct') is None)

        # Déterminer la phase depuis le chemin
        path = Path(file_path)
        phase = "unknown"
        if "phase1.0" in str(path):
            phase = "1.0"
        elif "phase1.1" in str(path):
            phase = "1.1"
        elif "phase1.2" in str(path):
            phase = "1.2"
        elif "phase1.3" in str(path):
            phase = "1.3"
        elif "phase1.4" in str(path):
            phase = "1.4"

        # Déterminer le dataset depuis le chemin
        dataset = "A"
        if "datasetB" in str(path) or "dataset_b" in str(path).lower():
            dataset = "B"
        elif "datasetC" in str(path) or "dataset_c" in str(path).lower():
            dataset = "C"

        return ModelResults(
            model=metadata.get('model', 'unknown'),
            phase=phase,
            dataset=dataset,
            total=len(questions),
            correct=correct,
            incorrect=incorrect,
            uncertain=uncertain,
            questions=questions
        )

    except Exception as e:
        print(f"  [!] Erreur chargement {file_path}: {e}")
        return None


def load_phase_results(pattern: str) -> Dict[str, ModelResults]:
    """Charge tous les résultats d'une phase, indexés par modèle"""
    results = {}
    files = glob.glob(pattern)

    # Filtrer pour ne garder que les fichiers jugés (contenant "_judged_")
    judged_files = [f for f in files if '_judged_' in f]

    if not judged_files:
        # Si pas de fichiers jugés, essayer tous les fichiers
        judged_files = files

    for file_path in judged_files:
        result = load_judged_results(file_path)
        if result:
            results[result.model] = result

    return results


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_accuracy_table(results_by_phase: Dict[str, Dict[str, ModelResults]]) -> List[dict]:
    """
    Génère un tableau d'accuracy avec intervalles de confiance Wilson.

    Returns:
        Liste de dicts avec colonnes pour chaque phase
    """
    # Collecter tous les modèles
    all_models = set()
    for phase_results in results_by_phase.values():
        all_models.update(phase_results.keys())

    table = []
    for model in sorted(all_models):
        row = {'model': model}

        for phase, phase_results in results_by_phase.items():
            if model in phase_results:
                result = phase_results[model]
                acc = result.accuracy
                ci_low, ci_high = wilson_confidence_interval(result.correct, result.total)
                row[f'phase_{phase}_acc'] = acc
                row[f'phase_{phase}_ci'] = (ci_low, ci_high)
                row[f'phase_{phase}_n'] = result.total
            else:
                row[f'phase_{phase}_acc'] = None
                row[f'phase_{phase}_ci'] = None
                row[f'phase_{phase}_n'] = None

        table.append(row)

    return table


def compute_delta_analysis(baseline_results: Dict[str, ModelResults],
                          treatment_results: Dict[str, ModelResults],
                          baseline_phase: str = "1.0",
                          treatment_phase: str = "1.1") -> List[dict]:
    """
    Analyse les deltas entre baseline et traitement avec tests statistiques.

    Returns:
        Liste de dicts avec delta, IC, p-value McNemar
    """
    analysis = []

    # Trouver les modèles communs
    common_models = set(baseline_results.keys()) & set(treatment_results.keys())

    for model in sorted(common_models):
        baseline = baseline_results[model]
        treatment = treatment_results[model]

        # Aligner les questions par ID
        baseline_by_id = baseline.get_correctness_by_id()
        treatment_by_id = treatment.get_correctness_by_id()

        # Questions communes
        common_ids = set(baseline_by_id.keys()) & set(treatment_by_id.keys())

        if len(common_ids) == 0:
            print(f"  [!] Aucune question commune pour {model}")
            continue

        # Vecteurs alignés
        baseline_vec = [baseline_by_id[qid] for qid in sorted(common_ids)]
        treatment_vec = [treatment_by_id[qid] for qid in sorted(common_ids)]

        # Delta et IC bootstrap
        delta, ci_low, ci_high = bootstrap_delta_ci(baseline_vec, treatment_vec)

        # Test de McNemar
        stat, p_value, contingency = mcnemar_test(baseline_vec, treatment_vec)

        analysis.append({
            'model': model,
            'baseline_phase': baseline_phase,
            'treatment_phase': treatment_phase,
            'baseline_acc': baseline.accuracy,
            'treatment_acc': treatment.accuracy,
            'delta': delta,
            'delta_ci_low': ci_low,
            'delta_ci_high': ci_high,
            'mcnemar_stat': stat,
            'mcnemar_p': p_value,
            'improved': contingency['c_improved'],
            'degraded': contingency['b_degraded'],
            'n_common': len(common_ids)
        })

    # Appliquer correction Benjamini-Hochberg
    if analysis:
        p_values = [a['mcnemar_p'] for a in analysis]
        adjusted_p, significant = benjamini_hochberg_correction(p_values)

        for i, a in enumerate(analysis):
            a['mcnemar_p_adjusted'] = adjusted_p[i]
            a['significant'] = significant[i]

    return analysis


def compute_stability_index(results_phase11: Dict[str, ModelResults],
                           results_phase12: Dict[str, ModelResults]) -> List[dict]:
    """
    Calcule l'indice de stabilité entre phase 1.1 et 1.2.
    Stabilité = 1 - |acc_1.1 - acc_1.2| / 100
    """
    stability = []

    common_models = set(results_phase11.keys()) & set(results_phase12.keys())

    for model in sorted(common_models):
        r11 = results_phase11[model]
        r12 = results_phase12[model]

        diff = abs(r11.accuracy - r12.accuracy)
        stability_index = 1 - diff / 100

        stability.append({
            'model': model,
            'acc_1.1': r11.accuracy,
            'acc_1.2': r12.accuracy,
            'diff': diff,
            'stability_index': stability_index
        })

    return stability


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def format_accuracy_table(table: List[dict], phases: List[str]) -> str:
    """Formate le tableau d'accuracy pour affichage/publication"""
    lines = []

    # En-tête
    header = f"{'Model':<25}"
    for phase in phases:
        header += f" | Phase {phase} Acc (95% CI)"
    lines.append(header)
    lines.append("-" * len(header))

    # Données
    for row in table:
        line = f"{row['model']:<25}"
        for phase in phases:
            acc = row.get(f'phase_{phase}_acc')
            ci = row.get(f'phase_{phase}_ci')
            if acc is not None and ci is not None:
                line += f" | {acc:5.1f}% ({ci[0]:4.1f}-{ci[1]:4.1f})"
            else:
                line += f" |        N/A        "
        lines.append(line)

    return "\n".join(lines)


def format_delta_table(analysis: List[dict]) -> str:
    """Formate le tableau des deltas pour affichage/publication"""
    lines = []

    # En-tête
    header = f"{'Model':<20} | {'Baseline':>8} | {'Treatment':>9} | {'Delta':>12} | {'McNemar p':>10} | {'Adj. p':>8} | {'Sig':>3}"
    lines.append(header)
    lines.append("-" * len(header))

    # Données
    for row in analysis:
        sig_marker = "*" if row.get('significant', False) else ""
        line = (f"{row['model']:<20} | "
                f"{row['baseline_acc']:7.1f}% | "
                f"{row['treatment_acc']:8.1f}% | "
                f"{row['delta']:+5.1f}% ({row['delta_ci_low']:+5.1f},{row['delta_ci_high']:+5.1f}) | "
                f"{row['mcnemar_p']:10.4f} | "
                f"{row['mcnemar_p_adjusted']:8.4f} | "
                f"{sig_marker:>3}")
        lines.append(line)

    return "\n".join(lines)


def format_stability_table(stability: List[dict]) -> str:
    """Formate le tableau de stabilité"""
    lines = []

    header = f"{'Model':<25} | {'Acc 1.1':>8} | {'Acc 1.2':>8} | {'Diff':>6} | {'Stability':>9}"
    lines.append(header)
    lines.append("-" * len(header))

    for row in sorted(stability, key=lambda x: -x['stability_index']):
        line = (f"{row['model']:<25} | "
                f"{row['acc_1.1']:7.1f}% | "
                f"{row['acc_1.2']:7.1f}% | "
                f"{row['diff']:5.1f}% | "
                f"{row['stability_index']:8.3f}")
        lines.append(line)

    return "\n".join(lines)


def export_to_csv(data: List[dict], output_file: str):
    """Exporte les données en CSV"""
    import csv

    if not data:
        return

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

    print(f"  Exporté: {output_file}")


def export_to_latex(analysis: List[dict], output_file: str):
    """Exporte le tableau des deltas en format LaTeX"""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Effect of correction on model accuracy}",
        r"\label{tab:correction_effect}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & Baseline & Treatment & Delta & 95\% CI & McNemar $p$ & Adj. $p$ \\",
        r"\midrule",
    ]

    for row in analysis:
        sig = r"$^*$" if row.get('significant', False) else ""
        line = (f"{row['model']} & "
                f"{row['baseline_acc']:.1f}\\% & "
                f"{row['treatment_acc']:.1f}\\% & "
                f"{row['delta']:+.1f}\\% & "
                f"({row['delta_ci_low']:+.1f}, {row['delta_ci_high']:+.1f}) & "
                f"{row['mcnemar_p']:.4f} & "
                f"{row['mcnemar_p_adjusted']:.4f}{sig} \\\\")
        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    print(f"  Exporté: {output_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyse statistique des résultats d'expérience"
    )
    parser.add_argument(
        "--phase1.0",
        type=str,
        help="Pattern glob pour les fichiers Phase 1.0 (ex: resultats_phase1.0/*.json)"
    )
    parser.add_argument(
        "--phase1.1",
        type=str,
        help="Pattern glob pour les fichiers Phase 1.1"
    )
    parser.add_argument(
        "--phase1.2",
        type=str,
        help="Pattern glob pour les fichiers Phase 1.2"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Répertoire de sortie pour les exports"
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Exporter les résultats en CSV"
    )
    parser.add_argument(
        "--export-latex",
        action="store_true",
        help="Exporter les tableaux en LaTeX"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Affichage détaillé"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("ANALYSE STATISTIQUE DES RÉSULTATS")
    print("=" * 80)

    # Charger les résultats
    results_by_phase = {}

    phase_args = {
        '1.0': getattr(args, 'phase1.0', None),
        '1.1': getattr(args, 'phase1.1', None),
        '1.2': getattr(args, 'phase1.2', None),
    }

    for phase, pattern in phase_args.items():
        if pattern:
            print(f"\nChargement Phase {phase}: {pattern}")
            results = load_phase_results(pattern)
            if results:
                results_by_phase[phase] = results
                print(f"  Modèles chargés: {', '.join(results.keys())}")
            else:
                print(f"  [!] Aucun résultat trouvé")

    if not results_by_phase:
        print("\n[ERREUR] Aucune donnée chargée. Vérifiez les chemins.")
        print("\nExemples d'utilisation:")
        print('  python statistical_analysis.py --phase1.0 "resultats_phase1.0/*_judged_*.json"')
        print('  python statistical_analysis.py --phase1.0 "resultats_phase1.0_datasetB/*_judged_*.json" --phase1.1 "resultats_phase1.1_datasetB/*_judged_*.json"')
        return 1

    # ==========================================================================
    # 1. TABLEAU D'ACCURACY
    # ==========================================================================
    print("\n" + "=" * 80)
    print("1. TABLEAU D'ACCURACY AVEC INTERVALLES DE CONFIANCE WILSON (95%)")
    print("=" * 80)

    phases = sorted(results_by_phase.keys())
    accuracy_table = compute_accuracy_table(results_by_phase)
    print("\n" + format_accuracy_table(accuracy_table, phases))

    if args.export_csv:
        export_to_csv(accuracy_table, Path(args.output_dir) / "accuracy_table.csv")

    # ==========================================================================
    # 2. ANALYSE DES DELTAS (si baseline + au moins un traitement)
    # ==========================================================================
    if '1.0' in results_by_phase:
        baseline = results_by_phase['1.0']

        for treatment_phase in ['1.1', '1.2']:
            if treatment_phase in results_by_phase:
                print("\n" + "=" * 80)
                print(f"2. ANALYSE DELTA: Phase 1.0 vs Phase {treatment_phase}")
                print("   (Test de McNemar + Bootstrap IC + Correction Benjamini-Hochberg)")
                print("=" * 80)

                treatment = results_by_phase[treatment_phase]
                delta_analysis = compute_delta_analysis(baseline, treatment, "1.0", treatment_phase)

                print("\n" + format_delta_table(delta_analysis))

                # Résumé
                n_improved = sum(1 for a in delta_analysis if a['delta'] > 0)
                n_degraded = sum(1 for a in delta_analysis if a['delta'] < 0)
                n_significant = sum(1 for a in delta_analysis if a.get('significant', False))

                print(f"\nRésumé:")
                print(f"  Modèles améliorés: {n_improved}")
                print(f"  Modèles dégradés: {n_degraded}")
                print(f"  Résultats significatifs (p_adj < 0.05): {n_significant}")

                if args.export_csv:
                    export_to_csv(delta_analysis,
                                 Path(args.output_dir) / f"delta_analysis_1.0_vs_{treatment_phase}.csv")

                if args.export_latex:
                    export_to_latex(delta_analysis,
                                   Path(args.output_dir) / f"delta_table_1.0_vs_{treatment_phase}.tex")

    # ==========================================================================
    # 3. ANALYSE DE STABILITÉ (si phase 1.1 et 1.2)
    # ==========================================================================
    if '1.1' in results_by_phase and '1.2' in results_by_phase:
        print("\n" + "=" * 80)
        print("3. ANALYSE DE STABILITÉ (Phase 1.1 vs Phase 1.2)")
        print("   Stabilité = 1 - |acc_1.1 - acc_1.2| / 100")
        print("=" * 80)

        stability = compute_stability_index(results_by_phase['1.1'], results_by_phase['1.2'])
        print("\n" + format_stability_table(stability))

        if args.export_csv:
            export_to_csv(stability, Path(args.output_dir) / "stability_analysis.csv")

    print("\n" + "=" * 80)
    print("ANALYSE TERMINÉE")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
