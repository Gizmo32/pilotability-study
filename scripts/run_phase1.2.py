"""
Script d'exécution de l'Expérience 1.2 - Relecture Dirigée
Implémente l'ÉTAPE 2 du protocole : relecture avec points d'attention par domaine/sous-domaine

Cette expérience mesure l'impact de la relecture dirigée avec guidance spécifique.
Les résultats seront évalués par judge_results.py.

Usage:
    python run_phase1.2.py --model mistral:latest
    python run_phase1.2.py --model llama3:latest --verbose
"""

import json
import argparse
import glob
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from client import LLMClient


# Points d'attention par domaine/sous-domaine selon le protocole
ATTENTION_POINTS = {
    "logic": {
        "syllogism": [
            "La structure du syllogisme est-elle valide ?",
            "Les termes moyens sont-ils correctement distribués ?",
            "Y a-t-il une erreur de distribution logique ?"
        ],
        "negation": [
            "La négation est-elle appliquée correctement ?",
            "La portée de la négation est-elle bien comprise ?",
            "Les quantificateurs sont-ils inversés correctement ?"
        ],
        "transitivity": [
            "Les relations d'ordre sont-elles respectées ?",
            "La transitivité est-elle correctement appliquée ?",
            "Les positions relatives sont-elles cohérentes ?"
        ],
        "implication": [
            "La structure de l'implication (si...alors) est-elle bien comprise ?",
            "Les règles du modus ponens et modus tollens sont-elles appliquées ?",
            "Y a-t-il confusion entre implication et causalité ?"
        ],
        "set_theory": [
            "Les relations d'inclusion sont-elles correctes ?",
            "Les opérations ensemblistes sont-elles valides ?",
            "Les définitions d'ensembles sont-elles respectées ?"
        ],
        "deduction": [
            "Le raisonnement déductif est-il valide ?",
            "Les prémisses conduisent-elles logiquement à la conclusion ?",
            "Y a-t-il des sauts logiques ?"
        ],
        "causality": [
            "La distinction entre corrélation et causalité est-elle claire ?",
            "Les liens de cause à effet sont-ils corrects ?",
            "Y a-t-il confusion entre causes et conséquences ?"
        ],
        "contradiction": [
            "Le principe de non-contradiction est-il respecté ?",
            "Y a-t-il des affirmations incompatibles ?",
            "Les lois logiques sont-elles appliquées ?"
        ],
        "binary": [
            "La logique binaire est-elle correctement appliquée ?",
            "Les valeurs vrai/faux sont-elles bien comprises ?",
            "Les opérations logiques sont-elles exactes ?"
        ],
        "temporal": [
            "L'ordre temporel est-il cohérent ?",
            "Les relations avant/après sont-elles correctes ?",
            "Les durées sont-elles bien comprises ?"
        ],
        "sorting": [
            "L'ordre de tri est-il correct ?",
            "Les critères de comparaison sont-ils valides ?",
            "Les positions relatives sont-elles cohérentes ?"
        ]
    },
    "math": {
        "arithmetic": [
            "Les calculs sont-ils exacts ?",
            "Les opérations sont-elles dans le bon ordre ?",
            "Les pourcentages sont-ils correctement appliqués ?"
        ],
        "comparison": [
            "Les comparaisons numériques sont-elles exactes ?",
            "Les nombres décimaux sont-ils correctement interprétés ?",
            "Les ordres de grandeur sont-ils respectés ?"
        ],
        "fractions": [
            "Les conversions entre fractions et décimales sont-elles correctes ?",
            "Les comparaisons de fractions sont-elles exactes ?",
            "Les simplifications sont-elles valides ?"
        ],
        "primes": [
            "La définition d'un nombre premier est-elle respectée ?",
            "Les diviseurs sont-ils correctement identifiés ?",
            "Les cas particuliers (1, 2) sont-ils traités correctement ?"
        ],
        "geometry": [
            "Les propriétés géométriques sont-elles correctes ?",
            "Les formules sont-elles appliquées correctement ?",
            "Les relations spatiales sont-elles valides ?"
        ],
        "probability": [
            "Les calculs de probabilité sont-ils corrects ?",
            "Les événements sont-ils correctement définis ?",
            "Les proportions sont-elles exactes ?"
        ],
        "logic": [
            "Le raisonnement logique mathématique est-il valide ?",
            "Les proportions sont-elles respectées ?",
            "Les relations sont-elles correctement établies ?"
        ],
        "units": [
            "Les unités de mesure sont-elles correctes ?",
            "Les conversions sont-elles exactes ?",
            "Les ordres de grandeur sont-ils respectés ?"
        ]
    },
    "sci": {
        "physics": [
            "Les lois physiques sont-elles correctement appliquées ?",
            "Les ordres de grandeur sont-ils respectés ?",
            "Les vitesses/forces sont-elles cohérentes ?"
        ],
        "biology": [
            "Les processus biologiques sont-ils correctement décrits ?",
            "Les classifications sont-elles exactes ?",
            "Les propriétés des organismes sont-elles correctes ?"
        ],
        "chemistry": [
            "Les propriétés chimiques sont-elles exactes ?",
            "Les formules et compositions sont-elles correctes ?",
            "Les réactions sont-elles valides ?"
        ],
        "astronomy": [
            "Les faits astronomiques sont-ils exacts ?",
            "Les positions et distances sont-elles correctes ?",
            "Les phénomènes célestes sont-ils bien compris ?"
        ]
    },
    "gen": {
        "geography": [
            "Les localisations sont-elles exactes ?",
            "Les noms de lieux sont-ils corrects ?",
            "Les positions relatives sont-elles valides ?"
        ],
        "history": [
            "Les dates historiques sont-elles exactes ?",
            "Les faits historiques sont-ils corrects ?",
            "Les personnages sont-ils correctement identifiés ?"
        ],
        "culture": [
            "Les faits culturels sont-ils exacts ?",
            "Les traditions sont-elles correctement décrites ?",
            "Les références sont-elles valides ?"
        ],
        "language": [
            "Les règles linguistiques sont-elles correctes ?",
            "Les définitions sont-elles exactes ?",
            "Les usages sont-ils appropriés ?"
        ],
        "logic_puzzle": [
            "Toutes les informations de l'énoncé sont-elles prises en compte ?",
            "Le raisonnement logique est-il correct ?",
            "Les indices sont-ils correctement interprétés ?"
        ],
        "instruction": [
            "L'instruction est-elle suivie exactement ?",
            "La contrainte est-elle respectée à la lettre ?",
            "La réponse correspond-elle exactement à ce qui est demandé ?"
        ]
    }
}


def load_dataset(dataset_path: str) -> List[Dict]:
    """Charge le dataset JSONL"""
    questions = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def get_attention_points(domain: str, subdomain: str) -> str:
    """Récupère les points d'attention pour un domaine/sous-domaine"""
    if domain in ATTENTION_POINTS and subdomain in ATTENTION_POINTS[domain]:
        points = ATTENTION_POINTS[domain][subdomain]
        return "\n".join(f"- {point}" for point in points)
    return ""


def run_directed_review(
    model_name: str,
    baseline_file: str = None,
    dataset_path: str = "dataset_a.jsonl",
    output_dir: str = "resultats_phase1.2_relecture_dirigee",
    temperature: float = 0.1,
    verbose: bool = False
) -> Dict:
    """
    Exécute l'expérience 1.2 : relecture dirigée avec points d'attention

    Args:
        model_name: Nom du modèle Ollama à tester
        baseline_file: Fichier JSON de la phase 1.0 (optionnel, réutilise llm_answer comme initial_answer)
        dataset_path: Chemin vers le dataset JSONL
        output_dir: Répertoire pour sauvegarder les résultats
        temperature: Température pour la génération (0.1 par défaut selon protocole)
        verbose: Afficher les détails pour chaque question

    Returns:
        Dictionnaire avec les résultats
    """
    # Créer le répertoire de sortie
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Initialiser le client
    client = LLMClient(
        model=model_name,
        temperature=temperature,
        max_tokens=200
    )

    # Vérifier la disponibilité
    if not client.is_available():
        raise ConnectionError("Ollama n'est pas accessible. Assurez-vous qu'il est démarré.")

    print(f"🔬 EXPÉRIENCE 1.2 - RELECTURE DIRIGÉE")
    print(f"📊 Modèle : {model_name}")
    print(f"📁 Dataset : {dataset_path}")
    print(f"🌡️  Température : {temperature}")
    print(f"📂 Sortie : {output_dir}")
    print("-" * 60)

    # Charger le fichier baseline si fourni
    baseline_answers = {}
    if baseline_file:
        # Résoudre les wildcards (*, ?) dans le chemin
        matching_files = glob.glob(baseline_file)
        if not matching_files:
            raise FileNotFoundError(f"Aucun fichier trouvé correspondant à : {baseline_file}")
        if len(matching_files) > 1:
            raise ValueError(f"Plusieurs fichiers trouvés ({len(matching_files)}). Veuillez spécifier un fichier unique.")

        baseline_file_resolved = matching_files[0]
        print(f"📥 Chargement baseline : {baseline_file_resolved}")

        with open(baseline_file_resolved, 'r', encoding='utf-8') as f:
            baseline_data = json.load(f)
        for q in baseline_data["questions"]:
            baseline_answers[q["id"]] = q.get("llm_answer", "")
        print(f"✓ {len(baseline_answers)} réponses baseline chargées (réutilisées comme initial_answer)")
        print("⚡ Mode optimisé : génération de final_answer uniquement\n")
    else:
        print("⚠️  Pas de baseline fourni : génération de initial_answer + final_answer\n")

    # Charger le dataset
    questions = load_dataset(dataset_path)
    print(f"✓ {len(questions)} questions chargées\n")

    # Préparer les résultats
    results = {
        "metadata": {
            "experiment": "1.2_directed_review",
            "model": model_name,
            "dataset": dataset_path,
            "temperature": temperature,
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(questions),
            "protocol_step": "ÉTAPE 2 - Relecture dirigée avec points d'attention"
        },
        "questions": [],
        "statistics": {
            "total_questions": len(questions),
            "answers_changed": 0,
            "answers_unchanged": 0
        }
    }

    # Prompt standardisé selon le protocole ÉTAPE 0 (réponse initiale)
    initial_prompt_template = """Tu es un agent d'évaluation de benchmark. Ta mission est de répondre aux questions de manière extrêmement concise.

DIRECTIVES :
1. Réponds uniquement par la valeur attendue (ex: "Vrai", "Faux", "0", ou un mot spécifique).
2. Ne donne AUCUNE explication, ne justifie pas ton choix.
3. Ne salue pas et ne conclus pas tes messages.
4. Si la consigne de la question demande un format spécifique, respecte-le à la lettre.
5. Si tu ne sais pas, choisis la réponse "Je ne sais pas".

Question : {question}

Réponse :"""

    # Prompt de relecture dirigée selon le protocole v3.0 - Approche "Stateless" (Apatride)
    # Corrigé pour éliminer la confusion de rôle (langage évaluatif supprimé)
    directed_review_prompt_template = """Question : {question}

CONTEXTE : Une tentative de réponse a été proposée : "{initial_answer}"

INDICES UTILES pour cette question :
{attention_points}

CONSIGNE : Réponds à la question ci-dessus en utilisant les indices fournis.
- Réponds UNIQUEMENT par la valeur attendue (ex: Vrai, Faux, 42, Paris, etc.)
- Ne donne AUCUNE explication, ne justifie pas
- Ne commente pas la tentative précédente ni les indices
- Donne simplement TA réponse à la question

Réponse :"""

    # Traiter chaque question
    for i, q in enumerate(questions, 1):
        question_id = q["id"]
        domain = q["domain"]
        subdomain = q["subdomain"]
        question_text = q["question"]
        expected_answer = q["expected_answer"]

        try:
            # Mode optimisé : réutiliser baseline ou générer initial_answer
            if baseline_file and question_id in baseline_answers:
                # OPTIMISÉ : Réutiliser la réponse 1.0 comme initial_answer
                initial_answer = baseline_answers[question_id]
                response_initial = None  # Pas de génération
            else:
                # STANDARD : Générer initial_answer
                initial_prompt = initial_prompt_template.format(question=question_text)
                response_initial = client.generate(initial_prompt)
                initial_answer = response_initial.text.strip()

            # Récupérer les points d'attention pour ce domaine/sous-domaine
            attention_points = get_attention_points(domain, subdomain)

            # ÉTAPE : Relecture dirigée (Approche Stateless)
            directed_prompt = directed_review_prompt_template.format(
                question=question_text,
                initial_answer=initial_answer,
                attention_points=attention_points
            )
            response_review = client.generate(directed_prompt)
            final_answer = response_review.text.strip()

            # Détecter si la réponse a changé
            changed = initial_answer.lower().strip() != final_answer.lower().strip()

            # Enregistrer le résultat
            question_result = {
                "id": question_id,
                "domain": domain,
                "subdomain": subdomain,
                "question": question_text,
                "expected_answer": expected_answer,
                "initial_answer": initial_answer,
                "final_answer": final_answer,
                "attention_points": attention_points,
                "changed": changed,
                "llm_answer": final_answer,  # Pour compatibilité avec judge_results.py
                "duration_ms": {
                    "initial": response_initial.duration_ms if response_initial else 0,
                    "review": response_review.duration_ms,
                    "total": (response_initial.duration_ms if response_initial else 0) + response_review.duration_ms
                },
                "tokens": {
                    "initial": {
                        "prompt": response_initial.prompt_tokens if response_initial else 0,
                        "completion": response_initial.completion_tokens if response_initial else 0,
                        "total": response_initial.total_tokens if response_initial else 0
                    },
                    "review": {
                        "prompt": response_review.prompt_tokens,
                        "completion": response_review.completion_tokens,
                        "total": response_review.total_tokens
                    },
                    "total": (response_initial.total_tokens if response_initial else 0) + response_review.total_tokens
                }
            }
            results["questions"].append(question_result)

            # Mettre à jour les statistiques
            if changed:
                results["statistics"]["answers_changed"] += 1
            else:
                results["statistics"]["answers_unchanged"] += 1

            # Affichage
            if verbose:
                print(f"[{i}/{len(questions)}] {question_id} - {domain}/{subdomain}")
                print(f"  Q: {question_text[:60]}...")
                print(f"  Attendu: {expected_answer}")
                print(f"  Initial: {initial_answer[:60]}...")
                print(f"  Final: {final_answer[:60]}...")
                print(f"  Changé: {'OUI' if changed else 'NON'}")
                print()
            else:
                status = "📍" if changed else "="
                print(f"{status} [{i}/{len(questions)}] {question_id} - {domain}/{subdomain}", end="\r")

        except Exception as e:
            print(f"\n⚠️ Erreur sur la question {question_id}: {e}")
            continue

    print("\n")

    # Calculer les taux de changement
    change_rate = (
        results["statistics"]["answers_changed"] / results["statistics"]["total_questions"]
        if results["statistics"]["total_questions"] > 0 else 0
    )

    results["statistics"]["change_rate"] = change_rate

    # Statistiques par domaine
    domains_count = {}
    domains_changed = {}
    for q in results["questions"]:
        domain = q["domain"]
        if domain not in domains_count:
            domains_count[domain] = 0
            domains_changed[domain] = 0
        domains_count[domain] += 1
        if q["changed"]:
            domains_changed[domain] += 1

    results["statistics"]["by_domain"] = {
        domain: {
            "total": domains_count[domain],
            "changed": domains_changed[domain],
            "change_rate": domains_changed[domain] / domains_count[domain] if domains_count[domain] > 0 else 0
        }
        for domain in domains_count
    }

    # Sauvegarder les résultats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"results_{model_name.replace(':', '_')}_{timestamp}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Afficher le résumé
    print("=" * 60)
    print("📊 RÉSULTATS - EXPÉRIENCE 1.2")
    print("=" * 60)
    print(f"\n✅ Questions traitées : {len(results['questions'])}/{len(questions)}")
    print(f"\n📝 Changements après relecture dirigée :")
    print(f"  Réponses modifiées : {results['statistics']['answers_changed']} ({change_rate:.1%})")
    print(f"  Réponses confirmées : {results['statistics']['answers_unchanged']} ({1-change_rate:.1%})")

    print("\n📈 Taux de changement par domaine :")
    for domain, stats in sorted(results["statistics"]["by_domain"].items()):
        rate = stats["change_rate"]
        bar_length = int(rate * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"  {domain:12s} {bar} {rate:5.1%} ({stats['changed']}/{stats['total']})")

    print(f"\n💾 Résultats sauvegardés : {output_file}")
    print(f"\n🔍 Prochaine étape : Évaluation avec judge_results.py")
    print(f"   python judge_results.py {output_file} --judge mistral:latest")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Expérience 1.2 : Relecture dirigée avec points d'attention (ÉTAPE 2)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistral:latest",
        help="Nom du modèle Ollama à tester"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Fichier JSON de la phase 1.0 (optionnel, pour réutiliser llm_answer comme initial_answer et diviser le temps par 2)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset_a.jsonl",
        help="Chemin vers le dataset JSONL"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="resultats_phase1.2_relecture_dirigee",
        help="Répertoire pour sauvegarder les résultats"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Température pour la génération (0.1 par défaut selon protocole)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Afficher les détails pour chaque question"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Lister les modèles Ollama disponibles"
    )

    args = parser.parse_args()

    # Lister les modèles si demandé
    if args.list_models:
        client = LLMClient()
        try:
            models = client.list_models()
            print("📋 Modèles Ollama disponibles :")
            for model in models:
                print(f"  - {model}")
        except Exception as e:
            print(f"⚠️ Erreur lors de la récupération des modèles : {e}")
        return

    # Exécuter l'expérience 1.2
    try:
        run_directed_review(
            model_name=args.model,
            baseline_file=args.baseline,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            temperature=args.temperature,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
