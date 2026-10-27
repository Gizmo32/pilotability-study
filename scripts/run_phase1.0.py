"""
Script d'exécution de l'Expérience 1.0 - Groupe témoin (baseline)
Collecte des réponses brutes des LLMs sans mécanismes correctifs

Cette expérience établit le groupe témoin avec les performances de base des modèles.
Les réponses seront ensuite évaluées par judge_results.py.

Usage:
    python run_phase1.0.py --model mistral:latest
    python run_phase1.0.py --model llama3:latest --verbose
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from client import LLMClient


def load_dataset(dataset_path: str) -> List[Dict]:
    """Charge le dataset JSONL"""
    questions = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def run_baseline(
    model_name: str,
    dataset_path: str = "dataset_a.jsonl",
    output_dir: str = "resultats_phase1.0",
    temperature: float = 0.1,
    verbose: bool = False
) -> Dict:
    """
    Exécute l'expérience 1.0 : collecte des réponses brutes (baseline)

    Args:
        model_name: Nom du modèle Ollama à tester
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

    print(f"🔬 EXPÉRIENCE 1.0 - GROUPE TÉMOIN (BASELINE)")
    print(f"📊 Modèle : {model_name}")
    print(f"📁 Dataset : {dataset_path}")
    print(f"🌡️  Température : {temperature}")
    print(f"📂 Sortie : {output_dir}")
    print("-" * 60)

    # Charger le dataset
    questions = load_dataset(dataset_path)
    print(f"✓ {len(questions)} questions chargées\n")

    # Préparer les résultats
    results = {
        "metadata": {
            "experiment": "1.0_baseline",
            "model": model_name,
            "dataset": dataset_path,
            "temperature": temperature,
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(questions),
            "protocol_step": "ÉTAPE 0 - Réponse brute"
        },
        "questions": []
    }

    # Prompt standardisé selon le protocole ÉTAPE 0
    prompt_template = """Tu es un agent d'évaluation de benchmark. Ta mission est de répondre aux questions de manière extrêmement concise.

DIRECTIVES :
1. Réponds uniquement par la valeur attendue (ex: "Vrai", "Faux", "0", ou un mot spécifique).
2. Ne donne AUCUNE explication, ne justifie pas ton choix.
3. Ne salue pas et ne conclus pas tes messages.
4. Si la consigne de la question demande un format spécifique, respecte-le à la lettre.
5. Si tu ne sais pas, choisis la réponse "Je ne sais pas".

Question : {question}

Réponse :"""

    # Traiter chaque question
    for i, q in enumerate(questions, 1):
        question_id = q["id"]
        domain = q["domain"]
        subdomain = q["subdomain"]
        question_text = q["question"]
        expected_answer = q["expected_answer"]

        # Générer le prompt
        prompt = prompt_template.format(question=question_text)

        # Interroger le LLM
        try:
            response = client.generate(prompt)
            llm_answer = response.text.strip()

            # Enregistrer le résultat (sans évaluation - sera fait par le juge)
            question_result = {
                "id": question_id,
                "domain": domain,
                "subdomain": subdomain,
                "question": question_text,
                "expected_answer": expected_answer,
                "llm_answer": llm_answer,
                "duration_ms": response.duration_ms,
                "tokens": {
                    "prompt": response.prompt_tokens,
                    "completion": response.completion_tokens,
                    "total": response.total_tokens
                }
            }
            results["questions"].append(question_result)

            # Affichage
            if verbose:
                print(f"[{i}/{len(questions)}] {question_id} - {domain}/{subdomain}")
                print(f"  Q: {question_text[:60]}...")
                print(f"  Attendu: {expected_answer}")
                print(f"  Réponse: {llm_answer[:60]}...")
                print()
            else:
                print(f"[{i}/{len(questions)}] {question_id} - {domain}/{subdomain}", end="\r")

        except Exception as e:
            print(f"\n⚠️ Erreur sur la question {question_id}: {e}")
            continue

    print("\n")

    # Statistiques de base (sans évaluation)
    domains_count = {}
    subdomains_count = {}
    for q in results["questions"]:
        domain = q["domain"]
        subdomain = f"{q['domain']}/{q['subdomain']}"

        domains_count[domain] = domains_count.get(domain, 0) + 1
        subdomains_count[subdomain] = subdomains_count.get(subdomain, 0) + 1

    results["statistics"] = {
        "domains_count": domains_count,
        "subdomains_count": subdomains_count,
        "completed_questions": len(results["questions"]),
        "note": "Les réponses seront évaluées par judge_results.py"
    }

    # Sauvegarder les résultats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"results_{model_name.replace(':', '_')}_{timestamp}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Afficher le résumé
    print("=" * 60)
    print("📊 RÉSULTATS - EXPÉRIENCE 1.0")
    print("=" * 60)
    print(f"\n✅ Réponses collectées : {len(results['questions'])}/{len(questions)}")
    print("\n📈 Distribution par domaine :")
    for domain, count in sorted(domains_count.items()):
        print(f"  {domain:12s} : {count:3d} questions")

    print(f"\n💾 Résultats sauvegardés : {output_file}")
    print(f"\n🔍 Prochaine étape : Évaluation avec judge_results.py")
    print(f"   python judge_results.py {output_file} --judge mistral:latest")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Expérience 1.0 : Collecte des réponses brutes (baseline)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistral:latest",
        help="Nom du modèle Ollama à tester"
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
        default="resultats_phase1.0",
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

    # Exécuter l'expérience 1.0
    try:
        run_baseline(
            model_name=args.model,
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
