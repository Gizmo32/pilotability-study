"""
Script pour tester automatiquement tous les modèles - Expérience 1.2
Relecture dirigée avec points d'attention pour tous les modèles (ÉTAPE 2 du protocole)

Les résultats seront ensuite évalués par judge_results.py

Usage:
    python run_all_models1.2.py
    python run_all_models1.2.py --baseline-dir resultats_phase1.0
    python run_all_models1.2.py --models mistral:latest llama3:latest
"""

import subprocess
import sys
import glob
from pathlib import Path
from datetime import datetime
from client import LLMClient

# Liste des modèles à tester selon le protocole
MODELS_TO_TEST = [
    "mistral:latest",
    "qwen2.5:latest",
    "llama3:latest",
    "gemma3:4b",
    "nemotron-mini:4b",
    "phi3:mini"
]


def check_model_available(model_name: str) -> bool:
    """Vérifie si un modèle est disponible dans Ollama"""
    try:
        client = LLMClient()
        available_models = client.list_models()
        return model_name in available_models
    except Exception:
        return False


def find_baseline_file(model_name: str, baseline_dir: str) -> str:
    """
    Trouve le fichier baseline correspondant au modèle dans le répertoire spécifié.

    Returns:
        Chemin vers le fichier baseline ou None si non trouvé
    """
    # Normaliser le nom du modèle pour le pattern (remplacer : par _)
    model_pattern = model_name.replace(":", "_").replace(".", "_")

    # Chercher les fichiers correspondants
    pattern = f"{baseline_dir}/results_{model_pattern}*.json"
    matching_files = glob.glob(pattern)

    if matching_files:
        # Prendre le plus récent (dernier dans l'ordre alphabétique avec timestamp)
        return sorted(matching_files)[-1]

    # Essayer avec un pattern plus flexible
    pattern_flexible = f"{baseline_dir}/results_*{model_pattern.split('_')[0]}*.json"
    matching_files = glob.glob(pattern_flexible)

    if matching_files:
        return sorted(matching_files)[-1]

    return None


def run_directed_review_for_model(
    model_name: str,
    dataset: str = "dataset_a.jsonl",
    output_dir: str = "resultats_phase1.2_relecture_dirigee",
    baseline_file: str = None,
    verbose: bool = False
) -> tuple:
    """
    Exécute run_phase1.2.py pour un modèle donné

    Returns:
        (success: bool, output_dir: str)
    """
    cmd = [
        sys.executable,
        "run_phase1.2.py",
        "--model", model_name,
        "--dataset", dataset,
        "--output-dir", output_dir
    ]

    if baseline_file:
        cmd.extend(["--baseline", baseline_file])

    if verbose:
        cmd.append("--verbose")

    print(f"\n{'='*60}")
    print(f"🚀 Test du modèle : {model_name}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            cwd=Path(__file__).parent
        )

        if result.returncode == 0:
            print(f"✅ Test terminé avec succès : {model_name}")
            return True, output_dir
        else:
            print(f"❌ Échec du test : {model_name}")
            return False, None

    except Exception as e:
        print(f"❌ Erreur lors de l'exécution pour {model_name}: {e}")
        return False, None


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Expérience 1.2 : Relecture dirigée pour tous les modèles"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mode verbose pour chaque test"
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Ne pas vérifier la disponibilité des modèles avant de commencer"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Liste des modèles à tester (remplace la liste par défaut)"
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
        "--baseline-dir",
        type=str,
        default=None,
        help="Répertoire contenant les résultats de la phase 1.0 (pour réutiliser comme initial_answer)"
    )
    parser.add_argument(
        "--judge",
        type=str,
        default="mistral:latest",
        help="Modèle à utiliser comme juge après la collecte (défaut: mistral:latest)"
    )
    parser.add_argument(
        "--auto-judge",
        action="store_true",
        help="Lancer automatiquement judge_results.py après la collecte"
    )

    args = parser.parse_args()

    # Utiliser la liste personnalisée si fournie
    models = args.models if args.models else MODELS_TO_TEST

    print("=" * 60)
    print("🔬 EXPÉRIENCE 1.2 - RELECTURE DIRIGÉE")
    print("=" * 60)
    print(f"\n📋 Modèles à tester ({len(models)}) :")
    for model in models:
        print(f"  • {model}")
    print(f"\n📂 Dossier de sortie : {args.output_dir}")
    if args.baseline_dir:
        print(f"📥 Baseline depuis : {args.baseline_dir} (mode optimisé)")
    else:
        print(f"⚠️  Pas de baseline : génération initial_answer + final_answer")
    print(f"👨‍⚖️  Modèle juge : {args.judge}")
    print(f"\n📍 Mécanisme : Relecture dirigée avec points d'attention par domaine/sous-domaine")

    # Vérifier la disponibilité des modèles
    if not args.skip_check:
        print("\n🔍 Vérification de la disponibilité des modèles...")

        try:
            client = LLMClient()
            if not client.is_available():
                print("❌ Ollama n'est pas accessible. Assurez-vous qu'il est démarré.")
                print("   Commande : ollama serve")
                return 1

            available_models = client.list_models()
            print(f"✅ Ollama accessible ({len(available_models)} modèles disponibles)")

            missing_models = []
            for model in models:
                if model not in available_models:
                    missing_models.append(model)

            if missing_models:
                print(f"\n⚠️  Modèles manquants ({len(missing_models)}) :")
                for model in missing_models:
                    print(f"  ❌ {model}")

                print("\n💡 Pour télécharger les modèles manquants :")
                for model in missing_models:
                    print(f"   ollama pull {model}")

                response = input("\nContinuer quand même avec les modèles disponibles ? (o/N) : ")
                if response.lower() not in ['o', 'oui', 'y', 'yes']:
                    print("Annulation.")
                    return 1

                # Filtrer pour ne garder que les modèles disponibles
                models = [m for m in models if m not in missing_models]
                if not models:
                    print("❌ Aucun modèle disponible à tester.")
                    return 1

                print(f"\n✅ Test de {len(models)} modèle(s) disponible(s)")

        except Exception as e:
            print(f"⚠️  Erreur lors de la vérification : {e}")
            response = input("\nContinuer quand même ? (o/N) : ")
            if response.lower() not in ['o', 'oui', 'y', 'yes']:
                return 1

    # Créer le répertoire de sortie
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    print(f"\n✅ Répertoire de sortie créé : {output_path.absolute()}")

    # Exécuter les tests
    start_time = datetime.now()
    results = []

    print(f"\n{'='*60}")
    print(f"🎯 Début de l'expérience à {start_time.strftime('%H:%M:%S')}")
    print(f"{'='*60}")

    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Modèle : {model}")

        # Trouver le fichier baseline si --baseline-dir est spécifié
        baseline_file = None
        if args.baseline_dir:
            baseline_file = find_baseline_file(model, args.baseline_dir)
            if baseline_file:
                print(f"  📥 Baseline trouvé : {Path(baseline_file).name}")
            else:
                print(f"  ⚠️  Baseline non trouvé pour {model}, génération sans baseline")

        success, output_dir = run_directed_review_for_model(
            model,
            dataset=args.dataset,
            output_dir=args.output_dir,
            baseline_file=baseline_file,
            verbose=args.verbose
        )

        results.append({
            "model": model,
            "success": success,
            "output_dir": output_dir
        })

    # Résumé
    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'='*60}")
    print(f"📊 RÉSUMÉ DE L'EXPÉRIENCE")
    print(f"{'='*60}")
    print(f"\n⏱️  Durée totale : {duration}")
    print(f"\n✅ Tests réussis : {sum(1 for r in results if r['success'])}/{len(results)}")

    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]

    if successful_results:
        print("\n✅ Modèles testés avec succès :")
        for r in successful_results:
            print(f"  • {r['model']}")

    if failed_results:
        print("\n❌ Modèles en échec :")
        for r in failed_results:
            print(f"  • {r['model']}")

    # Lister les fichiers de résultats générés
    if successful_results:
        print(f"\n📁 Fichiers de résultats dans {args.output_dir}/ :")
        result_files = sorted(output_path.glob("results_*.json"))
        for result_file in result_files:
            size_kb = result_file.stat().st_size / 1024
            print(f"  • {result_file.name} ({size_kb:.1f} KB)")

    # Lancer le juge automatiquement si demandé
    if args.auto_judge and successful_results:
        print(f"\n{'='*60}")
        print("👨‍⚖️  Lancement automatique du juge...")
        print(f"{'='*60}")

        result_files = sorted(output_path.glob("results_*.json"))
        output_files = [str(f) for f in result_files]

        if output_files:
            try:
                cmd = [
                    sys.executable,
                    "judge_results.py"
                ] + output_files + [
                    "--judge", args.judge,
                    "--compare"
                ]

                result = subprocess.run(cmd, cwd=Path(__file__).parent)

                if result.returncode == 0:
                    print("\n✅ Jugement terminé avec succès")
            except Exception as e:
                print(f"⚠️  Erreur lors du jugement : {e}")
    else:
        # Instructions pour lancer le juge manuellement
        if successful_results:
            print(f"\n{'='*60}")
            print("🔍 PROCHAINE ÉTAPE : ÉVALUATION PAR LE JUGE")
            print(f"{'='*60}")
            result_files = sorted(output_path.glob("results_*.json"))
            if result_files:
                files_str = " ".join([f'"{f}"' for f in result_files])
                print(f"\nCommande pour évaluer tous les résultats :")
                print(f"python judge_results.py {files_str} --judge {args.judge} --compare")

    print(f"\n{'='*60}")
    print("✅ Terminé !")
    print(f"{'='*60}\n")

    return 0 if all(r['success'] for r in results) else 1


if __name__ == "__main__":
    exit(main())
