"""
Module de jugement standardisé

Responsabilité unique : Comparer expected_answer et la réponse finale

Le juge se concentre UNIQUEMENT sur la comparaison de deux réponses :
- expected_answer (valeur attendue)
- Réponse finale (llm_answer pour phase 1.0, final_answer pour phases 1.1/1.2)

Le juge :
- NE voit PAS la question (évite la confusion)
- NE juge PAS initial_answer (conservé à titre indicatif seulement)
- Ignore la casse (Vrai = vrai = VRAI)
- Ignore la ponctuation
- Reconnaît les équivalences (6 = six, True = Vrai, etc.)
- Traite tous les fichiers de façon uniforme (peu importe la phase)

Comportement standardisé :
- Phase 1.0 (sans relecture) : Juge llm_answer
- Phase 1.1/1.2 (avec relecture) : Juge UNIQUEMENT final_answer
  → initial_answer est conservé pour analyser le changement mais n'est PAS jugé
  → Réduit la variabilité du juge (moitié moins de jugements)

Usage:
    python judge_results.py results_*.json --judge mistral:latest
    python judge_results.py results_*.json --judge mistral:latest --verbose
"""

import json
import argparse
import sys
import glob
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from client import LLMClient

# Fix encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass


# =============================================================================
# DICTIONNAIRES D'ÉQUIVALENCES POUR SIMPLE_COMPARE
# =============================================================================

# Nombres en lettres -> chiffres
NUMBER_WORDS = {
    'zero': '0', 'zéro': '0',
    'un': '1', 'une': '1', 'one': '1',
    'deux': '2', 'two': '2',
    'trois': '3', 'three': '3',
    'quatre': '4', 'four': '4',
    'cinq': '5', 'five': '5',
    'six': '6',
    'sept': '7', 'seven': '7',
    'huit': '8', 'eight': '8',
    'neuf': '9', 'nine': '9',
    'dix': '10', 'ten': '10',
    'onze': '11', 'eleven': '11',
    'douze': '12', 'twelve': '12',
    'treize': '13', 'quatorze': '14', 'quinze': '15', 'seize': '16',
    'vingt': '20', 'twenty': '20',
    'trente': '30', 'thirty': '30',
    'quarante': '40', 'forty': '40',
    'cinquante': '50', 'fifty': '50',
    'soixante': '60', 'sixty': '60',
    'cent': '100', 'hundred': '100',
}

# Équivalences fractions <-> décimales
FRACTION_EQUIVALENTS = {
    '1/2': ['0.5', '0,5', '05', 'un demi', 'une moitié'],
    '1/3': ['0.33', '0,33', '0.333', '0,333', 'un tiers'],
    '2/3': ['0.66', '0,66', '0.67', '0,67', '0.666', '0,666', 'deux tiers'],
    '1/4': ['0.25', '0,25', '025', 'un quart'],
    '3/4': ['0.75', '0,75', '075', 'trois quarts'],
    '1/5': ['0.2', '0,2', '02', 'un cinquième'],
    '2/5': ['0.4', '0,4', '04'],
    '3/5': ['0.6', '0,6', '06'],
    '4/5': ['0.8', '0,8', '08'],
    '1/10': ['0.1', '0,1', '01', 'un dixième'],
    '3/10': ['0.3', '0,3', '03'],
    '1/13': ['0.077', '0,077', '0.08', '0,08'],
}

# Traductions FR <-> EN et variantes orthographiques
TRANSLATIONS = {
    # Planètes
    'mercure': ['mercury'],
    'vénus': ['venus'],
    'terre': ['earth'],
    'mars': ['mars'],
    'jupiter': ['jupiter'],
    'saturne': ['saturn'],
    'uranus': ['uranus'],
    'neptune': ['neptune'],
    # Géographie
    'nil': ['nile'],
    'seine': ['seine'],
    'atlantique': ['atlantic', 'océan atlantique', 'ocean atlantique'],
    'afrique': ['africa'],
    'europe': ['europe'],
    'asie': ['asia'],
    'amérique': ['america'],
    'égypte': ['egypt', 'egypte'],
    'japon': ['japan'],
    'russie': ['russia'],
    'algérie': ['algeria', 'algerie'],
    'australie': ['australia'],
    'canberra': ['canberra'],
    'pise': ['pisa'],
    'tokyo': ['tokyo'],
    'paris': ['paris'],
    # Organes/biologie
    'foie': ['liver'],
    'rein': ['kidney', 'reins'],
    'reins': ['kidney', 'rein'],
    'peau': ['skin'],
    'coeur': ['heart', 'cœur'],
    'cœur': ['heart', 'coeur'],
    'poumon': ['lung', 'poumons'],
    # Chimie - uniquement traductions FR/EN du nom, PAS symbole <-> nom
    'hydrogène': ['hydrogen', 'hydrogene'],
    'oxygène': ['oxygen', 'oxygene'],
    'sodium': ['sodium'],
    'or': ['gold'],
    'carbone': ['carbon'],
    'azote': ['nitrogen'],
    'dioxyde de carbone': ['co2', 'carbon dioxide'],
    'co2': ['dioxyde de carbone', 'Carbon dioxide'],
    # Unités
    'newton': ['n', 'newtons'],
    'joule': ['j', 'joules'],
    'mètre': ['m', 'metre', 'meter'],
    'kilogramme': ['kg', 'kilogram'],
    # Jours
    'dimanche': ['sunday'],
    'lundi': ['monday'],
    'mardi': ['tuesday'],
    'mercredi': ['wednesday'],
    'jeudi': ['thursday'],
    'vendredi': ['friday'],
    'samedi': ['saturday'],
    # Personnages
    'beethoven': ['ludwig van beethoven'],
    'gutenberg': ['johannes gutenberg'],
    'van gogh': ['vincent van gogh', 'vangogh'],
    'napoléon': ['napoleon', 'napoleon bonaparte'],
    # Divers
    'photo': ['photograph', 'photographie', 'une photo'],
    'oui': ['yes'],
    'non': ['no'],
}

# Symboles chimiques courants
# NOTE: On ne fait PAS d'équivalence symbole <-> nom (ex: Au <-> or)
# car si la question demande le symbole, le nom n'est pas une réponse valide
# On garde uniquement les formules moléculaires courantes
CHEMICAL_SYMBOLS = {
    'co2': 'dioxyde de carbone',
    'h2o': 'eau',
}


JUDGE_PROMPT_TEMPLATE = """Compare deux réponses. Ignore casse/ponctuation.

ÉQUIVALENCES VALIDES :
- True/Vrai/Oui → groupe A
- False/Faux/Non → groupe B
- 6 = six

ATTENDU : {expected_answer}
RÉPONSE : {llm_answer}

Règle : CORRECT si même groupe/valeur, sinon INCORRECT.

Exemples :
- True vs Vrai → CORRECT
- False vs Faux → CORRECT
- True vs False → INCORRECT
- Oui vs Non → INCORRECT
- Vrai vs Faux → INCORRECT
- 6 vs six → CORRECT

Réponds UN SEUL MOT :
CORRECT / INCORRECT / UNCERTAIN

Verdict :"""


def normalize_answer(answer: str) -> str:
    """Normalise une réponse pour comparaison simple"""
    import re
    # Convertir en minuscules
    answer = answer.lower().strip()
    # Supprimer la ponctuation SAUF "/" (pour les fractions) et "," et "." (pour les décimaux)
    answer = re.sub(r'[^\w\s/,.]', '', answer)
    # Supprimer espaces multiples
    answer = re.sub(r'\s+', ' ', answer)
    # Supprimer la ponctuation finale (. , qui ne sont pas des décimaux)
    answer = re.sub(r'[.,]+$', '', answer)
    return answer


def normalize_answer_strict(answer: str) -> str:
    """
    Normalisation stricte : supprime articles, prepositions courantes
    pour comparaison semantique (ex: "la Russie" -> "russie", "ocean Atlantique" -> "atlantique")
    """
    import re
    answer = normalize_answer(answer)

    # Articles et mots a supprimer en debut de reponse
    articles = ['le ', 'la ', 'les ', 'l ', 'un ', 'une ', 'des ', 'du ', 'de ',
                'the ', 'a ', 'an ',
                'ocean ', 'océan ', 'mer ', 'mont ', 'montagne ', 'fleuve ', 'riviere ', 'rivière ',
                'pays ', 'ville ', 'capitale ', 'president ', 'président ', 'roi ', 'reine ']

    for article in articles:
        if answer.startswith(article):
            answer = answer[len(article):]

    return answer.strip()


def canonicalize_number(text: str) -> str:
    """Convertit les nombres en lettres vers leur forme numérique"""
    text_lower = text.lower().strip()
    # Vérifier si c'est un nombre en lettres
    if text_lower in NUMBER_WORDS:
        return NUMBER_WORDS[text_lower]
    return text


def canonicalize_fraction(text: str) -> str:
    """Normalise les fractions et décimales vers une forme canonique"""
    import re
    text = text.strip().lower()
    # Remplacer virgule par point pour les décimaux
    text = text.replace(',', '.')
    # Supprimer espaces dans les fractions (ex: "3 / 4" -> "3/4")
    text = re.sub(r'\s*/\s*', '/', text)
    return text


def check_fraction_equivalence(exp: str, given: str) -> bool:
    """Vérifie si deux valeurs sont des fractions/décimales équivalentes"""
    exp_canon = canonicalize_fraction(exp)
    given_canon = canonicalize_fraction(given)

    # Match direct après canonicalisation
    if exp_canon == given_canon:
        return True

    # Vérifier dans le dictionnaire des équivalences
    for fraction, equivalents in FRACTION_EQUIVALENTS.items():
        all_forms = [fraction] + equivalents
        all_forms_canon = [canonicalize_fraction(f) for f in all_forms]
        if exp_canon in all_forms_canon and given_canon in all_forms_canon:
            return True

    # Essayer de convertir en float et comparer
    try:
        # Gérer les fractions type "3/4"
        if '/' in exp_canon:
            num, denom = exp_canon.split('/')
            exp_float = float(num) / float(denom)
        else:
            exp_float = float(exp_canon)

        if '/' in given_canon:
            num, denom = given_canon.split('/')
            given_float = float(num) / float(denom)
        else:
            given_float = float(given_canon)

        # Tolérance pour les arrondis
        if abs(exp_float - given_float) < 0.01:
            return True
    except (ValueError, ZeroDivisionError):
        pass

    return False


def check_translation_equivalence(exp: str, given: str) -> bool:
    """Vérifie si deux valeurs sont des traductions équivalentes"""
    exp_lower = exp.lower().strip()
    given_lower = given.lower().strip()

    # Vérifier dans le dictionnaire de traductions
    for key, equivalents in TRANSLATIONS.items():
        all_forms = [key] + equivalents
        all_forms_lower = [f.lower() for f in all_forms]
        if exp_lower in all_forms_lower and given_lower in all_forms_lower:
            return True

    # Vérifier les symboles chimiques
    for symbol, name in CHEMICAL_SYMBOLS.items():
        if (exp_lower == symbol and given_lower == name.lower()) or \
           (exp_lower == name.lower() and given_lower == symbol):
            return True

    return False


def extract_numeric_value(text: str) -> str:
    """Extrait la valeur numérique d'une chaîne (ex: '60€' -> '60', '28 cm' -> '28')"""
    import re
    # Chercher un nombre (entier ou décimal)
    match = re.search(r'-?\d+(?:[.,]\d+)?', text)
    if match:
        return match.group().replace(',', '.')
    return text


def simple_compare(expected: str, given: str) -> tuple:
    """
    Comparaison simple en fallback si le juge échoue.
    Inclut de nombreuses équivalences pour éviter les faux négatifs.
    Returns: (is_correct: bool, explanation: str)
    """
    exp_norm = normalize_answer(expected)
    given_norm = normalize_answer(given)

    # Comparaison directe
    if exp_norm == given_norm:
        return True, "CORRECT (match exact normalise)"

    # Groupes d'équivalences booléennes
    true_group = {'true', 'vrai', 't', 'v', 'oui', 'yes', 'correct'}
    false_group = {'false', 'faux', 'f', 'non', 'no', 'incorrect'}

    exp_in_true = exp_norm in true_group
    exp_in_false = exp_norm in false_group
    given_in_true = given_norm in true_group
    given_in_false = given_norm in false_group

    # Si les deux sont dans le même groupe -> CORRECT
    if (exp_in_true and given_in_true) or (exp_in_false and given_in_false):
        return True, "CORRECT (equivalence booleenne)"

    # Si dans des groupes opposés -> INCORRECT
    if (exp_in_true and given_in_false) or (exp_in_false and given_in_true):
        return False, "INCORRECT (valeurs booleennes opposees)"

    # Vérifier équivalence nombres en lettres (ex: "3" vs "trois")
    exp_num = canonicalize_number(exp_norm)
    given_num = canonicalize_number(given_norm)
    if exp_num == given_num:
        return True, "CORRECT (equivalence numerique)"

    # Vérifier équivalence fractions/décimales (ex: "3/4" vs "0.75")
    if check_fraction_equivalence(exp_norm, given_norm):
        return True, "CORRECT (equivalence fraction/decimale)"

    # Vérifier équivalence traductions (ex: "Mercure" vs "Mercury")
    if check_translation_equivalence(exp_norm, given_norm):
        return True, "CORRECT (equivalence traduction)"

    # Extraire et comparer les valeurs numériques (ex: "60€" vs "60")
    # MAIS PAS pour les fractions (contenant "/") - elles sont traitées séparément
    if '/' not in exp_norm and '/' not in given_norm:
        exp_numeric = extract_numeric_value(exp_norm)
        given_numeric = extract_numeric_value(given_norm)
        if exp_numeric == given_numeric and exp_numeric != exp_norm:
            return True, "CORRECT (valeur numerique extraite)"

    # Normalisation stricte (supprime articles)
    exp_strict = normalize_answer_strict(expected)
    given_strict = normalize_answer_strict(given)

    # Comparaison après normalisation stricte
    if exp_strict == given_strict:
        return True, "CORRECT (match apres normalisation stricte)"

    # Vérifier équivalences après normalisation stricte
    if check_translation_equivalence(exp_strict, given_strict):
        return True, "CORRECT (equivalence traduction stricte)"

    # Vérifier si l'attendu est contenu dans la réponse
    if len(exp_strict) > 2 and exp_strict in given_strict:
        return True, "CORRECT (attendu contenu dans reponse)"

    # Vérifier si la réponse est contenue dans l'attendu
    if len(given_strict) > 2 and given_strict in exp_strict:
        return True, "CORRECT (reponse contenue dans attendu)"

    # Match partiel avec normalisation basique
    if len(exp_norm) > 2 and (exp_norm in given_norm or given_norm in exp_norm):
        return True, "CORRECT (match partiel)"

    # Dernière tentative: comparaison numérique directe (seulement si pas de fractions)
    if '/' not in exp_norm and '/' not in given_norm:
        try:
            exp_float = float(extract_numeric_value(exp_norm))
            given_float = float(extract_numeric_value(given_norm))
            if abs(exp_float - given_float) < 0.001:
                return True, "CORRECT (valeurs numeriques egales)"
        except ValueError:
            pass

    return False, "INCORRECT (pas de correspondance)"


def judge_answer(
    expected_answer: str,
    llm_answer: str,
    judge_client: LLMClient,
    verbose: bool = False
) -> tuple:
    """
    Utilise un LLM comme juge pour évaluer une réponse.
    Compare uniquement expected_answer et llm_answer (ignore casse, ponctuation, équivalences).

    Returns:
        (is_correct: bool, judge_response: str)
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        expected_answer=expected_answer,
        llm_answer=llm_answer
    )

    try:
        response = judge_client.generate(prompt)
        judge_verdict = response.text.strip().upper()

        # Extraire le verdict (priorité à la première occurrence pour éviter confusion)
        # On cherche le premier mot-clé rencontré dans la réponse
        verdict_type = None
        is_correct = None

        # Parser ligne par ligne pour trouver le verdict
        lines = judge_verdict.split('\n')
        for line in lines:
            line_clean = line.strip().upper()

            # Chercher INCORRECT en premier (plus spécifique que CORRECT)
            if "INCORRECT" in line_clean and verdict_type is None:
                verdict_type = "INCORRECT"
                is_correct = False
                break
            # Puis UNCERTAIN
            elif "UNCERTAIN" in line_clean and verdict_type is None:
                verdict_type = "UNCERTAIN"
                is_correct = None
                break
            # Puis CORRECT (doit vérifier que ce n'est pas INCORRECT)
            elif "CORRECT" in line_clean and "INCORRECT" not in line_clean and verdict_type is None:
                verdict_type = "CORRECT"
                is_correct = True
                break

        # Si aucun verdict trouvé, considérer comme UNCERTAIN
        if verdict_type is None:
            verdict_type = "UNCERTAIN"
            is_correct = None
            if verbose:
                print(f"    [!] Aucun verdict clair trouvé dans: {judge_verdict[:100]}")

        # Avertir si la réponse est trop longue (signe que le juge n'a pas suivi la consigne)
        if len(judge_verdict) > 50 and verbose:
            print(f"    [!] Réponse juge trop longue ({len(judge_verdict)} chars): {judge_verdict[:100]}...")

        if verbose:
            print(f"    Juge: {judge_verdict[:50]}... -> {verdict_type}")

        return is_correct, judge_verdict

    except Exception as e:
        print(f"    [!] Erreur jugement: {e}")
        # En cas d'erreur, on garde l'évaluation originale
        return None, str(e)


def judge_results_file(
    input_file: str,
    judge_model: str,
    output_file: str = None,
    verbose: bool = False,
    temperature: float = 0.0,
    use_simple_compare: bool = False
) -> Dict:
    """
    Évalue les réponses d'un fichier de résultats avec un juge LLM ou comparaison simple

    Comportement standardisé :
    - Phase 1.0 : Juge llm_answer
    - Phase 1.1/1.2 : Juge UNIQUEMENT final_answer (initial_answer conservé mais non jugé)

    Le juge compare uniquement expected_answer vs réponse finale.
    initial_answer est gardé à titre indicatif pour l'analyse de changement.

    Args:
        input_file: Fichier JSON d'entrée
        judge_model: Nom du modèle juge (ex: mistral:latest)
        output_file: Fichier JSON de sortie (optionnel)
        verbose: Afficher les détails
        temperature: Température pour le juge
        use_simple_compare: Si True, utilise comparaison simple sans LLM (plus rapide mais moins flexible)

    Returns:
        Statistiques simples du jugement
    """
    print(f"\n{'='*80}")
    print(f"JUGEMENT : {Path(input_file).name}")
    print(f"{'='*80}")

    with open(input_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    tested_model = results["metadata"]["model"]
    print(f"Modele teste : {tested_model}")
    if use_simple_compare:
        print(f"Mode : Comparaison simple (sans LLM)")
    else:
        print(f"Modele juge : {judge_model}")
    print(f"Questions : {len(results['questions'])}")

    # Initialiser le juge (seulement si on n'utilise pas la comparaison simple)
    judge_client = None
    if not use_simple_compare:
        judge_client = LLMClient(
            model=judge_model,
            temperature=temperature,
            max_tokens=20  # Réduit pour forcer des réponses courtes
        )

        # Vérifier la disponibilité
        if not judge_client.is_available():
            raise ConnectionError("Ollama n'est pas accessible")

    # Détecter si c'est un fichier avec initial_answer et final_answer
    has_initial_and_final = any("initial_answer" in q and "final_answer" in q for q in results["questions"])

    # Statistiques simples
    stats = {
        "model": tested_model,
        "judge_model": judge_model,
        "total": len(results["questions"]),
        "has_review": has_initial_and_final,
        "correct": 0,
        "incorrect": 0,
        "uncertain": 0,
        "errors": 0
    }

    print(f"\nType de fichier : {'Avec relecture (initial + final)' if has_initial_and_final else 'Sans relecture (réponse unique)'}")
    if has_initial_and_final:
        print(f"INFO : initial_answer est conservé à titre indicatif mais n'est PAS jugé")
    print(f"Debut du jugement...")
    print("-" * 80)

    # Juger chaque question
    for i, q in enumerate(results["questions"], 1):
        if verbose:
            print(f"\n[{i}/{len(results['questions'])}] {q['id']} - {q['domain']}/{q['subdomain']}")
            print(f"    Attendu: {q['expected_answer']}")

        # Déterminer quelle réponse juger
        if has_initial_and_final:
            # Phase 1.1/1.2 : juger UNIQUEMENT final_answer (initial_answer gardé à titre indicatif)
            answer_to_judge = q.get("final_answer", "")
            if verbose:
                print(f"    Initial (non jugé): {q.get('initial_answer', '')[:60]}...")
                print(f"    Final: {answer_to_judge[:60]}...")
        else:
            # Phase 1.0 : juger llm_answer
            answer_to_judge = q.get("llm_answer", "")
            if verbose:
                print(f"    Reponse: {answer_to_judge[:60]}...")

        # Juger la réponse
        if use_simple_compare:
            # Mode comparaison simple (sans LLM)
            is_correct, judge_verdict = simple_compare(
                q["expected_answer"],
                answer_to_judge
            )
        else:
            # Mode juge LLM (par défaut)
            is_correct, judge_verdict = judge_answer(
                q["expected_answer"],
                answer_to_judge,
                judge_client,
                verbose
            )

        # Gérer les erreurs de jugement
        if is_correct is None and "ERROR:" in str(judge_verdict):
            stats["errors"] += 1
            if verbose:
                print(f"    [!] Erreur jugement: {judge_verdict}")

        # Ajouter les champs de jugement
        q["judge_verdict"] = judge_verdict
        q["is_correct"] = is_correct

        # Compter les verdicts
        if is_correct is True:
            stats["correct"] += 1
        elif is_correct is False:
            stats["incorrect"] += 1
        elif is_correct is None:
            stats["uncertain"] += 1

        # Affichage simplifié en mode non-verbose
        if not verbose:
            verdict_display = "✓" if is_correct is True else "✗" if is_correct is False else "?"
            print(f"[{i}/{len(results['questions'])}] {q['id']} - {verdict_display}", end="\r")

    print("\n")

    # Ajouter metadata de jugement
    if "judge_metadata" not in results["metadata"]:
        results["metadata"]["judge_metadata"] = {}

    results["metadata"]["judge_metadata"] = {
        "judged": True,
        "judged_at": datetime.now().isoformat(),
        "judge_mode": "simple_compare" if use_simple_compare else "llm_judge",
        "judge_model": judge_model if not use_simple_compare else "N/A",
        "judge_temperature": temperature if not use_simple_compare else "N/A"
    }

    # Sauvegarder
    if output_file is None:
        input_path = Path(input_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = input_path.parent / f"{input_path.stem}_judged_{timestamp}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Afficher le résumé
    print("=" * 80)
    print("RESUME DU JUGEMENT")
    print("=" * 80)

    # Affichage uniforme pour tous les fichiers
    print(f"\nVerdicts (basés sur {'final_answer' if has_initial_and_final else 'llm_answer'}) :")
    print(f"  CORRECT   : {stats['correct']:3d} ({stats['correct']/stats['total']:.1%})")
    print(f"  INCORRECT : {stats['incorrect']:3d} ({stats['incorrect']/stats['total']:.1%})")
    print(f"  UNCERTAIN : {stats['uncertain']:3d} ({stats['uncertain']/stats['total']:.1%})")

    if stats['errors'] > 0:
        print(f"\nERREURS   : {stats['errors']:3d}")

    print(f"\nFichier sauvegarde : {output_file}")

    return stats


def display_summary(stats_list: List[Dict]) -> None:
    """Affiche un résumé simple des verdicts pour plusieurs fichiers"""
    print("\n" + "=" * 80)
    print("RESUME MULTI-FICHIERS")
    print("=" * 80)

    print(f"\n{'Modele':<28} {'Correct':>8} {'Incorrect':>10} {'Uncertain':>10} {'Total':>8}")
    print("-" * 80)

    total_correct = 0
    total_incorrect = 0
    total_uncertain = 0
    total_questions = 0

    for stats in stats_list:
        model = stats.get("model", "Unknown")[:26]
        correct = stats.get("correct", 0)
        incorrect = stats.get("incorrect", 0)
        uncertain = stats.get("uncertain", 0)
        total = stats.get("total", 0)

        total_correct += correct
        total_incorrect += incorrect
        total_uncertain += uncertain
        total_questions += total

        print(f"{model:<28} {correct:>8d} {incorrect:>10d} {uncertain:>10d} {total:>8d}")

    print("-" * 80)
    print(f"{'TOTAL':<28} {total_correct:>8d} {total_incorrect:>10d} {total_uncertain:>10d} {total_questions:>8d}")


def main():
    parser = argparse.ArgumentParser(
        description="Module de jugement standardisé - Compare expected_answer vs réponse finale (llm_answer ou final_answer)"
    )
    parser.add_argument(
        "files",
        nargs='+',
        help="Fichiers JSON de résultats à juger"
    )
    parser.add_argument(
        "--judge",
        type=str,
        default="phi3:mini",
        help="Modèle LLM à utiliser comme juge (défaut: phi3:mini)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Température pour le juge (défaut: 0.0)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Afficher les détails pour chaque question"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Afficher un résumé multi-fichiers à la fin"
    )
    parser.add_argument(
        "--simple-compare",
        action="store_true",
        help="Utiliser la comparaison simple (sans LLM) au lieu du juge LLM"
    )

    args = parser.parse_args()

    # Résoudre les wildcards dans les chemins de fichiers
    resolved_files = []
    for file_pattern in args.files:
        # Gérer les wildcards Windows et Unix
        matches = glob.glob(file_pattern)
        if matches:
            resolved_files.extend(matches)
        else:
            # Essayer avec une extension .json si pas déjà présente
            if not file_pattern.endswith('.json') and not '*' in file_pattern:
                matches = glob.glob(f"{file_pattern}*.json")
                if matches:
                    resolved_files.extend(matches)
                else:
                    print(f"[!] Aucun fichier trouvé pour : {file_pattern}")
            else:
                print(f"[!] Aucun fichier trouvé pour : {file_pattern}")

    # Vérifier qu'on a au moins un fichier
    if not resolved_files:
        print("\n[ERREUR] Aucun fichier trouvé correspondant aux patterns fournis.")
        print("\nExemples de commandes valides :")
        print("  python judge_results.py resultats_phase1.0/*.json --judge phi3:mini")
        print("  python judge_results.py resultats_phase1.1_relecture_simple/*.json --judge phi3:mini")
        print("  python judge_results.py \"resultats_phase1.1_relecture_simple/results_*.json\" --judge phi3:mini")
        return 1

    print("=" * 80)
    print("MODULE DE JUGEMENT")
    print("=" * 80)
    print(f"\nModele juge : {args.judge}")
    print(f"Fichiers a juger : {len(resolved_files)}")
    for f in resolved_files:
        print(f"  - {Path(f).name}")

    all_stats = []

    for input_file in resolved_files:
        try:
            stats = judge_results_file(
                input_file,
                args.judge,
                output_file=None,  # Auto-généré
                verbose=args.verbose,
                temperature=args.temperature,
                use_simple_compare=args.simple_compare
            )

            all_stats.append(stats)

        except Exception as e:
            print(f"\n[ERREUR] {input_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Résumé multi-fichiers si demandé
    if args.summary and len(all_stats) > 1:
        display_summary(all_stats)

    print("\n" + "=" * 80)
    print("TERMINE !")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
