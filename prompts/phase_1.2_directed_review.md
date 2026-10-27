# Phase 1.2 — Stateless directed review

Identical to Phase 1.1, plus a short list of domain-specific attention cues ("INDICES
UTILES") drawn from a fixed dictionary keyed by (domain, subdomain). Cues are written
at the subdomain level, never reference the specific expected answer, and are identical
for every model. `{question}`, `{initial_answer}`, and `{attention_points}` are
substituted per item.

## Prompt (French, verbatim)

```
Question : {question}

CONTEXTE : Une tentative de réponse a été proposée : "{initial_answer}"

INDICES UTILES pour cette question :
{attention_points}

CONSIGNE : Réponds à la question ci-dessus en utilisant les indices fournis.
- Réponds UNIQUEMENT par la valeur attendue (ex: Vrai, Faux, 42, Paris, etc.)
- Ne donne AUCUNE explication, ne justifie pas
- Ne commente pas la tentative précédente ni les indices
- Donne simplement TA réponse à la question

Réponse :
```

## English translation

> Question: {question}. CONTEXT: An answer attempt has been proposed: "{initial_answer}".
> USEFUL HINTS for this question: {attention_points}. INSTRUCTION: Answer the question
> above using the provided hints. Answer ONLY with the expected value. Give NO
> explanation. Do not comment on the previous attempt or the hints. Simply give YOUR
> answer. Answer:

## Attention cue example

Cues are drawn from a fixed dictionary keyed by (domain, subdomain), with one to three
cues per item. Example for logic/implication:

> *"La structure de l'implication (si...alors) est-elle bien comprise ? Les règles du
> modus ponens et modus tollens sont-elles appliquées ? Y a-t-il confusion entre
> implication et causalité ?"*

Cues never reference the expected answer and are identical for all models. See the
avoided-anti-patterns note in `phase_1.1_simple_review.md` — it applies identically here.
