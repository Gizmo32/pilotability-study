# Phase 1.1 — Stateless simple review

The prior (Phase 1.0) answer is injected as an external artifact, never attributed to
the model. The model is instructed to answer directly, not to comment on, validate, or
revise the previous attempt. `{question}` and `{initial_answer}` are substituted per item.

## Prompt (French, verbatim)

```
Question : {question}

CONTEXTE : Une tentative de réponse a été proposée : "{initial_answer}"

CONSIGNE : Réponds maintenant à la question ci-dessus.
- Réponds UNIQUEMENT par la valeur attendue (ex: Vrai, Faux, 42, Paris, etc.)
- Ne donne AUCUNE explication, ne justifie pas
- Ne commente pas la tentative précédente
- Donne simplement TA réponse à la question

Réponse :
```

## English translation

> Question: {question}. CONTEXT: An answer attempt has been proposed: "{initial_answer}".
> INSTRUCTION: Now answer the question above. Answer ONLY with the expected value (e.g.,
> True, False, 42, Paris, etc.). Give NO explanation, do not justify. Do not comment on
> the previous attempt. Simply give YOUR answer to the question. Answer:

## Avoided anti-patterns

The following formulations were explicitly excluded from all review prompts (Phase 1.1
and 1.2), because pilot testing showed they induce meta-evaluation, sycophantic
agreement, or role confusion in small models:

- *"Vérifie si la réponse précédente est correcte"* (verify whether the previous answer
  is correct)
- *"Si la réponse était fausse, corrige-la"* (if the answer was wrong, fix it)
- *"Es-tu d'accord avec la réponse précédente ?"* (do you agree with the previous answer?)

Outputs were additionally screened for meta-evaluative patterns (explicit judgments or
commentary on the prior answer); such outputs are scored as incorrect by the
conservative judge.
