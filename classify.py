import json
from gemini_client import generate_json


def classify_batch(texts, labels):
    """
    Classifica um batch de textos usando IDs para reduzir tamanho e evitar truncamento.
    """
    labels_str = ", ".join([f'"{l}"' for l in labels])

    prompt = f"""
Classifique cada texto abaixo usando APENAS uma das labels.

Labels permitidas:
[{labels_str}]

Retorne SOMENTE uma lista JSON válida NO FORMATO:

[
  {{"id": 0, "label": "..." }},
  {{"id": 1, "label": "..." }},
  ...
]

NÃO repita o texto original. Use SOMENTE o campo "id".

Textos:
"""
    prompt += "\n".join(f"{i}: {t}" for i, t in enumerate(texts))

    return generate_json(prompt)


def classify_batch_only(df, labels, batch_size=10):
    texts = df["description"].tolist()
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"    → batch {i}–{i + len(batch) - 1}")

        batch_result = classify_batch(batch, labels)

        if isinstance(batch_result, str):
            try:
                parsed = json.loads(batch_result)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSON inválido retornado pelo modelo no batch {i}: {e}\n"
                    f"Saída bruta:\n{batch_result}"
                )
        else:
            parsed = batch_result

        for item in parsed:
            global_index = i + item["id"]
            label = item["label"]
            results.append({
                "text": texts[global_index],
                "label": label
            })

    return results

