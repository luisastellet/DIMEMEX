# main.py
import json
import pandas as pd
from classify import classify_batch_only
from cleaner import clean_text

datasets = ["train","validation", "test"]
labels = ["discurso de ódio", "conteúdo inapropriado", "nenhum"]


def open_dataset(name):
    path = f"data/{name}/{name}_data_translated.json"
    with open(path, "r", encoding="utf-8") as f:
        df = pd.DataFrame(json.load(f))

    df["text"] = df["text"].apply(clean_text)
    return df


def main():
    for ds in datasets:
        print(f"\nProcessando dataset '{ds}'...")

        df = open_dataset(ds)

        print("Classificando TEXT (batch)...")
        df["label_description_only"] = classify_batch_only(df, labels)


        out = f"data/{ds}/{ds}_description_labeled.json"
        df.to_json(out, orient="records", force_ascii=False, indent=2)

        print(f"Finalizado e salvo em: {out}")


if __name__ == "__main__":
    main()
