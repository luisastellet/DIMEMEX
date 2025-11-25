import argparse
import pandas as pd
from pathlib import Path

CLASSES = ["hate speech", "inappropriate content", "neither"]

def normalize_label(x: str) -> str:
    x = str(x).strip().lower()
    if "hate" in x:
        return "hate speech"
    if "inap" in x or "inappropriate" in x:
        return "inappropriate content"
    if "neither" in x or "none" in x or "neutral" in x:
        return "neither"
    return x  # mantém como veio se não bater (a contagem mostrará)

def count_file(csv_path: Path, name: str):
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError(f"O arquivo {csv_path} não tem coluna 'label'.")
    labels = df["label"].map(normalize_label)
    counts = labels.value_counts().reindex(CLASSES, fill_value=0)
    total = int(counts.sum())
    perc = (counts / max(total, 1) * 100).round(2)

    print(f"\n=== {name} ({total} amostras) ===")
    for c in CLASSES:
        print(f"- {c:>24}: {int(counts[c]):>6}  ({perc[c]:>6.2f}%)")

    # Métrica simples de desbalanceamento: razão max/min
    min_c = counts[counts > 0].min() if (counts > 0).any() else 0
    max_c = counts.max()
    ratio = round(max_c / min_c, 2) if min_c > 0 else "inf"
    print(f"Desbalance ratio (max/min): {ratio}")
    return counts

def main():
    ap = argparse.ArgumentParser(description="Verifica balanceamento das classes.")
    ap.add_argument("--train", default="train/dados_espanhol_balanceado.csv")
    ap.add_argument("--val",   default="validation/dados_espanhol.csv")
    ap.add_argument("--test",  default="test/dados_espanhol.csv")
    args = ap.parse_args()

    all_counts = []
    if Path(args.train).exists():
        all_counts.append(count_file(args.train, "TRAIN"))
    if Path(args.val).exists():
        all_counts.append(count_file(args.val, "VALIDATION"))
    if Path(args.test).exists():
        all_counts.append(count_file(args.test, "TEST"))

    if all_counts:
        total_counts = sum(all_counts)
        total = int(total_counts.sum())
        perc = (total_counts / max(total, 1) * 100).round(2)
        print(f"\n=== TOTAL (todos os splits) ({total} amostras) ===")
        for c in CLASSES:
            print(f"- {c:>24}: {int(total_counts[c]):>6}  ({perc[c]:>6.2f}%)")
        min_c = total_counts[total_counts > 0].min() if (total_counts > 0).any() else 0
        max_c = total_counts.max()
        ratio = round(max_c / min_c, 2) if min_c > 0 else "inf"
        print(f"Desbalance ratio (max/min): {ratio}")

if __name__ == "__main__":
    main()