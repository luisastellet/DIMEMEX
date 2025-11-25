import pandas as pd
from math import floor

# Aplicamos proportional undersampling para reduzir a classe majoritária. Mantivemos as classes minoritárias intactas e definimos o novo tamanho da classe majoritária multiplicando o tamanho da menor classe por uma razão máxima de 1,55, resultando em 600 exemplos. Essa abordagem reduz o desbalanceamento mantendo a proporcionalidade entre as classes.

df = pd.read_csv("train/dados_espanhol.csv")

df["__orig_idx"] = df.index

df_major = df[df['label'] == 'hate speech']
df_inapp = df[df['label'] == 'inappropriate content']
df_nei   = df[df['label'] == 'neither']

minor = min(len(df_inapp), len(df_nei))
ratio = 1.55
target_major = floor(minor * ratio)

df_major_sampled = df_major.sample(target_major, random_state=42)

df_balanced = pd.concat([
    df_major_sampled,
    df_inapp,
    df_nei
], ignore_index=False)

df_balanced = df_balanced.sort_values("__orig_idx")

df_balanced = df_balanced.drop(columns=["__orig_idx"])

df_final = df_balanced[["image_path", "label", "text", "description"]]

df_final.to_csv("train/dados_espanhol_balanceado.csv", index=False)
