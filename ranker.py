import json
import pandas as pd

## Variables
metric_map = {
    "text": "mt",
    "description": "md",
}
datasets = ["train", "validation", "test"]
percentage = 0.1  
final_rank_value = "rank_total_final"

## Functions
def open_dataset(name: str):
    """Lê o JSON de métricas."""
    with open(f"{name}/metricas.json", "r", encoding="utf-8") as f:
        return json.load(f)

def rename_columns(df):
    rename_cols = {
        f"metricas_{tipo}_{metrica}": f"{prefixo}_{metrica}"
        for tipo, prefixo in metric_map.items()
        for metrica in ["bertscore", "bleurt", "cometkiwi", "chrf"]
    }
    return df.rename(columns=rename_cols)

def ranking_metrics(df: pd.DataFrame):
    """Cria colunas de ranking (1 = melhor)."""
    
    metrics = [f"{m}_{x}" for m in ["mt", "md"] for x in ["bertscore", "bleurt", "cometkiwi", "chrf"]]
    df = df[["MEME-ID"] + metrics].copy()
    
    for col in metrics:
        df[f"rank_{col}"] = df[col].rank(method="dense", ascending=False).astype(int)

    return df

def apply_metrics_ranks(dataset_name: str):
    
    data = open_dataset(dataset_name)
    df = pd.json_normalize(data, sep="_")
    df = rename_columns(df)

    df_reduzido = ranking_metrics(df)

    ## Building total ranks for each modality
    df_reduzido["rank_total_mt"] = df_reduzido[[f"rank_mt_{m}" for m in ["bertscore", "bleurt", "cometkiwi", "chrf"]]].sum(axis=1)
    df_reduzido["rank_total_md"] = df_reduzido[[f"rank_md_{m}" for m in ["bertscore", "bleurt", "cometkiwi", "chrf"]]].sum(axis=1)

    ## Using sum to select best overall ranks
    df_reduzido[final_rank_value] = df_reduzido[["rank_total_mt", "rank_total_md"]].sum(axis=1)
    
    df_ranked = df_reduzido[["MEME-ID", final_rank_value]]
    df_ranked = df_ranked.sort_values(final_rank_value, ascending=False)

    df_ranked.to_json(f"{dataset_name}/{dataset_name}_ranked.json",orient="records",force_ascii=False,indent=2)
    return df_ranked

def load_info_dataset(dataset_name: str):
    data = open_dataset(dataset_name)

    df_info = pd.DataFrame({
        "MEME-ID": [x["MEME-ID"] for x in data],
        "original": [x["original"] for x in data],
        "traduzido": [x["traduzido"] for x in data],
    })

    return df_info

def apply():
    for dataset in datasets:
        apply_metrics_ranks(dataset)
        df_rank = apply_metrics_ranks(dataset)

        n_cases = max(1, int(len(df_rank) * percentage))
    
        # Menor soma de ranks = melhor (melhores métricas de tradução)
        better_cases = df_rank.nsmallest(n_cases, final_rank_value)
        # Maior soma de ranks = pior (piores métricas de tradução)
        worse_cases = df_rank.nlargest(n_cases, final_rank_value)
        
        
        ####### INCLUDING FULL DATA FOR BETTER AND WORSE CASES ########
        df_info = load_info_dataset(dataset)

        df_final_better_cases = better_cases.merge(df_info, on="MEME-ID", how="left")
        df_final_worse_cases = worse_cases.merge(df_info, on="MEME-ID", how="left")
        
        better_cases.to_json(f"{dataset}/{dataset}_better_cases_{percentage}.json",orient="records",force_ascii=False,indent=2)
        worse_cases.to_json(f"{dataset}/{dataset}_worse_cases_{percentage}.json",orient="records",force_ascii=False,indent=2)
        
        df_final_better_cases.to_json(f"{dataset}/{dataset}_better_cases_junto.json",orient="records",force_ascii=False,indent=2)
        df_final_worse_cases.to_json(f"{dataset}/{dataset}_worse_cases_junto.json",orient="records",force_ascii=False,indent=2)
        
        print(f"Processado: {dataset} | melhores: {len(better_cases)} | piores: {len(worse_cases)}")
        
        

apply()
