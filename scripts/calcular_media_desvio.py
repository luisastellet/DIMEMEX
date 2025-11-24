import pandas as pd
import json
import math


def calcular_metricas_pasta(pasta):
    """Calcula médias e desvios padrão para text e description de uma pasta"""
    caminho = f"{pasta}/metricas.json"

    with open(caminho, "r") as f:
        dados = json.load(f)
    
    qtd = len(dados)
    
    # Inicializar somas para text e description
    metricas = {
        "text": {"bertscore": 0.0, "bleurt": 0.0, "cometkiwi": 0.0, "chrf": 0.0},
        "description": {"bertscore": 0.0, "bleurt": 0.0, "cometkiwi": 0.0, "chrf": 0.0}
    }
    
    # Somar todas as métricas
    for objeto in dados:
        for elemento in ["text", "description"]:
            metricas[elemento]["bertscore"] += objeto["metricas"][elemento]["bertscore"]
            metricas[elemento]["bleurt"] += objeto["metricas"][elemento]["bleurt"]
            metricas[elemento]["cometkiwi"] += objeto["metricas"][elemento]["cometkiwi"]
            metricas[elemento]["chrf"] += objeto["metricas"][elemento]["chrf"]
    
    # Calcular médias
    medias = {}
    for elemento in ["text", "description"]:
        medias[elemento] = {
            "bertscore": metricas[elemento]["bertscore"] / qtd,
            "bleurt": metricas[elemento]["bleurt"] / qtd,
            "cometkiwi": metricas[elemento]["cometkiwi"] / qtd,
            "chrf": metricas[elemento]["chrf"] / qtd
        }
    
    # Inicializar somas dos quadrados das diferenças para desvio padrão
    desvios = {
        "text": {"bertscore": 0.0, "bleurt": 0.0, "cometkiwi": 0.0, "chrf": 0.0},
        "description": {"bertscore": 0.0, "bleurt": 0.0, "cometkiwi": 0.0, "chrf": 0.0}
    }
    
    # Calcular somas dos quadrados das diferenças
    for objeto in dados:
        for elemento in ["text", "description"]:
            desvios[elemento]["bertscore"] += (objeto["metricas"][elemento]["bertscore"] - medias[elemento]["bertscore"]) ** 2
            desvios[elemento]["bleurt"] += (objeto["metricas"][elemento]["bleurt"] - medias[elemento]["bleurt"]) ** 2
            desvios[elemento]["cometkiwi"] += (objeto["metricas"][elemento]["cometkiwi"] - medias[elemento]["cometkiwi"]) ** 2
            desvios[elemento]["chrf"] += (objeto["metricas"][elemento]["chrf"] - medias[elemento]["chrf"]) ** 2
    
    # Retornar médias e desvios padrão
    resultados = []
    for elemento in ["text", "description"]:
        resultado = {
            "dataset": pasta,
            "elemento": elemento,
            "bertscore_media": round(medias[elemento]["bertscore"], 3),
            "bertscore_desvio": round(math.sqrt(desvios[elemento]["bertscore"] / qtd), 3),
            "bleurt_media": round(medias[elemento]["bleurt"], 3),
            "bleurt_desvio": round(math.sqrt(desvios[elemento]["bleurt"] / qtd), 3),
            "cometkiwi_media": round(medias[elemento]["cometkiwi"], 3),
            "cometkiwi_desvio": round(math.sqrt(desvios[elemento]["cometkiwi"] / qtd), 3),
            "chrf_media": round(medias[elemento]["chrf"], 3),
            "chrf_desvio": round(math.sqrt(desvios[elemento]["chrf"] / qtd), 3)
        }
        resultados.append(resultado)
    
    return resultados

if __name__ == "__main__":
    
    # Lista para armazenar todos os dados consolidados
    dados_consolidados = []
    
    # Processar cada pasta
    for pasta in ["train", "test", "validation"]:
        print(f"Processando {pasta}...")
        resultados = calcular_metricas_pasta(pasta)
        dados_consolidados.extend(resultados)
    
    # Criar DataFrame consolidado
    df_consolidado = pd.DataFrame(dados_consolidados)
    
    # Reordenar colunas para melhor visualização
    colunas_ordem = [
        "dataset", "elemento",
        "bertscore_media", "bertscore_desvio",
        "bleurt_media", "bleurt_desvio",
        "cometkiwi_media", "cometkiwi_desvio",
        "chrf_media", "chrf_desvio"
    ]
    df_consolidado = df_consolidado[colunas_ordem]
    
    # Salvar CSV consolidado
    df_consolidado.to_csv("media_desvio.csv", index=False)
    print(f"\n✓ CSV consolidado salvo em media_desvio.csv")
    print(f"Total de linhas: {len(df_consolidado)}")
    
    print("\nPré-visualização:")
    print(df_consolidado.to_string())