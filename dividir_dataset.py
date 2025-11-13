import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import re

def extrair_numero_imagem(meme_id):
    """
    Extrai o número da imagem do MEME-ID (ex: DS_IMG_2448.jpg -> 2448)
    """
    match = re.search(r'DS_IMG_(\d+)\.jpg', meme_id)
    return int(match.group(1)) if match else None

def dividir_dados():
    """
    Divide os dados em 70% treino, 20% validação e 10% teste
    """
    
    # Carregar dados JSON
    print(" Carregando dados JSON...")
    
    with open(f"train/train_data.json", 'r', encoding='utf-8') as f:
        dados_originais = json.load(f)
    
    with open(f"train/train_data_translated.json", 'r', encoding='utf-8') as f:
        dados_traduzidos = json.load(f)
    
    with open(f"train/train_data_junto.json", 'r', encoding='utf-8') as f:
        dados_junto = json.load(f)
    
    # Extrair números das imagens dos dados originais (mantém ordem)
    print(" Extraindo números das imagens...")
    numeros_imagens = []
    for item in dados_originais:
        numero = extrair_numero_imagem(item['MEME-ID'])
        if numero is not None:
            numeros_imagens.append(numero)
        else:
            raise ValueError(f"Não foi possível extrair número da imagem: {item['MEME-ID']}")
    
    # Carregar labels
    print(" Carregando labels...")
    
    labels_task = pd.read_csv(f"train/train_labels_tasks_1_3.csv", header=None)
    
    # Verificar se todos têm o mesmo número de amostras
    n_samples = len(dados_originais)
    print(f" Total de amostras: {n_samples}")
    
    assert len(dados_traduzidos) == n_samples, "Dados traduzidos têm tamanho diferente!"
    assert len(dados_junto) == n_samples, "Dados junto têm tamanho diferente!" 
    assert len(labels_task) == n_samples, "Labels task têm tamanho diferente!"
    assert len(numeros_imagens) == n_samples, "Números das imagens têm tamanho diferente!"
    
    # Criar índices para a divisão
    indices = np.arange(n_samples)
    
    # Primeira divisão: 70% treino, 30% restante
    indices_treino, indices_temp = train_test_split(
        indices, 
        test_size=0.3, 
        random_state=42, 
        shuffle=True
    )
    
    # Segunda divisão: do 30% restante -> 20% validação (66.67% de 30%) e 10% teste (33.33% de 30%)
    indices_validacao, indices_teste = train_test_split(
        indices_temp, 
        test_size=0.3333, # 10% do total / 30% restante = 0.3333
        random_state=42, 
        shuffle=True
    )
    
    print(f" Divisão criada:")
    print(f"   - Treino: {len(indices_treino)} amostras ({len(indices_treino)/n_samples*100:.1f}%)")
    print(f"   - Validação: {len(indices_validacao)} amostras ({len(indices_validacao)/n_samples*100:.1f}%)")
    print(f"   - Teste: {len(indices_teste)} amostras ({len(indices_teste)/n_samples*100:.1f}%)")
    
    # Função para dividir lista por índices
    def dividir_por_indices(dados_lista, indices):
        return [dados_lista[i] for i in indices]
    
    # Dividir os dados JSON
    print(" Dividindo dados JSON...")
    
    # Dados originais
    treino_orig = dividir_por_indices(dados_originais, indices_treino)
    val_orig = dividir_por_indices(dados_originais, indices_validacao)
    teste_orig = dividir_por_indices(dados_originais, indices_teste)
    
    # Dados traduzidos
    treino_trad = dividir_por_indices(dados_traduzidos, indices_treino)
    val_trad = dividir_por_indices(dados_traduzidos, indices_validacao)
    teste_trad = dividir_por_indices(dados_traduzidos, indices_teste)
    
    # Dados junto
    treino_junto = dividir_por_indices(dados_junto, indices_treino)
    val_junto = dividir_por_indices(dados_junto, indices_validacao)
    teste_junto = dividir_por_indices(dados_junto, indices_teste)
    
    # Dividir labels e adicionar coluna de imagem
    print("Dividindo labels e adicionando coluna de imagem...")
    
    # Dividir números das imagens
    treino_nums_img = [numeros_imagens[i] for i in indices_treino]
    val_nums_img = [numeros_imagens[i] for i in indices_validacao]  
    teste_nums_img = [numeros_imagens[i] for i in indices_teste]
    
    # Labels 
    treino_labels = labels_task.iloc[indices_treino].copy()
    val_labels = labels_task.iloc[indices_validacao].copy()
    teste_labels = labels_task.iloc[indices_teste].copy()
    
    # Adicionar coluna 'image' no início
    treino_labels.insert(0, 'image', treino_nums_img)
    val_labels.insert(0, 'image', val_nums_img)
    teste_labels.insert(0, 'image', teste_nums_img)
    
    # Criar diretórios se não existirem
    print(" Criando diretórios...")
    
    dirs = ['train', 'validation', 'test']
    for dir_path in dirs:
        full_path = f"{dir_path}"
        os.makedirs(full_path, exist_ok=True)
    
    # Salvar dados JSON
    print(" Salvando arquivos JSON...")
    
    # Treino
    with open(f"train/train_data.json", 'w', encoding='utf-8') as f:
        json.dump(treino_orig, f, ensure_ascii=False, indent=2)
    
    with open(f"train/train_data_translated.json", 'w', encoding='utf-8') as f:
        json.dump(treino_trad, f, ensure_ascii=False, indent=2)
    
    with open(f"train/train_data_junto.json", 'w', encoding='utf-8') as f:
        json.dump(treino_junto, f, ensure_ascii=False, indent=2)
    
    # Validação
    with open(f"validation/validation_data.json", 'w', encoding='utf-8') as f:
        json.dump(val_orig, f, ensure_ascii=False, indent=2)
    
    with open(f"validation/validation_data_translated.json", 'w', encoding='utf-8') as f:
        json.dump(val_trad, f, ensure_ascii=False, indent=2)
    
    with open(f"validation/validation_data_junto.json", 'w', encoding='utf-8') as f:
        json.dump(val_junto, f, ensure_ascii=False, indent=2)
    
    # Teste
    with open(f"test/test_data.json", 'w', encoding='utf-8') as f:
        json.dump(teste_orig, f, ensure_ascii=False, indent=2)
    
    with open(f"test/test_data_translated.json", 'w', encoding='utf-8') as f:
        json.dump(teste_trad, f, ensure_ascii=False, indent=2)
    
    with open(f"test/test_data_junto.json", 'w', encoding='utf-8') as f:
        json.dump(teste_junto, f, ensure_ascii=False, indent=2)
    
    # Definir nomes das colunas
    nomes_colunas = ['image', 'hate speech', 'inappropriate content', 'neither']
    
    # Salvar labels CSV
    print(" Salvando arquivos CSV...")
    
    # Treino
    treino_labels.columns = nomes_colunas
    treino_labels.to_csv(f"train/train_labels.csv", header=True, index=False)
    
    # Validação
    val_labels.columns = nomes_colunas
    val_labels.to_csv(f"validation/validation_labels.csv", header=True, index=False)
    
    # Teste
    teste_labels.columns = nomes_colunas
    teste_labels.to_csv(f"test/test_labels.csv", header=True, index=False)
    
    print("\n Divisão completa!")
    print(f" Arquivos salvos em: /")
    print("\n Resumo:")
    print(f"    train/ - {len(indices_treino)} amostras")
    print(f"    validation/ - {len(indices_validacao)} amostras") 
    print(f"    test/ - {len(indices_teste)} amostras")
    print(f"   Colunas com cabeçalho adicionadas aos CSVs: {', '.join(nomes_colunas)}")
    
    # Criar arquivo de resumo
    resumo = {
        "total_amostras": n_samples,
        "divisao": {
            "treino": {
                "quantidade": len(indices_treino),
                "porcentagem": len(indices_treino)/n_samples*100
            },
            "validacao": {
                "quantidade": len(indices_validacao),
                "porcentagem": len(indices_validacao)/n_samples*100
            },
            "teste": {
                "quantidade": len(indices_teste),
                "porcentagem": len(indices_teste)/n_samples*100
            }
        },
        "arquivos_criados": {
            "train": [
                "train_data.json",
                "train_data_translated.json", 
                "train_data_junto.json",
                "train_labels.csv"
            ],
            "validation": [
                "validation_data.json",
                "validation_data_translated.json",
                "validation_data_junto.json", 
                "validation_labels.csv"
            ],
            "test": [
                "test_data.json",
                "test_data_translated.json",
                "test_data_junto.json",
                "test_labels.csv"
            ]
        }
    }
    
    with open(f"resumo_divisao.json", 'w', encoding='utf-8') as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)
    
    return resumo

if __name__ == "__main__":
    dividir_dados()