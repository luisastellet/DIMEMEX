import os
import glob
import csv

try:
    import matplotlib
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"Erro: {e}")
    print("Instale matplotlib: pip3 install matplotlib")
    exit(1)

def plot_confusion_matrix(csv_path: str, output_path: str) -> None:

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    labels = rows[0][1:] 
    cm = []
    for row in rows[1:]:
        cm.append([int(val) for val in row[1:]])
    
    cm = np.array(cm)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', rotation=270, labelpad=30, fontsize=22, fontweight='bold')
    cbar.ax.tick_params(labelsize=20)
    
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=20, fontweight='bold')
    ax.set_yticklabels(labels, fontsize=20, fontweight='bold')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, str(cm[i, j]),
                          ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white",
                          fontsize=24, fontweight='bold')
    
    ax.set_xlabel('Predicted', fontsize=22, fontweight='bold')
    ax.set_ylabel('True', fontsize=22, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Gerado: {output_path}")

def main():
    if os.path.isdir("matriz_confusao"):
        search_dir = "matriz_confusao"
        os.chdir(search_dir)
        print(f"Trabalhando no diretório: {search_dir}/\n")
    else:
        search_dir = "."
        print(f"Trabalhando no diretório atual.\n")
    
    all_csv_files = glob.glob("matrix_confusion_*.csv")
    csv_files = [f for f in all_csv_files if "***" in f]
    
    if not csv_files:
        print("Nenhum arquivo matrix_confusion_**** .csv encontrado.")
        print(f"Diretório atual: {os.getcwd()}")
        return
    
    print(f"Encontrados {len(csv_files)} arquivos CSV de matriz de confusão.\n")
    
    success_count = 0
    for csv_path in sorted(csv_files):
        png_path = csv_path.replace('.csv', '.png')
        
        try:
            plot_confusion_matrix(csv_path, png_path)
            success_count += 1
        except Exception as e:
            print(f"✗ Erro ao processar {csv_path}: {e}")
    
    print(f"\nGeradas {success_count}/{len(csv_files)} visualizações PNG.")

if __name__ == "__main__":
    main()
