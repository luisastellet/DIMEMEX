#!/usr/bin/env python3
import os
import csv
import glob
import json
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

BETTER_PATH = os.path.join(ROOT_DIR, 'test_better_cases.csv')
PREDICOES_DIR = os.path.join(ROOT_DIR, 'resultados_inferencia_smolvlm')
OUT_DIR = os.path.join(ROOT_DIR, 'analise_melhores')

MODES = [
    'text_description_image',
    'text_description',
    'image',
    'text',
]

def ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)

def carregar_subset(path):
    """Carrega MEME-IDs do subset (coluna MEME-ID em test_better_cases.csv)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'Arquivo não encontrado: {path}')
    ids = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if 'MEME-ID' not in reader.fieldnames:
            raise ValueError("CSV sem coluna 'MEME-ID'.")
        for row in reader:
            mid = row['MEME-ID'].strip()
            if mid:
                ids.append(mid)
    return ids

def listar_predicoes(dir_path):
    pattern = os.path.join(dir_path, 'predicoes_*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'Nenhum arquivo encontrado em {pattern}')
    return files

def parse_model_mode(filename_no_ext):
    # Remove prefixo se presente
    if filename_no_ext.startswith('predicoes_'):
        name_part = filename_no_ext[len('predicoes_'):]
    else:
        name_part = filename_no_ext
    # Tenta casar modos pelo final
    for mode in sorted(MODES, key=len, reverse=True):
        if name_part.endswith(mode):
            prefix = name_part[:-len(mode)].rstrip('_')
            model = prefix if prefix else mode  
            return model, mode
    return name_part, 'desconhecido'

def ler_predicoes(path):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        needed = {'image_path','ground_truth','prediction'}
        if not needed.issubset(set(reader.fieldnames)):
            raise ValueError(f'Arquivo {path} sem colunas necessárias {needed}')
        rows = []
        for row in reader:
            rows.append({
                'image_path': row['image_path'].strip(),
                'ground_truth': row['ground_truth'].strip(),
                'prediction': row['prediction'].strip(),
            })
    return rows

def avaliar(predicoes, subset_ids):
    filtradas = [r for r in predicoes if r['image_path'] in subset_ids]
    total = len(filtradas)
    correct = sum(1 for r in filtradas if r['ground_truth']==r['prediction'])
    errors = total - correct
    pairs = Counter((r['ground_truth'], r['prediction']) for r in filtradas)
    gt_dist = Counter(r['ground_truth'] for r in filtradas)
    pred_dist = Counter(r['prediction'] for r in filtradas)
    return {
        'total': total,
        'correct': correct,
        'errors': errors,
        'accuracy': (correct/total*100.0) if total else 0.0,
        'error_rate': (errors/total*100.0) if total else 0.0,
        'pairs': pairs,
        'gt_dist': gt_dist,
        'pred_dist': pred_dist,
    }

def salvar_csv_resumo_por_arquivo(rows):
    path = os.path.join(OUT_DIR, 'resumo_por_arquivo.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        fn = ['arquivo','model','mode','total','correct','errors','accuracy','error_rate']
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fn})
    return path

def salvar_csv_detalhes_pares(rows):
    path = os.path.join(OUT_DIR, 'detalhes_pares.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        fn = ['arquivo','model','mode','ground_truth','prediction','count','pct_subset','pct_errors']
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path

def salvar_agregado_base_model(model_group):
    """Gera CSV agregado para BASE_MODEL com todas as modalidades."""
    if 'BASE_MODEL' not in model_group:
        return None
    
    entries = model_group['BASE_MODEL']
    path = os.path.join(OUT_DIR, 'agregado_base_model.csv')
    
    with open(path, 'w', newline='', encoding='utf-8') as f:
        fn = ['model', 'mode', 'total', 'correct', 'errors', 'accuracy', 'error_rate']
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        
        # Linhas individuais por modo
        for e in entries:
            w.writerow({k: e[k] for k in fn})
        
        # Linha agregada
        total = sum(e['total'] for e in entries)
        correct = sum(e['correct'] for e in entries)
        errors = total - correct
        w.writerow({
            'model': 'BASE_MODEL',
            'mode': 'AGREGADO',
            'total': total,
            'correct': correct,
            'errors': errors,
            'accuracy': (correct/total*100.0) if total else 0.0,
            'error_rate': (errors/total*100.0) if total else 0.0,
        })
    
    return path

def salvar_agregado_fine_tuning(model_group):
    """Gera CSV agregado para todos os modelos de Fine-Tuning."""
    ft_models = ['text', 'image', 'text_description', 'text_description_image']
    all_entries = []
    
    for model in ft_models:
        if model in model_group:
            all_entries.extend(model_group[model])
    
    if not all_entries:
        return None
    
    path = os.path.join(OUT_DIR, 'agregado_fine_tuning.csv')
    
    with open(path, 'w', newline='', encoding='utf-8') as f:
        fn = ['model', 'mode', 'total', 'correct', 'errors', 'accuracy', 'error_rate']
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        
        # Linhas individuais de todos os FTs
        for e in all_entries:
            w.writerow({k: e[k] for k in fn})
        
        # Linha agregada de todos os FTs
        total = sum(e['total'] for e in all_entries)
        correct = sum(e['correct'] for e in all_entries)
        errors = total - correct
        w.writerow({
            'model': 'FINE_TUNING',
            'mode': 'AGREGADO',
            'total': total,
            'correct': correct,
            'errors': errors,
            'accuracy': (correct/total*100.0) if total else 0.0,
            'error_rate': (errors/total*100.0) if total else 0.0,
        })
    
    return path

def salvar_csv_por_modelo(model_map):
    created = []
    for model, entries in model_map.items():
        path = os.path.join(OUT_DIR, f'modelo_{model}_resumo.csv')
        with open(path, 'w', newline='', encoding='utf-8') as f:
            fn = ['model','mode','total','correct','errors','accuracy','error_rate']
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            for e in entries:
                w.writerow({k: e[k] for k in fn})
            total = sum(e['total'] for e in entries)
            correct = sum(e['correct'] for e in entries)
            errors = total - correct
            w.writerow({
                'model': model,
                'mode': 'AGREGADO',
                'total': total,
                'correct': correct,
                'errors': errors,
                'accuracy': (correct/total*100.0) if total else 0.0,
                'error_rate': (errors/total*100.0) if total else 0.0,
            })
        created.append(path)
    return created

def main():
    ensure_out()
    subset_ids = carregar_subset(BETTER_PATH)
    pred_files = listar_predicoes(PREDICOES_DIR)

    resumo_rows = []
    pares_rows = []
    model_group = defaultdict(list)

    for pf in pred_files:
        base = os.path.basename(pf).replace('.csv','')
        model, mode = parse_model_mode(base)
        predicoes = ler_predicoes(pf)
        aval = avaliar(predicoes, set(subset_ids))

        resumo = {
            'arquivo': base,
            'model': model,
            'mode': mode,
            'total': aval['total'],
            'correct': aval['correct'],
            'errors': aval['errors'],
            'accuracy': aval['accuracy'],
            'error_rate': aval['error_rate'],
        }
        resumo_rows.append(resumo)
        model_group[model].append(resumo)

        # Pares
        for (gt,pred), cnt in aval['pairs'].items():
            pares_rows.append({
                'arquivo': base,
                'model': model,
                'mode': mode,
                'ground_truth': gt,
                'prediction': pred,
                'count': cnt,
                'pct_subset': (cnt/aval['total']*100.0) if aval['total'] else 0.0,
                'pct_errors': (cnt/aval['errors']*100.0) if aval['errors'] and gt!=pred else (100.0 if gt==pred and aval['correct'] else 0.0),
            })

    path_resumo = salvar_csv_resumo_por_arquivo(resumo_rows)
    path_pares = salvar_csv_detalhes_pares(pares_rows)
    path_base = salvar_agregado_base_model(model_group)
    path_ft = salvar_agregado_fine_tuning(model_group)

    json_path = os.path.join(OUT_DIR, 'analise_completa.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'subset_size': len(subset_ids),
            'resumo_por_arquivo': resumo_rows,
            'pares': pares_rows,
            'modelos': {m: v for m,v in model_group.items()},
        }, f, ensure_ascii=False, indent=2)

    print(f"Subset de melhores casos: {len(subset_ids)} memes.")
    for r in resumo_rows:
        print(f"- {r['arquivo']}: {r['accuracy']:.2f}% acurácia (acertos={r['correct']}/{r['total']})")
    print("\nArquivos gerados:")
    print(f"  {path_resumo}\n  {path_pares}")
    if path_base:
        print(f"  {path_base}")
    if path_ft:
        print(f"  {path_ft}")
    print(f"  {json_path}")

if __name__ == '__main__':
    main()
