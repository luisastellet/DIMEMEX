#!/usr/bin/env python3
"""
Script de inferência comparativa para múltiplos fine-tunings SmolVLM.

Executa:
1. Inferência IN-DOMAIN nos 4 modelos Fine-Tuned (FT).
2. Inferência dos 4 MODOS no Modelo Base (Zero-Shot).

Gera e salva:
1. Métricas agregadas (CSV e JSON)
2. Predições detalhadas (CSV)
3. Matrizes de Confusão (CSV e PNG)
"""

import os
import time
import json
from typing import List, Dict, Optional
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import numpy as np

# Dependências para plotagem
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("[WARN] Matplotlib/Seaborn não instalados. Matrizes de confusão serão salvas apenas em CSV.")

from transformers import AutoProcessor, Idefics3ForConditionalGeneration
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None  

# ===================== CONFIGURAÇÃO =====================
BASE_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

# --- Configuração FT ---
MODEL_DIRS = {
    "text": "/home/amandazirpolo/DIMEMEX/FT_text/SmolVLM_DIMEMEX_20251124_212719 ***",
    "image": "/home/amandazirpolo/DIMEMEX/FT_image/SmolVLM_DIMEMEX_20251125_133530 ***",
    "text_description": "/home/amandazirpolo/DIMEMEX/FT_text_description/SmolVLM_DIMEMEX_20251124_183536 ***",
    "text_description_image": "/home/amandazirpolo/DIMEMEX/FT_image_text_description/SmolVLM_DIMEMEX_20251127_191312 ***",
}

# --- Configuração BASE ---
BASE_MODEL_MODES = [
    "text",
    "image",
    "text_description",
    "text_description_image",
]
BASE_MODEL_NAME = "BASE_MODEL" # Nome que aparecerá nos arquivos de saída

TEST_CSV = "/home/amandazirpolo/DIMEMEX/test/dados_espanhol_teste.csv"
TEST_IMAGES_DIR = "/home/amandazirpolo/DIMEMEX/test_images"  

OUTPUT_DIR = "resultados_inferencia_smolvlm" 

LABELS = ["hate speech", "inappropriate content", "neither"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
GEN_MAX_NEW_TOKENS = 32
BATCH_SIZE = 8  
TEMPERATURE = 0.0 


# ===================== FUNÇÃO DE PLOTAGEM =====================

def plot_confusion_matrix_to_png(cm: np.ndarray, labels: List[str], path: str, title: str):
    if not PLOTTING_AVAILABLE:
        return
        
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=labels, 
        yticklabels=labels,
        cbar=False
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# ===================== UTILIDADES =====================
def safe_image_open(path: str) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), color="black")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "image_path" not in df.columns and "mage_path" in df.columns:
        df = df.rename(columns={"mage_path": "image_path"})
    for col in ["text", "description", "image_path", "label"]:
        if col not in df.columns:
            if col == "text":
                df[col] = ""
            elif col == "description":
                df[col] = ""
            elif col == "image_path":
                df[col] = ""
            elif col == "label":
                raise ValueError("CSV sem coluna 'label'.")
    return df

# Função para carregar modelos FINE-TUNED (original)
def load_model(model_dir: str, processor: AutoProcessor) -> Optional[Idefics3ForConditionalGeneration]:
    if not os.path.isdir(model_dir):
        print(f"[WARN] Diretório de modelo não encontrado: {model_dir}. Pulando.")
        return None
    try:
        has_flash = False
        try:
            import flash_attn  
            has_flash = True
        except Exception:
            has_flash = False

        kwargs = {"torch_dtype": DTYPE, "device_map": "auto"}
        if torch.cuda.is_available() and has_flash:
            kwargs["_attn_implementation"] = "flash_attention_2"
        else:
            if torch.cuda.is_available():
                print("[INFO] flash_attn não encontrado — usando implementação padrão de atenção.")

        model = Idefics3ForConditionalGeneration.from_pretrained(
            model_dir,
            **kwargs
        )
        print(f"[OK] Carregado modelo full de {model_dir}")
        return model
    except Exception:
        pass

    try:
        has_flash = False
        try:
            import flash_attn 
            has_flash = True
        except Exception:
            has_flash = False

        kwargs = {"torch_dtype": DTYPE, "device_map": "auto"}
        if torch.cuda.is_available() and has_flash:
            kwargs["_attn_implementation"] = "flash_attention_2"

        base = Idefics3ForConditionalGeneration.from_pretrained(
            BASE_MODEL_ID,
            **kwargs
        )
        if PeftModel is None:
            print(f"[ERRO] PEFT não instalado para aplicar adapter em {model_dir}")
            return None
        model = PeftModel.from_pretrained(base, model_dir)
        print(f"[OK] Carregado base + adapter de {model_dir}")
        return model
    except Exception as e:
        print(f"[ERRO] Falha ao carregar modelo/adapter de {model_dir}: {e}")
        return None

# Função para carregar o MODELO BASE (NOVA)
def load_base_model(processor: AutoProcessor) -> Idefics3ForConditionalGeneration:
    try:
        has_flash = False
        try:
            import flash_attn  
            has_flash = True
        except Exception:
            has_flash = False

        kwargs = {"torch_dtype": DTYPE, "device_map": "auto"}
        if torch.cuda.is_available() and has_flash:
            kwargs["_attn_implementation"] = "flash_attention_2"
        else:
            if torch.cuda.is_available():
                print("[INFO] flash_attn não encontrado — usando implementação padrão de atenção.")

        model = Idefics3ForConditionalGeneration.from_pretrained(
            BASE_MODEL_ID,
            **kwargs
        )
        print(f"[OK] Carregado modelo base: {BASE_MODEL_ID}")
        return model
    except Exception as e:
        raise SystemExit(f"[ERRO FATAL] Falha ao carregar o modelo base {BASE_MODEL_ID}: {e}")


def build_messages(row: Dict, mode: str) -> Dict:
    if mode == "text":
        prompt = (
            "Analice el TEXTO VISUAL. Clasifique este meme en: hate speech, inappropriate content, o neither.\n"
            f"TEXTO VISUAL: '{row['text']}'"
        )
        content = [{"type": "text", "text": prompt}]
    elif mode == "image":
        prompt = (
            "Analice la IMAGEN. Clasifique este meme en: hate speech, inappropriate content, o neither."
        )
        content = [{"type": "image"}, {"type": "text", "text": prompt}]
    elif mode == "text_description":
        prompt = (
            "Analice el TEXTO VISUAL y la DESCRIPCIÓN CONTEXTUAL. Clasifique en: hate speech, inappropriate content, o neither.\n"
            f"TEXTO VISUAL: '{row['text']}'\nDESCRIPCIÓN CONTEXTUAL: '{row['description']}'"
        )
        content = [{"type": "text", "text": prompt}]
    elif mode == "text_description_image":
        prompt = (
            "Analice la IMAGEN, el TEXTO VISUAL y la DESCRIPCIÓN CONTEXTUAL. Clasifique en: hate speech, inappropriate content, o neither.\n"
            f"TEXTO VISUAL: '{row['text']}'\nDESCRIPCIÓN CONTEXTUAL: '{row['description']}'"
        )
        content = [{"type": "image"}, {"type": "text", "text": prompt}]
    else:
        raise ValueError(f"Modo desconhecido: {mode}")

    messages = [
        {"role": "user", "content": content}
    ]
    return messages

def parse_prediction(output_text: str) -> str:
    text_low = output_text.lower()
    for lbl in LABELS:
        if lbl in text_low:
            return lbl
    if "hate" in text_low:
        return "hate speech"
    if "inappropriate" in text_low or "inaprop" in text_low:
        return "inappropriate content"
    return "neither"

def generate_batch(model, processor, batch_rows: List[Dict], mode: str) -> List[str]:
    texts = []
    images = []
    has_images_in_batch = False

    for row in batch_rows:
        messages = build_messages(row, mode)
        chat_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        texts.append(chat_text)

        if any(c.get("type") == "image" for c in messages[0]["content"]):
            img_path = row.get("image_path", "")
            # Assume que TEST_IMAGES_DIR é um caminho absoluto aqui, conforme a configuração
            if img_path and not os.path.isabs(img_path):
                 img_path = os.path.join(TEST_IMAGES_DIR, img_path)
            
            # Garante que o caminho exista (ou usa a imagem preta de fallback)
            images.append([safe_image_open(img_path) if os.path.exists(img_path) else safe_image_open("")])
            has_images_in_batch = True
        else:
            images.append([]) 

    kwargs = {"text": texts, "return_tensors": "pt", "padding": True}
    
    if has_images_in_batch:
        kwargs["images"] = images

    batch_inputs = processor(**kwargs).to(DEVICE)
    
    with torch.no_grad():
        generated = model.generate(
            **batch_inputs,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            do_sample=bool(TEMPERATURE > 0),
            temperature=TEMPERATURE if TEMPERATURE > 0 else None
        )
    preds = []
    for i, gen_ids in enumerate(generated):
        input_len = batch_inputs["input_ids"][i].shape[0]
        new_ids = gen_ids[input_len:]
        text_out = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        preds.append(parse_prediction(text_out))
    return preds

# Função de avaliação para modelos FINE-TUNED (IN-DOMAIN)
def evaluate_ft(model_name: str, model_dir: str, df: pd.DataFrame, processor: AutoProcessor) -> List[Dict]:
    mode = model_name 
    
    model = load_model(model_dir, processor)
    if model is None:
        return []
    model.eval()
    
    rows = df.to_dict(orient="records")
    start = time.time()
    preds = []
    gts = []
    prediction_data = [] 

    for i in tqdm(range(0, len(rows), BATCH_SIZE), desc=f"Inferência FT {model_name}/{mode}"):
        batch = rows[i:i+BATCH_SIZE]
        batch_preds = generate_batch(model, processor, batch, mode)
        
        for j, row in enumerate(batch):
            gt_label = row["label"]
            pred_label = batch_preds[j]
            
            preds.append(pred_label)
            gts.append(gt_label)
            
            prediction_data.append({
                "text": row["text"],
                "description": row["description"],
                "image_path": row["image_path"],
                "ground_truth": gt_label,
                "prediction": pred_label,
            })

    elapsed = time.time() - start
    avg_time = elapsed / max(1, len(rows))
    
    # --- 1. Salvar Predições Detalhadas ---
    df_predictions = pd.DataFrame(prediction_data)
    pred_path = os.path.join(OUTPUT_DIR, f"predicoes_{model_name}_{mode}.csv")
    df_predictions.to_csv(pred_path, index=False)
    print(f"\n[OK] Predições detalhadas salvas em {pred_path}")

    # --- 2. Calcular e Salvar Matriz de Confusão (CSV e PNG) ---
    cm = confusion_matrix(gts, preds, labels=LABELS)

    cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
    cm_csv_path = os.path.join(OUTPUT_DIR, f"matrix_confusion_{model_name}_{mode}.csv")
    cm_df.to_csv(cm_csv_path, index=True)
    
    cm_png_path = os.path.join(OUTPUT_DIR, f"matrix_confusion_{model_name}_{mode}.png")
    if PLOTTING_AVAILABLE:
        plot_confusion_matrix_to_png(
            cm, 
            LABELS, 
            cm_png_path, 
            title=f"Matriz de Confusão - {model_name.upper()} ({mode})"
        )
        print(f"[OK] Matriz de confusão em PNG salva em {cm_png_path}")
    else:
        cm_png_path = "N/A (Plotagem não disponível)"


    # --- 3. Calcular Métricas Agregadas ---
    report = classification_report(gts, preds, labels=LABELS, output_dict=True, zero_division=0)

    res = {
        "model": model_name,
        "model_dir": model_dir,
        "mode": mode,
        "samples": len(rows),
        "accuracy": accuracy_score(gts, preds),
        "f1_macro": f1_score(gts, preds, labels=LABELS, average="macro", zero_division=0),
        "precision_macro": precision_score(gts, preds, labels=LABELS, average="macro", zero_division=0),
        "recall_macro": recall_score(gts, preds, labels=LABELS, average="macro", zero_division=0),
        "time_total_sec": elapsed,
        "time_avg_sec": avg_time,
        "per_class": {lbl: report.get(lbl, {}) for lbl in LABELS},
        "support": {lbl: report.get(lbl, {}).get("support", 0) for lbl in LABELS},
        "confusion_matrix_path_csv": cm_csv_path, 
        "confusion_matrix_path_png": cm_png_path, 
        "predictions_path": pred_path,
    }
    
    print(f"[OK] {model_name} | {mode} -> F1 macro {res['f1_macro']:.4f} | tempo médio {avg_time:.4f}s")
    
    return [res] 

# Função de avaliação para o MODELO BASE (NOVA)
def evaluate_base(df: pd.DataFrame, processor: AutoProcessor, model_instance: Idefics3ForConditionalGeneration) -> List[Dict]:
    all_base_results = []
    
    rows = df.to_dict(orient="records")
    
    for mode in BASE_MODEL_MODES:
        model_name = BASE_MODEL_NAME
        start = time.time()
        preds = []
        gts = []
        prediction_data = [] 

        for i in tqdm(range(0, len(rows), BATCH_SIZE), desc=f"Inferência BASE {model_name}/{mode}"):
            batch = rows[i:i+BATCH_SIZE]
            batch_preds = generate_batch(model_instance, processor, batch, mode)
            
            for j, row in enumerate(batch):
                gt_label = row["label"]
                pred_label = batch_preds[j]
                
                preds.append(pred_label)
                gts.append(gt_label)
                
                prediction_data.append({
                    "text": row["text"],
                    "description": row["description"],
                    "image_path": row["image_path"],
                    "ground_truth": gt_label,
                    "prediction": pred_label,
                })

        elapsed = time.time() - start
        avg_time = elapsed / max(1, len(rows))
        
        # --- 1. Salvar Predições Detalhadas ---
        df_predictions = pd.DataFrame(prediction_data)
        # Caminho com o prefixo BASE_MODEL
        pred_path = os.path.join(OUTPUT_DIR, f"predicoes_{model_name}_{mode}.csv")
        df_predictions.to_csv(pred_path, index=False)
        print(f"\n[OK] Predições detalhadas (BASE) salvas em {pred_path}")

        # --- 2. Calcular e Salvar Matriz de Confusão (CSV e PNG) ---
        cm = confusion_matrix(gts, preds, labels=LABELS)

        cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
        cm_csv_path = os.path.join(OUTPUT_DIR, f"matrix_confusion_{model_name}_{mode}.csv")
        cm_df.to_csv(cm_csv_path, index=True)
        
        cm_png_path = os.path.join(OUTPUT_DIR, f"matrix_confusion_{model_name}_{mode}.png")
        if PLOTTING_AVAILABLE:
            plot_confusion_matrix_to_png(
                cm, 
                LABELS, 
                cm_png_path, 
                title=f"Matriz de Confusão - BASE ({mode})"
            )
            print(f"[OK] Matriz de confusão (BASE) em PNG salva em {cm_png_path}")
        else:
            cm_png_path = "N/A (Plotagem não disponível)"


        # --- 3. Calcular Métricas Agregadas ---
        report = classification_report(gts, preds, labels=LABELS, output_dict=True, zero_division=0)

        res = {
            "model": model_name,
            "model_dir": BASE_MODEL_ID, # O dir do modelo base é o ID
            "mode": mode,
            "samples": len(rows),
            "accuracy": accuracy_score(gts, preds),
            "f1_macro": f1_score(gts, preds, labels=LABELS, average="macro", zero_division=0),
            "precision_macro": precision_score(gts, preds, labels=LABELS, average="macro", zero_division=0),
            "recall_macro": recall_score(gts, preds, labels=LABELS, average="macro", zero_division=0),
            "time_total_sec": elapsed,
            "time_avg_sec": avg_time,
            "per_class": {lbl: report.get(lbl, {}) for lbl in LABELS},
            "support": {lbl: report.get(lbl, {}).get("support", 0) for lbl in LABELS},
            "confusion_matrix_path_csv": cm_csv_path, 
            "confusion_matrix_path_png": cm_png_path, 
            "predictions_path": pred_path,
        }
        
        print(f"[OK] {model_name} | {mode} -> F1 macro {res['f1_macro']:.4f} | tempo médio {avg_time:.4f}s")
        all_base_results.append(res)
        
    return all_base_results 


def main():
    # Cria o diretório de saída, se não existir
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"[SETUP] Criado diretório de saída: {OUTPUT_DIR}")
        
    if not os.path.exists(TEST_CSV):
        raise SystemExit(f"Arquivo de teste não encontrado: {TEST_CSV}")
    df = pd.read_csv(TEST_CSV)
    df = normalize_columns(df)

    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    all_results = []

    # --- 1. Avaliação dos Modelos FINE-TUNED (FT) - 4 inferências IN-DOMAIN ---
    print("\n\n##############################################")
    print("## INÍCIO DA AVALIAÇÃO: MODELOS FINE-TUNED ##")
    print("##############################################")
    for key, mdir in MODEL_DIRS.items():
        print(f"\n=== Avaliando modelo IN-DOMAIN: '{key}' ({mdir}) ===")
        results = evaluate_ft(key, mdir, df, processor)
        all_results.extend(results)

    # --- 2. Avaliação do Modelo BASE - 4 inferências para cada modo ---
    print("\n\n################################################")
    print("## INÍCIO DA AVALIAÇÃO: MODELO BASE (4 MODOS) ##")
    print("################################################")
    
    # Carrega o modelo base uma única vez
    base_model = load_base_model(processor)
    
    # Executa a avaliação do modelo base nos 4 modos
    base_results = evaluate_base(df, processor, base_model)
    all_results.extend(base_results)


    if not all_results:
        print("\nNenhum resultado gerado (model dirs ausentes?).")
        return

    # --- 3. Salvamento dos Resultados Finais Agregados (FT + BASE) ---

    # Flatten para CSV (métricas agregadas)
    flat_rows = []
    for r in all_results:
        base = {k: v for k, v in r.items() if k not in ("per_class", "support")}
        for lbl in LABELS:
            pc = r["per_class"].get(lbl, {})
            flat_rows.append({
                **base,
                "label": lbl,
                "precision_label": pc.get("precision", 0),
                "recall_label": pc.get("recall", 0),
                "f1_label": pc.get("f1-score", 0),
                "support_label": pc.get("support", 0),
            })
    df_out = pd.DataFrame(flat_rows)
    
    # Salva os arquivos finais dentro do diretório de saída
    final_csv_path = os.path.join(OUTPUT_DIR, "resultados_inferencia_FT_e_BASE.csv")
    final_json_path = os.path.join(OUTPUT_DIR, "resultados_inferencia_FT_e_BASE.json")
    
    df_out.to_csv(final_csv_path, index=False)
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
        
    print(f"\n[FINAL] Métricas agregadas (FT + BASE) salvas em {final_csv_path} e {final_json_path}")

if __name__ == "__main__":
    main()