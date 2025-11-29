#!/usr/bin/env python3
"""
Script de inferência comparativa para múltiplos fine-tunings SmolVLM.
# ... (descrição e dependências omitidas para brevidade) ...
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

from transformers import AutoProcessor, Idefics3ForConditionalGeneration
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None  # Se não estiver instalado

# ===================== CONFIGURAÇÃO =====================
BASE_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

# Diretórios genéricos (substitua pelos reais)
MODEL_DIRS = {
    "text": "/home/amandazirpolo/DIMEMEX/FT_text/SmolVLM_DIMEMEX_20251124_212719 ***",
    "image": "/home/amandazirpolo/DIMEMEX/FT_image/SmolVLM_DIMEMEX_20251125_133530 ***",
    "text_desc": "/home/amandazirpolo/DIMEMEX/FT_text_description/SmolVLM_DIMEMEX_20251124_183536 ***",
    "full": "/home/amandazirpolo/DIMEMEX/FT_image_text_description/SmolVLM_DIMEMEX_20251127_191312 ***",
}

TEST_CSV = "/home/amandazirpolo/DIMEMEX/test/dados_espanhol_teste.csv"
TEST_IMAGES_DIR = "test_images"  # prefixo para image_path se necessário

LABELS = ["hate speech", "inappropriate content", "neither"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
GEN_MAX_NEW_TOKENS = 32
BATCH_SIZE = 8  # aumentar conforme VRAM
TEMPERATURE = 0.0  # geração determinística

MODES = [
    "text",
    "image",
    "text_description",
    "text_description_image",
]

# ===================== UTILIDADES =====================
def safe_image_open(path: str) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), color="black")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Corrigir typo mage_path -> image_path
    if "image_path" not in df.columns and "mage_path" in df.columns:
        df = df.rename(columns={"mage_path": "image_path"})
    # Garantir colunas relevantes
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

def load_model(model_dir: str, processor: AutoProcessor) -> Optional[Idefics3ForConditionalGeneration]:
    if not os.path.isdir(model_dir):
        print(f"[WARN] Diretório de modelo não encontrado: {model_dir}. Pulando.")
        return None
    # Tenta carregar como modelo já full fine-tuned primeiro
    try:
        # Ativa FlashAttention2 somente se o pacote estiver instalado e houver GPU
        has_flash = False
        try:
            import flash_attn  # type: ignore
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

    # Caso LoRA: carregar base e depois aplicar adapter
    try:
        # Mesma lógica para base (LoRA): só ativa FlashAttention2 se disponível
        has_flash = False
        try:
            import flash_attn  # type: ignore
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

def build_messages(row: Dict, mode: str) -> Dict:
    # Monta mensagens estilo chat conforme modalidade.
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
    # fallback heurística simples
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
            # concatenar diretório se não incluir separador
            if img_path and not os.path.isfile(img_path):
                candidate = os.path.join(TEST_IMAGES_DIR, img_path)
                img_path = candidate
            images.append([safe_image_open(img_path)])
            has_images_in_batch = True
        else:
            # Correção anterior: usar lista vazia [] em vez de None
            images.append([]) 

    # CORREÇÃO: Passar o argumento 'images' para o processador SOMENTE se
    # pelo menos uma amostra no batch atual realmente tiver imagens.
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
    # Decodificar somente novos tokens após input_ids
    preds = []
    for i, gen_ids in enumerate(generated):
        input_len = batch_inputs["input_ids"][i].shape[0]
        new_ids = gen_ids[input_len:]
        text_out = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        preds.append(parse_prediction(text_out))
    return preds

def evaluate(model_name: str, model_dir: str, modes: List[str], df: pd.DataFrame, processor: AutoProcessor) -> List[Dict]:
    model = load_model(model_dir, processor)
    if model is None:
        return []
    model.eval()
    results = []
    rows = df.to_dict(orient="records")
    for mode in modes:
        start = time.time()
        preds = []
        gts = []
        for i in tqdm(range(0, len(rows), BATCH_SIZE), desc=f"Inferência {model_name}/{mode}"):
            batch = rows[i:i+BATCH_SIZE]
            batch_preds = generate_batch(model, processor, batch, mode)
            preds.extend(batch_preds)
            gts.extend([r["label"] for r in batch])
        elapsed = time.time() - start
        avg_time = elapsed / max(1, len(rows))
        report = classification_report(gts, preds, labels=LABELS, output_dict=True, zero_division=0)
        cm = confusion_matrix(gts, preds, labels=LABELS)

        # Salvar matriz confusão
        cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
        cm_path = f"matrix_confusion_{model_name}_{mode}.csv"
        cm_df.to_csv(cm_path, index=True)

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
            "confusion_matrix_path": cm_path,
        }
        results.append(res)
        print(f"[OK] {model_name} | {mode} -> F1 macro {res['f1_macro']:.4f} | tempo médio {avg_time:.4f}s")
    return results

def main():
    if not os.path.exists(TEST_CSV):
        raise SystemExit(f"Arquivo de teste não encontrado: {TEST_CSV}")
    df = pd.read_csv(TEST_CSV)
    df = normalize_columns(df)

    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

    all_results = []
    for key, mdir in MODEL_DIRS.items():
        print(f"\n=== Avaliando modelo '{key}' ({mdir}) ===")
        results = evaluate(key, mdir, MODES, df, processor)
        all_results.extend(results)

    if not all_results:
        print("Nenhum resultado gerado (model dirs ausentes?).")
        return

    # Flatten para CSV
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
    df_out.to_csv("resultados_inferencia.csv", index=False)
    with open("resultados_inferencia.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("\n[FINAL] Métricas salvas em resultados_inferencia.csv / resultados_inferencia.json")

if __name__ == "__main__":
    main()