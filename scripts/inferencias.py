#!/usr/bin/env python3
"""
Script de inferência comparativa para múltiplos fine-tunings SmolVLM.

Para cada modelo avaliado (texto, imagem, texto+descrição, completo), executa 4 modos:
 1. text
 2. image
 3. text+description
 4. text+description+image

Recomendações científicas:
- Use métricas principais apenas nos modos compatíveis com a modalidade treinada (in-domain).
- Modos fora do domínio servem apenas como análise de robustez / degradação.

Nomes genéricos de diretórios (ajuste depois para os caminhos reais dos checkpoints):
  CHECKPOINT_TEXT
  CHECKPOINT_IMAGE
  CHECKPOINT_TEXT_DESC
  CHECKPOINT_FULL

Cada diretório pode conter:
  - Modelo full fine-tuned (config + pesos) OU
  - Adapter LoRA (adapter_model.safetensors + adapter_config.json) usando base "HuggingFaceTB/SmolVLM-256M-Instruct".

Saídas:
  - resultados_inferencia.csv : métricas agregadas por modelo/modo
  - resultados_inferencia.json : mesma info em JSON + tempos
  - matrix_confusion_<modelo>_<modo>.csv : matriz de confusão por variante

Dependências: torch, transformers, peft, pandas, scikit-learn, Pillow, tqdm
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
	"text": "CHECKPOINT_TEXT",
	"image": "CHECKPOINT_IMAGE",
	"text_desc": "CHECKPOINT_TEXT_DESC",
	"full": "CHECKPOINT_FULL",
}

TEST_CSV = "test/dados_espanhol.csv"
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
		model = Idefics3ForConditionalGeneration.from_pretrained(
			model_dir,
			torch_dtype=DTYPE,
			device_map="auto",
			_attn_implementation="flash_attention_2"
		)
		print(f"[OK] Carregado modelo full de {model_dir}")
		return model
	except Exception:
		pass

	# Caso LoRA: carregar base e depois aplicar adapter
	try:
		base = Idefics3ForConditionalGeneration.from_pretrained(
			BASE_MODEL_ID,
			torch_dtype=DTYPE,
			device_map="auto",
			_attn_implementation="flash_attention_2"
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
		else:
			images.append(None)

	batch_inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(DEVICE)
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
		for i in range(0, len(rows), BATCH_SIZE):
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

