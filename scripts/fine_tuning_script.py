# !pip install accelerate datasets peft bitsandbytes tensorboard torch transformers pandas
# !pip install flash-attn --no-build-isolation

import os
# Garante que o script veja apenas 1 GPU para evitar erro de DataParallel com QLoRA
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoProcessor, 
    BitsAndBytesConfig, 
    Idefics3ForConditionalGeneration, 
    TrainingArguments, 
    Trainer, 
    EarlyStoppingCallback,
    TrainerCallback
)
from datasets import Dataset
from PIL import Image
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import time

# Hiperparâmetros de Treino 
BATCH_SIZE = 8
EPOCHS = 8
LR = 1e-3
EVAL_STEPS = 50  # Avalia, salva e loga a cada 50 passos

USE_LORA = False
USE_QLORA = True

model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"

# === Caminhos ===
CSV_TRAIN = "train/dados_espanhol_balanceado.csv"
CSV_VAL   = "validation/dados_espanhol.csv"
CSV_TEST  = "test/dados_espanhol.csv"

TRAIN_IMAGES_DIR = "train_images"
VAL_IMAGES_DIR   = "validation_images"
TEST_IMAGES_DIR  = "test_images"

# Carregamento de Dados 
print("📊 Carregando datasets...")
df_train = pd.read_csv(CSV_TRAIN)
df_val   = pd.read_csv(CSV_VAL)
df_test  = pd.read_csv(CSV_TEST)

# Ajuste de caminhos
df_train["image_path"] = df_train["image_path"].apply(lambda x: os.path.join(TRAIN_IMAGES_DIR, x))
df_val["image_path"]   = df_val["image_path"].apply(lambda x: os.path.join(VAL_IMAGES_DIR, x))
df_test["image_path"]  = df_test["image_path"].apply(lambda x: os.path.join(TEST_IMAGES_DIR, x))

ds_train = Dataset.from_pandas(df_train.reset_index(drop=True))
ds_val = Dataset.from_pandas(df_val.reset_index(drop=True))

def load_image(example):
    try:
        example["image"] = Image.open(example["image_path"]).convert("RGB")
    except Exception as e:
        print(f"Erro imagem: {example['image_path']}")
        example["image"] = Image.new('RGB', (224, 224), color='black')
    return example

print("🖼️  Processando imagens (map)...")
ds_train = ds_train.map(load_image)
ds_val = ds_val.map(load_image)
print("✅ Imagens prontas!")

processor = AutoProcessor.from_pretrained(model_id)

# Configuração do Modelo
if USE_QLORA or USE_LORA:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
        init_lora_weights="gaussian",
        inference_mode=False
    )
    
    bnb_config = None
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        _attn_implementation="flash_attention_2",
        device_map={"": 0} 
    )
    
    model.add_adapter(lora_config)
    model.enable_adapters()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    print("🔄 Forçando camadas LoRA para a GPU...")
    for name, module in model.named_modules():
        if "lora_" in name:
            module.to("cuda")
            
    dtype_to_use = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for name, param in model.named_parameters():
        if "lora" in name and param.requires_grad:
            param.data = param.data.to(device="cuda", dtype=dtype_to_use)
else:
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, _attn_implementation="flash_attention_2", device_map="auto"
    )

# Collate Function
image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")]

def collate_fn(examples):
    texts = []
    images = []
    prompt = "Clasifique este meme: hate speech, inappropriate content o neither."

    for example in examples:
        image = example["image"]
        if image.mode != 'RGB': image = image.convert('RGB')
        answer = str(example["label"]) 

        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]
        
        text = processor.apply_chat_template(messages, add_generation_prompt=False)
        texts.append(text.strip())
        images.append([image])

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    batch["labels"] = labels
    return batch

# Callbacks e Configuração de Treino 
model_name = model_id.split("/")[-1]
output_dir_checkpoints = f"./{model_name}-checkpoints"
os.makedirs(output_dir_checkpoints, exist_ok=True)

# Callback para logar em arquivo txt
class FileLoggingCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        with open(self.log_path, 'w') as f:
            f.write(f"Iniciando treinamento em {datetime.now()}\n")
            f.write("-" * 60 + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            loss = logs.get("loss", "N/A")
            eval_loss = logs.get("eval_loss", "N/A")
            lr = logs.get("learning_rate", "N/A")
            epoch = logs.get("epoch", "N/A")
            
            # Monta string dependendo do que tem no log (treino ou validação)
            log_str = f"[Epoch: {epoch:.2f}] [Step: {step}] "
            if loss != "N/A": log_str += f"Train Loss: {loss:.4f} "
            if eval_loss != "N/A": log_str += f"| Val Loss: {eval_loss:.4f} "
            if lr != "N/A": log_str += f"| LR: {lr:.2e}"
            log_str += "\n"
            
            with open(self.log_path, 'a') as f:
                f.write(log_str)

training_args = TrainingArguments(
    num_train_epochs=EPOCHS,                 
    per_device_train_batch_size=BATCH_SIZE,  
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,    
    learning_rate=LR,                        

    # Estratégia Unificada: Loga, Salva e Avalia nos mesmos passos
    logging_steps=25,
    eval_strategy="steps",      
    eval_steps=EVAL_STEPS,      
    save_strategy="steps",      
    save_steps=EVAL_STEPS,

    # Early Stopping Config
    load_best_model_at_end=True, 
    metric_for_best_model="eval_loss", 
    greater_is_better=False,

    warmup_steps=50,
    weight_decay=0.01,
    save_total_limit=1, 
    optim="paged_adamw_8bit",
    bf16=True,
    output_dir=output_dir_checkpoints,
    report_to="tensorboard",
    remove_unused_columns=False,
    gradient_checkpointing=True
)

log_txt_path = os.path.join(output_dir_checkpoints, "training_progress_log.txt")

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=ds_train,
    eval_dataset=ds_val, 
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01),
        FileLoggingCallback(log_txt_path)
    ]
)

print(f"🚀 Iniciando treinamento! Log em: {log_txt_path}")
training_start_time = time.time()
train_result = trainer.train()
training_end_time = time.time()
training_duration = training_end_time - training_start_time

print(f"\n⏱️  Tempo total de treinamento: {training_duration/60:.2f} minutos ({training_duration:.2f} segundos)")

# Salvamento Final e Gráficos Completos 
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
final_output_dir = f"./SmolVLM_DIMEMEX_{timestamp}"
os.makedirs(final_output_dir, exist_ok=True)

print(f"💾 Salvando modelo final em {final_output_dir}...")
trainer.save_model(final_output_dir)
processor.save_pretrained(final_output_dir)

# Extração e Salvamento de Métricas
history = trainer.state.log_history

# Separa dados de treino e validação para gráficos
train_steps, train_losses = [], []
eval_steps, eval_losses = [], []

for entry in history:
    if 'loss' in entry:
        train_steps.append(entry['step'])
        train_losses.append(entry['loss'])
    if 'eval_loss' in entry:
        eval_steps.append(entry['step'])
        eval_losses.append(entry['eval_loss'])

# Plotagem Melhorada
plt.figure(figsize=(12, 6))
if train_steps:
    plt.plot(train_steps, train_losses, label='Training Loss', color='blue', alpha=0.6)
if eval_steps:
    plt.plot(eval_steps, eval_losses, label='Validation Loss', color='red', linewidth=2, marker='o')

plt.title(f'Training vs Validation Loss\n(Best Eval Loss: {min(eval_losses) if eval_losses else "N/A"})')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig(os.path.join(final_output_dir, "loss_curve.png"))
print(f"📈 Gráfico de Loss salvo em {final_output_dir}/loss_curve.png")

# Salvar JSON de resumo
summary = {
    "final_train_loss": train_losses[-1] if train_losses else None,
    "best_eval_loss": min(eval_losses) if eval_losses else None,
    "total_steps": trainer.state.global_step,
    "epoch": trainer.state.epoch,
    "training_runtime": train_result.metrics.get("train_runtime"),
    "hyperparameters": {
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "lora": USE_LORA or USE_QLORA
    }
}

with open(os.path.join(final_output_dir, "training_summary.json"), "w") as f:
    json.dump(summary, f, indent=4)

with open(os.path.join(final_output_dir, "full_history.json"), "w") as f:
    json.dump(history, f, indent=4)

print(f"{'='*50}")
print("📊 Gerando Classification Report no dataset de VALIDAÇÃO...")
print(f"{'='*50}\n")

# Função para fazer predições
def predict_single_sample(image_path, label_text):
    """Faz predição em uma única imagem"""
    try:
        image = Image.open(image_path).convert("RGB")
        prompt = "Clasifique este meme: hate speech, inappropriate content o neither."
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=[image], return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=15,
                do_sample=False,
                num_beams=1,
                pad_token_id=processor.tokenizer.pad_token_id
            )
        
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = processor.decode(generated_ids, skip_special_tokens=True).strip().lower()
        
        # Mapear resposta
        if "hate speech" in response or "hate" in response:
            return "hate speech"
        elif "inappropriate" in response:
            return "inappropriate content"
        elif "neither" in response:
            return "neither"
        else:
            return response  # Retorna resposta bruta se não mapear
            
    except Exception as e:
        print(f"Erro ao processar {image_path}: {e}")
        return "error"

# Fazer predições no conjunto de validação
print("🔮 Gerando predições no dataset de validação...")
predictions = []
true_labels = []

for idx in tqdm(range(len(ds_val)), desc="Validação"):
    row = df_val.iloc[idx]
    pred = predict_single_sample(row['image_path'], row['label'])
    predictions.append(pred)
    true_labels.append(row['label'])

# Filtrar predições válidas
valid_labels = ["hate speech", "inappropriate content", "neither"]
valid_indices = [i for i, p in enumerate(predictions) if p in valid_labels]

if len(valid_indices) < len(predictions):
    print(f"\n⚠️  {len(predictions) - len(valid_indices)} predições inválidas removidas")
    predictions = [predictions[i] for i in valid_indices]
    true_labels = [true_labels[i] for i in valid_indices]

# Gerar classification report
print("\n" + "="*70)
print("CLASSIFICATION REPORT - VALIDAÇÃO")
print("="*70 + "\n")

report_text = classification_report(
    true_labels, 
    predictions, 
    labels=valid_labels,
    target_names=valid_labels,
    digits=4,
    zero_division=0
)

print(report_text)

# Métricas resumidas
metrics_summary = {
    "accuracy": accuracy_score(true_labels, predictions),
    "f1_weighted": f1_score(true_labels, predictions, labels=valid_labels, average='weighted', zero_division=0),
    "f1_macro": f1_score(true_labels, predictions, labels=valid_labels, average='macro', zero_division=0),
    "precision_weighted": precision_score(true_labels, predictions, labels=valid_labels, average='weighted', zero_division=0),
    "recall_weighted": recall_score(true_labels, predictions, labels=valid_labels, average='weighted', zero_division=0),
    "total_samples": len(true_labels),
    "training_duration_seconds": training_duration,
    "training_duration_minutes": training_duration / 60
}

print("\nSummary Metrics:")
print(json.dumps(metrics_summary, indent=4))

# Salvar classification report
report_file = os.path.join(final_output_dir, "classification_report_validation.txt")
with open(report_file, 'w') as f:
    f.write("="*70 + "\n")
    f.write("CLASSIFICATION REPORT - VALIDATION SET\n")
    f.write("="*70 + "\n\n")
    f.write(report_text)
    f.write("\n\nSummary Metrics:\n")
    f.write(json.dumps(metrics_summary, indent=4))
    f.write(f"\n\nTraining Duration: {training_duration/60:.2f} minutes\n")

print(f"\n✅ Classification report salvo em: {report_file}")

# Salvar predições detalhadas
predictions_df = pd.DataFrame({
    'image_path': [df_val.iloc[i]['image_path'] for i in valid_indices],
    'true_label': true_labels,
    'predicted_label': predictions,
    'correct': [t == p for t, p in zip(true_labels, predictions)]
})

predictions_file = os.path.join(final_output_dir, "validation_predictions.csv")
predictions_df.to_csv(predictions_file, index=False)
print(f"✅ Predições detalhadas salvas em: {predictions_file}")

# Atualizar summary com métricas de classificação
summary["classification_metrics"] = metrics_summary
with open(os.path.join(final_output_dir, "training_summary.json"), "w") as f:
    json.dump(summary, f, indent=4)

print(f"{'='*50}")
print(f"✅ Treinamento finalizado com sucesso!")
print(f"📁 Todos os arquivos salvos em: {final_output_dir}")
print(f"{'='*50}")