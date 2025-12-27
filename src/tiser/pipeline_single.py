import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
# Importa la tua classe dataset definita sopra
# from dataset_module import TiserFinetuningDataset

# --- 1. CONFIGURAZIONE ---
model_id = "Qwen/Qwen2.5-7B-Instruct" # Il modello usato nell'esempio del paper [cite: 125, 187]
data_path = "path/al/tuo/dataset.json"
output_dir = "./tiser_qwen_finetuned"

# --- 2. CARICAMENTO TOKENIZER ---
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# --- 3. PREPARAZIONE DATASET ---
train_dataset = TiserFinetuningDataset(data_path, tokenizer)

# --- 4. PREPARAZIONE DATA COLLATOR (Cruciale) ---
# Questo serve per calcolare la loss SOLO sulla risposta dell'assistente, ignorando il prompt.
# Qwen/Mistral usano formati specifici per separare le risposte.
# Esempio per Qwen2.5 che usa ChatML: la risposta inizia dopo "<|im_start|>assistant\n"
response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer
)

# --- 5. CARICAMENTO MODELLO ---
# Carichiamo in bfloat16 (formato standard per A100 citate nel paper [cite: 188])
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2" # Opzionale ma consigliato per velocità
)

# --- 6. CONFIGURAZIONE LORA ---
# Parametri standard per LoRA come citato implicitamente nel paper
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# --- 7. CONFIGURAZIONE TRAINING ---
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3, # Il paper non specifica, ma 3 è standard per SFT
    per_device_train_batch_size=4, # Aggiusta in base alla tua VRAM
    gradient_accumulation_steps=4,
    learning_rate=2e-4, # LR tipico per LoRA
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    bf16=True, # Raccomandato su GPU moderne
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)

# --- 8. INIZIALIZZAZIONE TRAINER ---
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field="text", # La chiave restituita dal tuo Dataset
    max_seq_length=4096, # I prompt TISER sono lunghi, serve spazio
    data_collator=collator, # Usa il collator per mascherare il prompt
    tokenizer=tokenizer,
    peft_config=peft_config,
    args=training_args,
    packing=False
)

# --- 9. TRAINING ---
print("Avvio del fine-tuning TISER...")
trainer.train()

trainer.save_model(output_dir)
print("Modello salvato.")