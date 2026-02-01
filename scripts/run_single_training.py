"""
FILE: run_single_training.py
SCOPO: Esegue il fine-tuning di Qwen2.5-7B (4-bit) utilizzando Unsloth per ottimizzare i tempi e la memoria.
       Specificamente progettato per dataset Chain-of-Thought (CoT) in formato JSONL.

COME RUNNARE:
    python scripts/run_single_training.py \
        --data data/processed/TISER_train_10pct.json \
        --output experiments/run_qwen_10pct
"""

import argparse
import os
import sys
import torch
import logging
import json
from pathlib import Path
from datasets import Dataset as HFDataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForLanguageModeling

# Ottimizzazione memoria PyTorch
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
logging.basicConfig(level=logging.INFO)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Percorso al file JSONL del dataset")
    parser.add_argument('--output', type=str, required=True, help="Cartella dove salvare i checkpoint e il modello finale")
    args = parser.parse_args()

    MAX_SEQ = 1024

    # 1. Caricamento Modello con Unsloth
    print("⏳ Caricamento modello Qwen2.5 in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        max_seq_length=MAX_SEQ,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model, 
        r=8, 
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth"
    )

    # 2. Preparazione Dataset (Caricamento Robusto JSONL)
    print(f"📂 Preparazione dataset da: {args.data}")
    processed_data = []
    with open(args.data, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                prompt = item['prompt'].strip()
                output = item['output'].strip()
                # Formato ChatML per Qwen2.5
                text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
                processed_data.append({"text": text})
            except Exception as e:
                continue

    dataset = HFDataset.from_list(processed_data)

    # 3. Configurazione Trainer con Checkpoint e Monitoring
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ,
        dataset_num_proc=2,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=5,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            
            # --- MONITORING & LOGGING ---
            logging_steps=1,             # Aggiorna la Loss ad ogni step
            eval_strategy="no",          # Non perdiamo tempo con la valutazione se non serve
            
            # --- CHECKPOINTING ---
            save_strategy="steps",       # Salva ogni X step
            save_steps=50,               # Salva un checkpoint ogni 50 iterazioni
            save_total_limit=2,          # Tiene solo gli ultimi 2 checkpoint (risparmia spazio disco)
            
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=args.output,
            report_to="none",
        ),
    )

    print(f"\n🚀 Training avviato su {len(dataset)} esempi!")
    print(f"📊 Puoi monitorare la Loss e l'ETA nella barra qui sotto.\n")
    
    trainer.train()

    # 4. Salvataggio Finale
    print(f"\n✅ Training completato! Salvataggio in corso in {args.output}...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("🎉 Tutto salvato con successo!")

if __name__ == "__main__":
    train()