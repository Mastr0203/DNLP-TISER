"""
DEPRECATED: This file is deprecated and kept for backward compatibility only.

This module has been refactored and moved to a CLI script with improved
functionality and better organization. Please use the new script instead:

    scripts/run_single_training.py

The new script provides:
- Command-line interface with argparse
- Automatic device detection (CUDA, MPS, CPU)
- Configurable training hyperparameters via CLI arguments
- Better error handling and logging
- Consistent style with other project scripts

Example usage of the new script:
    python scripts/run_single_training.py \\
        --data src/data/processed/TISER_train_10pct.json \\
        --output experiments/qwen_finetuned \\
        --epochs 3 \\
        --batch-size 1 \\
        --learning-rate 2e-4

This file may be removed in future versions.
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# Importa le classi dal file dataset corretto
from src.data.tiser_dataset import TiserDataset, load_tiser_file

# --- 0. CONFIGURAZIONE AMBIENTE ---
# Disabilita il parallelismo dei tokenizer per evitare deadlock su Mac/Linux
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configurazione Percorsi
# Usa percorsi relativi o assoluti corretti
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Aggiusta questo percorso se necessario per puntare al tuo JSON
DATA_PATH = "/Users/gabrielemerlino/scrivania/DNLP_HW/DNLP-TISER/src/data/processed/TISER_train_20pct.json"
OUTPUT_DIR = "./experiments/tiser_qwen_finetuned"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def main():
    print("=" * 80)
    print("WARNING: This script is DEPRECATED!")
    print("=" * 80)
    print("Please use the new CLI script instead:")
    print("  python scripts/run_single_training.py --data <path> --output <path>")
    print("=" * 80)
    print()
    
    print(f"--- Inizio Pipeline di Training su {MODEL_ID} ---")

    # --- 1. RILEVAMENTO DEVICE ---
    # Controlla se siamo su Mac (MPS), Nvidia (CUDA) o CPU
    if torch.backends.mps.is_available():
        device = "mps"
        print("Device rilevato: Apple Silicon (MPS). Ottimizzazioni Mac attive.")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Device rilevato: NVIDIA GPU (CUDA).")
    else:
        device = "cpu"
        print("Attenzione: Device CPU rilevato. Il training sarà molto lento.")

    # --- 2. CARICAMENTO TOKENIZER ---
    print("Caricamento Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Qwen usa token specifici, assicuriamoci che il pad sia eos o uno dedicato
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 3. PREPARAZIONE DATASET ---
    print(f"Caricamento dati da: {DATA_PATH}")
    # Carichiamo gli esempi grezzi e creiamo il dataset PyTorch
    raw_examples = load_tiser_file(DATA_PATH)
    train_dataset = TiserDataset(raw_examples, tokenizer)
    print(f"Dataset caricato: {len(train_dataset)} esempi.")

    # --- 4. DATA COLLATOR ---
    # Serve per calcolare la loss SOLO sulla risposta dell'assistente.
    # Per Qwen (formato ChatML), la risposta inizia dopo questo marcatore:
    response_template = "<|im_start|>assistant\n"

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer
    )

    # --- 5. CARICAMENTO MODELLO ---
    print("Caricamento Modello (potrebbe richiedere tempo)...")

    # Configurazione specifica per Mac vs Cuda
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,  # bfloat16 è supportato bene su M1/M2/M3
    }

    # Se NON siamo su Mac, proviamo Flash Attention 2 (se installato)
    if device == "cuda":
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        **model_kwargs
    )

    # Abilita il gradient checkpointing per risparmiare VRAM (fondamentale su 7B)
    model.gradient_checkpointing_enable()

    # --- 6. CONFIGURAZIONE LoRA (PEFT) ---
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Moduli target per Qwen (attention + feed forward)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # --- 7. CONFIGURAZIONE TRAINING ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Basso per evitare OutOfMemory su Mac
        gradient_accumulation_steps=8,  # Alto per compensare il batch size basso (1*8 = batch effettivo 8)
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="epoch",
        fp16=False,  # Disabilita fp16 su Mac
        bf16=True,  # Usa bfloat16 (Apple Silicon lo ama)
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_torch",  # Ottimizzatore standard PyTorch (stabile su MPS)
        report_to="none",  # Disabilita WandB/Tensorboard per ora
        # fix per compatibilità dataset text field
        remove_unused_columns=True
    )

    # --- 8. INIZIALIZZAZIONE TRAINER ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=2048,  # Ridotto da 4096 per memoria. Alzalo se hai 32GB+ RAM
        data_collator=collator,
        tokenizer=tokenizer,
        peft_config=peft_config,
        args=training_args,
        packing=False
    )

    # --- 9. AVVIO TRAINING ---
    print("\n--- Avvio Training ---")
    trainer.train()

    print("Salvataggio modello...")
    trainer.save_model(OUTPUT_DIR)
    print(f"Finito! Modello salvato in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()