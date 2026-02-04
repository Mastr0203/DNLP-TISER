"""
Ginny:
python scripts/run_stats.py \
    --variant no_reasoning \
    --test-file data/processed/TISER_test_10pct.json \
    --lora-path experiments/run_qwen_10pct \

Mastro:
python scripts/run_stats.py \
    --variant all_stages,standard \
    --test-file data/processed/TISER_test_10pct.json \
    --lora-path unsloth/Qwen2.5-7B-Instruct-bnb-4bit \
    --tag qwen_base_baseline 
"""

import sys
import os
import torch
import json
import csv
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from tqdm import tqdm
from unsloth import FastLanguageModel

# Root del progetto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.tiser_dataset import load_tiser_file
from src.tiser.metrics import compute_em_f1
from src.tiser.prompts import (
    STANDARD_PROMPT_TEMPLATE,               
    ABLATION_ONLY_REASONING_PROMPT_TEMPLATE,         
    ABLATION_ONLY_TIMELINE_PROMPT_TEMPLATE,          
    ABLATION_NO_REFLECTION_PROMPT_TEMPLATE,          
    ABLATION_NO_TIMELINE_PROMPT_TEMPLATE,          
    ABLATION_NO_REASONING_PROMPT_TEMPLATE,           
    TISER_PROMPT_TEMPLATE, 
)

VARIANT_PROMPTS: Dict[str, str] = {
    "standard": STANDARD_PROMPT_TEMPLATE,
    "only_reasoning": ABLATION_ONLY_REASONING_PROMPT_TEMPLATE,
    "only_timeline": ABLATION_ONLY_TIMELINE_PROMPT_TEMPLATE,
    "no_reflection": ABLATION_NO_REFLECTION_PROMPT_TEMPLATE,
    "no_timeline": ABLATION_NO_TIMELINE_PROMPT_TEMPLATE,
    "no_reasoning": ABLATION_NO_REASONING_PROMPT_TEMPLATE,
    "all_stages": TISER_PROMPT_TEMPLATE,
}

def super_aggressive_clean(text: str) -> str:
    if not text: return ""
    text = text.lower().strip().strip('.,!?;:()[]{}')
    patterns = [r"^the event is\s*", r"^the answer is\s*", r"^answer:\s*", r"^it is\s*", r"^the duration is\s*"]
    for p in patterns: text = re.sub(p, "", text, flags=re.IGNORECASE)
    num_map = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
    for word, num in num_map.items(): text = re.sub(rf"\b{word}\b", num, text)
    text = text.replace(" years", "").replace(" year", "").replace(" yrs", "").replace(" yr", "")
    return text.strip(' " \',')

def robust_extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match: answer = match.group(1).strip()
    else:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        answer = lines[-1] if lines else text.strip()
    return super_aggressive_clean(answer)

def run_multi_ablation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--lora-path", type=str, required=True)
    parser.add_argument("--variant", type=str, default="all", help="Nomi varianti separati da virgola o 'all'")
    parser.add_argument("--tag", type=str, default="nightly_run")
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    # --- FIX: Parsing Multi-Variante ---
    if args.variant == "all":
        selected_variants = list(VARIANT_PROMPTS.keys())
    else:
        # Divide la stringa per virgola e pulisce gli spazi
        selected_variants = [v.strip() for v in args.variant.split(",")]
        # Verifica che siano tutte varianti valide
        for v in selected_variants:
            if v not in VARIANT_PROMPTS:
                print(f"❌ Errore: la variante '{v}' non esiste!")
                print(f"Scegli tra: {list(VARIANT_PROMPTS.keys())}")
                return

    # Caricamento Modello
    MAX_SEQ = 4096
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.lora_path,
        max_seq_length = MAX_SEQ,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    examples = load_tiser_file(Path(args.test_file), max_examples=args.max_examples)
    results_dir = Path("results/nightly_evals")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    summary_rows = []

    for variant in selected_variants:
        print(f"\n🌙 ESECUZIONE VARIANTE: {variant}")
        template = VARIANT_PROMPTS[variant]
        rows = []
        skipped = 0

        for ex in tqdm(examples, desc=f"Processing {variant}"):
            prompt = template.format(question=ex.question, context=ex.context)
            input_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(input_text, return_tensors="pt", padding=True).to("cuda")
            
            if inputs.input_ids.shape[1] > (MAX_SEQ - 1024):
                skipped += 1
                continue 

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    use_cache=True,
                    do_sample=False, 
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id
                )

            raw = tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant\n")[-1]
            pred = robust_extract_answer(raw)
            gold = super_aggressive_clean(ex.answer)
            rows.append({"dataset": ex.dataset_name, "gold": gold, "pred": pred})

        # Statistiche
        if not rows: continue
        all_p = [(r["pred"], r["gold"]) for r in rows]
        em, f1 = compute_em_f1(all_p)
        
        # Salvataggio SUMMARY parziale
        res_entry = {"variant": variant, "em": f"{em:.4f}", "f1": f"{f1:.4f}", "n": len(rows), "skipped": skipped}
        summary_rows.append(res_entry)
        
        # Scrive/Aggiorna un file CSV di riepilogo per non perdere dati se crasha dopo
        with open(results_dir / f"SUMMARY_{args.tag}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["variant", "em", "f1", "n", "skipped"])
            w.writeheader()
            w.writerows(summary_rows)

        print(f"✅ Finito {variant}: EM={em:.4f} | F1={f1:.4f}")

    print(f"\n✨ Tutte le varianti selezionate sono state completate! Report: {results_dir}")

if __name__ == "__main__":
    run_multi_ablation()