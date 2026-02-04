"""
CLI Script for TISER Actor-Critic Inference Pipeline

This script executes the multi-stage reasoning pipeline (Actor -> Critic -> Solver)
for the TISER dataset, supporting both base models and fine-tuned LoRA adapters.

Key Features:
- Multi-Stage Inference: Implements the full Actor (Reasoning), Critic (Reflection),
  and Solver (Adjustment) loop.
- Dynamic Prompting: Automatically switches between instruction-heavy prompts for
  base models and minimal inputs for fine-tuned (LoRA) models.
- Robust Generation: Handles XML tag validation and retries for malformed outputs.
- Comprehensive Logging: Exports detailed CSV results including raw outputs from
  all stages for error analysis.

Examples:
    # Run standard inference with a base model (Zero-Shot/Few-Shot)
    python scripts/run_actor_critic.py --test-file data/processed/demo_test.json --tag base_run --lora-path unsloth/Qwen2.5-7B-Instruct-bnb-4bit
    
    # Run inference with a Fine-Tuned LoRA adapter
    python scripts/run_actor_critic.py --test-file data/processed/demo_test.json --lora-path experiments/run_qwen_10pct --tag ft_run
    
    # Quick debug run on first 5 examples
    python scripts/run_actor_critic.py --test-file data/processed/TISER_test.json --max-examples 5 --tag debug_quick --lora-path experiments/run_qwen_10pct
    
    # Run with higher temperature for creative critique
    python scripts/run_actor_critic.py --test-file data/processed/TISER_test.json --temp 0.7 --lora-path experiments/run_qwen_10pct
"""

import sys
import os
import torch
import csv
import re
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from collections import defaultdict
from tqdm import tqdm

# Import Unsloth
from unsloth import FastLanguageModel

# Root del progetto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.tiser_dataset import load_tiser_file
from src.tiser.metrics import compute_em_f1
from src.tiser.parsing import extract_answer, extract_section
from src.tiser.prompts import (
    ACTOR_FINETUNED_TEMPLATE,
    CRITIC_PROMPT_TEMPLATE,
    FINAL_SOLVER_PROMPT_TEMPLATE
)

# ==============================================================================
# UTILS & CLEANING
# ==============================================================================

def super_aggressive_clean(text: str) -> str:
    if not text: return ""
    text = text.lower().strip().strip('.,!?;:()[]{}')
    patterns = [r"^the event is\s*", r"^the answer is\s*", r"^answer:\s*", r"^it is\s*", r"^the duration is\s*"]
    for p in patterns: text = re.sub(p, "", text, flags=re.IGNORECASE)
    num_map = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}
    for word, num in num_map.items(): text = re.sub(rf"\b{word}\b", num, text)
    text = text.replace(" years", "").replace(" year", "").replace(" yrs", "").replace(" yr", "")
    return text.strip(' " \',')

def compute_metrics_by_dataset(rows: List[Dict[str, str]]) -> Tuple[Dict[str, Dict[str, float]], float]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["dataset_name"]].append((r["pred_answer"], r["gold_answer"]))
    per_dataset = {}
    for ds, pairs in grouped.items():
        em, f1 = compute_em_f1(pairs)
        per_dataset[ds] = {"em": float(em), "f1": float(f1), "n": len(pairs)}
    macro_avg_em = sum(v["em"] for v in per_dataset.values()) / len(per_dataset) if per_dataset else 0.0
    return per_dataset, float(macro_avg_em)

# ==============================================================================
# UNLSOTH GENERATION WRAPPER
# ==============================================================================

def unsloth_generate(model, tokenizer, prompt, max_new_tokens, temperature):
    input_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to("cuda")
    
    # Safety check for context length
    if inputs.input_ids.shape[1] > 3800:
        return "[ERROR: CONTEXT TOO LONG]"

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=True if temperature > 0.01 else False,
            temperature=temperature,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id
        )
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("assistant\n")[-1].strip()

# ==============================================================================
# ACTOR-CRITIC PIPELINE
# ==============================================================================

def run_pipeline_step(model, tokenizer, ex, args):
    # 1. ACTOR (Reasoning + Timeline)
    actor_prompt = ACTOR_FINETUNED_TEMPLATE.format(question=ex.question, context=ex.context)
    raw_stage_1 = unsloth_generate(model, tokenizer, actor_prompt, 1024, args.temp)
    
    draft_reasoning = extract_section(raw_stage_1, "reasoning") or raw_stage_1
    draft_timeline = extract_section(raw_stage_1, "timeline") or "Timeline tag missing."

    # 2. CRITIC (Reflection)
    critic_prompt = CRITIC_PROMPT_TEMPLATE.format(
        question=ex.question, context=ex.context,
        draft_reasoning=draft_reasoning, draft_timeline=draft_timeline
    )
    raw_stage_2 = unsloth_generate(model, tokenizer, critic_prompt, 512, args.temp)
    critic_reflection = extract_section(raw_stage_2, "reflection") or raw_stage_2

    # 3. SOLVER (Final Answer)
    final_prompt = FINAL_SOLVER_PROMPT_TEMPLATE.format(
        question=ex.question, context=ex.context,
        draft_reasoning=draft_reasoning, draft_timeline=draft_timeline,
        critic_reflection=critic_reflection
    )
    raw_stage_3 = unsloth_generate(model, tokenizer, final_prompt, 512, args.temp)
    
    final_answer_text = extract_answer(raw_stage_3)
    # Applichiamo pulizia aggressiva per EM
    final_answer_cleaned = super_aggressive_clean(final_answer_text)

    return {
        "pred": final_answer_cleaned,
        "raw_combined": f"S1: {raw_stage_1}\n\nS2: {raw_stage_2}\n\nS3: {raw_stage_3}",
        "has_tag": "<answer>" in raw_stage_3.lower()
    }

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", type=str, required=True, help="LoRA or Base Model path")
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--tag", type=str, default="pipeline_unsloth")
    parser.add_argument("--temp", type=float, default=0.01)
    args = parser.parse_args()

    # Load Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.lora_path,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # Load Data
    examples = load_tiser_file(Path(args.test_file), max_examples=args.max_examples)
    out_dir = Path("results/pipeline_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    print(f"🚀 Starting Pipeline with {len(examples)} examples...")

    for i, ex in enumerate(tqdm(examples)):
        res = run_pipeline_step(model, tokenizer, ex, args)
        gold_cleaned = super_aggressive_clean(ex.answer)
        
        csv_rows.append({
            "idx": i,
            "dataset_name": ex.dataset_name,
            "question": ex.question,
            "gold_answer": gold_cleaned,
            "pred_answer": res["pred"],
            "raw_output": res["raw_combined"],
            "has_answer_tag": res["has_tag"]
        })

    # Metrics
    per_ds, macro_em = compute_metrics_by_dataset(csv_rows)
    print(f"\n📊 FINAL MACRO EM: {macro_em:.4f}")

    # Save
    out_file = out_dir / f"pipeline_results_{args.tag}.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        w.writeheader()
        w.writerows(csv_rows)
    
    print(f"✅ Results saved to {out_file}")

if __name__ == "__main__":
    main()