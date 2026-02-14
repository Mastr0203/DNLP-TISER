#!/usr/bin/env python
"""
TISER prompt ablation runner (single-prompt inference).

Runs multiple prompt variants over the same test set with one model.
Supports incremental CSV saving, --resume, optional --eval-on-raw, and OOM recovery.

Examples:
    python scripts/run_inference.py --test-file data/processed/TISER_test_demo_10pct.json --tag ablation
    python scripts/run_inference.py --resume --save-every 50
    python scripts/run_inference.py --eval-on-raw --lora-path models/qwen_finetuned
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import csv
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from collections import defaultdict

import torch

from src.config import (
    RESULTS_DIR,
    PROCESSED_DIR,
    GEN_TEMPERATURE,
    GEN_TOP_P,
    get_model_name,
)
from src.models.base_model import LLMWrapper
from src.data.tiser_dataset import load_tiser_file
from src.utils.metrics import compute_em_f1
from src.utils.parsing import extract_answer
from src.utils.prompts import (
    STANDARD_PROMPT_TEMPLATE,
    ABLATION_ONLY_REASONING_PROMPT_TEMPLATE,
    ABLATION_ONLY_TIMELINE_PROMPT_TEMPLATE,
    ABLATION_NO_REFLECTION_PROMPT_TEMPLATE,
    ABLATION_NO_TIMELINE_PROMPT_TEMPLATE,
    ABLATION_NO_REASONING_PROMPT_TEMPLATE,
    TISER_PROMPT_TEMPLATE,
    TISER_PROMPT_TEMPLATE_IT,
    STANDARD_PROMPT_TEMPLATE_IT
)

VARIANT_PROMPTS: Dict[str, str] = {
    "standard": STANDARD_PROMPT_TEMPLATE,
    "only_reasoning": ABLATION_ONLY_REASONING_PROMPT_TEMPLATE,
    "only_timeline": ABLATION_ONLY_TIMELINE_PROMPT_TEMPLATE,
    "no_reflection": ABLATION_NO_REFLECTION_PROMPT_TEMPLATE,
    "no_timeline": ABLATION_NO_TIMELINE_PROMPT_TEMPLATE,
    "no_reasoning": ABLATION_NO_REASONING_PROMPT_TEMPLATE,
    "all_stages": TISER_PROMPT_TEMPLATE
}

# ==============================================================================
# UTILITIES
# ==============================================================================

def flatten_text(text: str) -> str:
    """Flatten newlines for single-line CSV logging."""
    if not text:
        return ""
    return text.replace("\n", " ").replace("\r", " ")

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def fallback_extract_from_raw(raw: str) -> str:
    """
    Lightweight fallback extractor for temporal QA outputs when tag-based extraction fails.
    """
    raw = normalize_text(raw)
    if not raw:
        return ""

    low = raw.lower()

    for token in ["true", "false", "vero", "falso"]:
        if re.search(rf"\b{token}\b", low):
            return token.capitalize()

    nums = re.findall(r"[-+]?\d+(?:[\.,]\d+)?", raw)
    if nums:
        return nums[-1].replace(",", ".")

    separators = ["<answer>", "answer:", "risposta:", "finale:", "=>", "->"]
    for sep in separators:
        if sep in low:
            tail = raw.split(sep, 1)[-1]
            tail = normalize_text(tail)
            tail = re.split(r"</answer>|<\|im_end\|>", tail, flags=re.IGNORECASE)[0]
            tail = normalize_text(tail)
            if tail:
                return tail

    parts = re.split(r"(?<=[\.\?\!])\s+", raw)
    parts = [p.strip() for p in parts if p.strip()]
    return parts[-1] if parts else raw

def extract_answer_rawfirst(raw: str) -> str:
    """Try tag-based extract_answer; if empty, use fallback_extract_from_raw."""
    raw = raw or ""
    a = normalize_text(extract_answer(raw))
    if a:
        return a
    return normalize_text(fallback_extract_from_raw(raw))

def compute_detailed_metrics_from_pairs(
    pairs_by_dataset: Dict[str, List[Tuple[str, str]]],
    all_pairs: List[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    """Compute EM and F1 per dataset plus overall (micro)."""
    metrics_list: List[Dict[str, Any]] = []

    for ds_name, pairs in pairs_by_dataset.items():
        em, f1 = compute_em_f1(pairs)
        metrics_list.append({
            "dataset_name": ds_name,
            "n": len(pairs),
            "em": em,
            "f1": f1
        })

    if all_pairs:
        ov_em, ov_f1 = compute_em_f1(all_pairs)
        metrics_list.append({
            "dataset_name": "__OVERALL__",
            "n": len(all_pairs),
            "em": ov_em,
            "f1": ov_f1
        })

    metrics_list.sort(key=lambda x: x["dataset_name"])
    return metrics_list

# ==============================================================================
# GENERATION HELPERS
# ==============================================================================

def generate_until_answer(
    llm: LLMWrapper,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_retries: int = 2,
    growth: float = 2.0,
    hard_cap: int = 1024,
) -> str:
    """Generate until <answer> or </answer> is present; retry with more tokens if missing."""
    cur = max_new_tokens
    out = llm.generate(prompt=prompt, max_new_tokens=cur, temperature=temperature, top_p=top_p)

    def has_answer_marker(txt: str) -> bool:
        low = (txt or "").lower()
        return ("<answer>" in low) or ("</answer>" in low)

    for r in range(max_retries):
        if has_answer_marker(out):
            return out
        cur = min(int(cur * growth), hard_cap)
        print(f"    [WARN] Missing <answer>/</answer>. Retrying with max_new_tokens={cur} (retry {r+1}/{max_retries})")
        out = llm.generate(prompt=prompt, max_new_tokens=cur, temperature=temperature, top_p=top_p)

    return out

# ==============================================================================
# MODEL
# ==============================================================================

def build_model(mode: str = "dev", lang: str = "en", lora_path: Optional[str] = None) -> LLMWrapper:
    model_name = get_model_name(mode=mode, lang=lang, role="actor")

    print(f"[MODEL] Base model: {model_name}")
    if lora_path:
        print(f"[MODEL] Loading LoRA adapter from: {lora_path}")
        return LLMWrapper(model_name=model_name, lora_path=lora_path)
    else:
        return LLMWrapper(model_name=model_name)

# ==============================================================================
# RESUME HELPERS
# ==============================================================================

def load_done_question_ids(csv_path: Path) -> set:
    """For --resume: return set of question_ids already present in the CSV."""
    if not csv_path.exists():
        return set()
    done = set()
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if "question_id" in df.columns:
            for qid in df["question_id"].fillna("").astype(str).tolist():
                done.add(qid)
    except Exception:
         # If pandas not available or CSV is partial/corrupted, fall back to csv module
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                qid = (row.get("question_id") or "").strip()
                done.add(qid)
    return done

def ensure_csv_writer(csv_path: Path, fieldnames: List[str]) -> csv.DictWriter:
    """Placeholder: caller opens file and creates DictWriter."""
    raise NotImplementedError

# ==============================================================================
# MAIN
# ==============================================================================

DEFAULT_VARIANTS = ["standard", "only_reasoning", "only_timeline", "no_reflection", "no_timeline", "no_reasoning", "all_stages"]

def main():
    default_test = str(PROCESSED_DIR / "TISER_test_demo_10pct.json")
    parser = argparse.ArgumentParser(description="TISER Ablation Runner (Vertical Summary)")
    parser.add_argument("--mode", type=str, default="dev", choices=["dev", "train"])
    parser.add_argument("--test-file", type=str, default=default_test, help="Path to JSON/JSONL test file")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--tag", type=str, default="ablation")
    parser.add_argument("--temp", type=float, default=GEN_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=GEN_TOP_P)

    # Generation parameters
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--hard-cap", type=int, default=1024)

    # Evaluation parameters
    parser.add_argument(
        "--eval-on-raw",
        action="store_true",
        help="If set, compute metrics on pred_from_raw (extract_answer(raw) with fallback). "
             "CSV still stores both pred_answer and pred_from_raw."
    )

    # CSV saving and resume parameters
    parser.add_argument("--save-every", type=int, default=50, help="Append rows to CSV every N examples.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing per-variant CSV (skip done question_ids).")

    parser.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--lora-path", type=str, default=None, help="Path to fine-tuned LoRA adapter (optional)")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "it"], help="Language for model and prompt templates")

    args = parser.parse_args()

    if args.lang == "it":
        VARIANT_PROMPTS["standard"] = STANDARD_PROMPT_TEMPLATE_IT
        VARIANT_PROMPTS["all_stages"] = TISER_PROMPT_TEMPLATE_IT

    test_path = Path(args.test_file)
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in VARIANT_PROMPTS:
            raise ValueError(f"Unknown variant '{v}'. Available: {list(VARIANT_PROMPTS.keys())}")

    print(f"[INFO] Loading test set: {test_path}")
    examples = load_tiser_file(test_path, max_examples=args.max_examples)
    print(f"[INFO] Loaded {len(examples)} examples")

    print(f"[INFO] Initializing model (mode={args.mode})")
    llm = build_model(mode=args.mode, lang=args.lang, lora_path=args.lora_path)

    global_summary_rows: List[Dict[str, Any]] = []

    for variant in variants:
        print(f"\n==============================")
        print(f"[RUN] Variant: {variant}")
        print(f"==============================")

        prompt_template = VARIANT_PROMPTS[variant]

        out_csv = RESULTS_DIR / f"ablation_{args.tag}_{variant}.csv"
        summary_csv = RESULTS_DIR / f"ablation_summary_{args.tag}.csv"

        # For --resume: return set of question_ids already present in the CSV
        done_qids = set()
        if args.resume:
            done_qids = load_done_question_ids(out_csv)
            if done_qids:
                print(f"[RESUME] Found {len(done_qids)} already-processed question_id in {out_csv.name}")

        fieldnames = [
            "idx", "variant", "dataset_name", "question_id",
            "question", "gold_answer",
            "pred_answer", "pred_from_raw",
            "raw_output", "has_answer_tag",
            "oom"
        ]

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        file_exists = out_csv.exists() and out_csv.stat().st_size > 0
        f_out = out_csv.open("a", encoding="utf-8", newline="")
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        # We’ll accumulate pairs for metrics (we might be resuming)
        pairs_by_dataset: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        all_pairs: List[Tuple[str, str]] = []

        if args.resume and file_exists:
            try:
                import pandas as pd
                df_prev = pd.read_csv(out_csv)
                pred_col = "pred_from_raw" if args.eval_on_raw else "pred_answer"
                if pred_col in df_prev.columns and "gold_answer" in df_prev.columns and "dataset_name" in df_prev.columns:
                    for _, row in df_prev.iterrows():
                        ds = str(row.get("dataset_name", ""))
                        pred = str(row.get(pred_col, "") or "")
                        gold = str(row.get("gold_answer", "") or "")
                        pairs_by_dataset[ds].append((pred, gold))
                        all_pairs.append((pred, gold))
            except Exception:
                pass

        buffer_rows: List[Dict[str, Any]] = []
        processed_now = 0

        for i, ex in enumerate(examples, start=1):
            qid = ex.question_id
            if args.resume and qid in done_qids:
                continue

            print(f"  [{i}/{len(examples)}] qid={qid} ({ex.dataset_name})")

            question = ex.question
            context = ex.context
            prompt = prompt_template.format(question=question, context=context)

            raw = ""
            oom_flag = False

            try:
                raw = generate_until_answer(
                    llm=llm,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temp,
                    top_p=args.top_p,
                    max_retries=args.max_retries,
                    growth=2.0,
                    hard_cap=args.hard_cap,
                )
            except torch.OutOfMemoryError:
                oom_flag = True
                print("    [OOM] CUDA out of memory. Clearing cache and skipping this example.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raw = ""

            pred_answer = extract_answer(raw)
            pred_from_raw = extract_answer_rawfirst(raw)
            has_answer_tag = bool((pred_answer or "").strip())
            gold = ex.answer
            pred_for_metrics = pred_from_raw if args.eval_on_raw else pred_answer

            row = {
                "idx": i,
                "variant": variant,
                "dataset_name": ex.dataset_name,
                "question_id": qid,
                "question": question,
                "gold_answer": gold,
                "pred_answer": pred_answer,
                "pred_from_raw": pred_from_raw,
                "raw_output": flatten_text(raw),
                "has_answer_tag": has_answer_tag,
                "oom": oom_flag,
            }
            buffer_rows.append(row)
            pairs_by_dataset[ex.dataset_name].append((pred_for_metrics, gold))
            all_pairs.append((pred_for_metrics, gold))

            processed_now += 1
            if processed_now % args.save_every == 0:
                writer.writerows(buffer_rows)
                f_out.flush()
                buffer_rows = []
        if buffer_rows:
            writer.writerows(buffer_rows)
            f_out.flush()
        f_out.close()
        stats_list = compute_detailed_metrics_from_pairs(pairs_by_dataset, all_pairs)

        print(f"[METRICS] Summary for {variant} (eval_on_raw={args.eval_on_raw}):")
        for stat in stats_list:
            print(f"  - {stat['dataset_name']:20s} | N={stat['n']:4d} | EM={stat['em']:.4f} | F1={stat['f1']:.4f}")
            global_summary_rows.append({
                "tag": args.tag,
                "variant": variant,
                "dataset_name": stat["dataset_name"],
                "n": stat["n"],
                "em": f"{stat['em']:.4f}",
                "f1": f"{stat['f1']:.4f}",
            })
    summary_csv = RESULTS_DIR / f"ablation_summary_{args.tag}.csv"
    print(f"\n[INFO] Saving global vertical summary -> {summary_csv}")

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["tag", "variant", "dataset_name", "n", "em", "f1"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(global_summary_rows)

    print("[DONE] Ablation completed.")

if __name__ == "__main__":
    main()