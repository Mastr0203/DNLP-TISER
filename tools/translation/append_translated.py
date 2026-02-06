#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Union

def read_input_text(path: Path | None) -> str:
    if path is None:
        # stdin
        return input("Incolla il chunk tradotto (JSON list oppure JSONL). Poi premi Invio.\n> ")
    return path.read_text(encoding="utf-8")

def parse_chunk(text: str) -> List[Dict[str, Any]]:
    text = text.strip()
    if not text:
        raise ValueError("Chunk vuoto.")

    # Caso 1: JSON list
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Il JSON non è una lista.")
        # filtra solo dict
        out = []
        for i, x in enumerate(data):
            if not isinstance(x, dict):
                raise ValueError(f"Elemento {i} non è un oggetto JSON (dict).")
            out.append(x)
        return out

    # Caso 2: JSONL (più righe)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out: List[Dict[str, Any]] = []
    for i, ln in enumerate(lines, start=1):
        obj = json.loads(ln)
        if not isinstance(obj, dict):
            raise ValueError(f"Riga {i} non è un oggetto JSON (dict).")
        out.append(obj)
    return out

def choose_split(cli_split: str | None) -> str:
    if cli_split in {"train", "test"}:
        return cli_split
    while True:
        s = input("Questo chunk è TRAIN o TEST? (train/test): ").strip().lower()
        if s in {"train", "test"}:
            return s
        print("Valore non valido. Scrivi 'train' oppure 'test'.")

def append_jsonl(out_path: Path, items: List[Dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser(description="Append di chunk tradotti a un file JSONL (train/test).")
    ap.add_argument("--chunk-file", type=Path, default=None,
                    help="File che contiene il chunk tradotto (JSON list o JSONL). Se omesso, legge da input().")
    ap.add_argument("--split", type=str, default=None, choices=[None, "train", "test"],
                    help="Se specificato, evita la domanda interattiva.")
    ap.add_argument("--out-dir", type=Path, default=Path("data/translated"),
                    help="Cartella di output (default: data/translated)")
    ap.add_argument("--prefix", type=str, default="TISER_it",
                    help="Prefisso file output (default: TISER_it). Output: <prefix>_train.jsonl e <prefix>_test.jsonl")
    args = ap.parse_args()

    split = choose_split(args.split)

    text = read_input_text(args.chunk_file)
    items = parse_chunk(text)

    out_path = args.out_dir / f"{args.prefix}_{split}.jsonl"
    append_jsonl(out_path, items)

    print(f"[OK] Appesi {len(items)} esempi a: {out_path.resolve()}")

if __name__ == "__main__":
    main()