# src/data/tiser_dataset.py
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Iterable
from torch.utils.data import Dataset


@dataclass
class TiserExample:
    """
    Rappresenta un singolo esempio del dataset TISER.
    """
    dataset_name: str
    question_id: str
    question: str
    answer: str
    prompt: str
    output: Optional[str] = None  # None durante il test/inferenza


def _normalize_item(raw: dict) -> TiserExample:
    """Converte un dizionario grezzo in un oggetto TiserExample."""
    return TiserExample(
        dataset_name=raw.get("dataset_name", ""),
        question_id=str(raw.get("question_id", "")),
        question=raw.get("question", ""),
        answer=raw.get("answer", ""),
        prompt=raw.get("prompt", ""),
        output=raw.get("output"),
    )


def load_tiser_file(
        path: Path | str,
        max_examples: Optional[int] = None,
        dataset_filter: Optional[Iterable[str]] = None,
) -> List[TiserExample]:
    """
    Legge un file JSONL (JSON Lines) TISER e restituisce una lista di esempi.
    Gestisce correttamente la lettura riga per riga per evitare JSONDecodeError.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File TISER non trovato: {path}")

    data = []
    # Lettura robusta per file JSONL (una struttura JSON per riga)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Attenzione: saltata riga corrotta in {path}: {e}")

    examples: List[TiserExample] = []
    ds_filter_set = set(dataset_filter) if dataset_filter is not None else None

    for raw in data:
        ex = _normalize_item(raw)
        # Filtro opzionale per dataset specifico
        if ds_filter_set is not None and ex.dataset_name not in ds_filter_set:
            continue

        examples.append(ex)
        if max_examples is not None and len(examples) >= max_examples:
            break

    return examples


class TiserDataset(Dataset):
    def __init__(self, examples: list[TiserExample], tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]

        # Safety check per il training
        if item.output is None:
            # Fallback o errore a seconda della preferenza. Qui solleviamo errore.
            raise ValueError(f"Esempio {item.question_id} senza output (target). Impossibile usare per training.")

        # Costruzione messaggi stile Chat
        messages = [
            {"role": "user", "content": item.prompt},  # Mapping: prompt -> user message
            {"role": "assistant", "content": item.output}  # Mapping: output -> assistant message
        ]

        # Applica il template di Qwen/ChatML senza tokenizzare subito
        formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False)

        return {"text": formatted_text}