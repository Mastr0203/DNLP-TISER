from __future__ import annotations

import re
import string
from typing import List, Tuple, Any, Set
from collections import Counter


# ==============================================================================
# TEXT NORMALIZATION & EXTRACTION UTILITIES
# ==============================================================================

STOPWORDS = {
    "the", "a", "an",
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
    "of", "in", "on", "at", "to", "for", "with", "from", "by",
    "and", "or", "but",
    "di", "da", "in", "con", "su", "per", "tra", "fra", "e", "ed",
    "between", "among",
}


def _normalize_text(s: Any) -> str:
    """
    Basic normalization: lowercase, strip, collapse whitespace.
    Used by exact_match and token_f1.
    """
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return " ".join(s.strip().lower().split())


def _normalize_for_soft_match(text: str) -> str:
    """
    Advanced normalization for Soft Match:
    - Lowercase
    - Convert hyphens to spaces (e.g., "Marie-Antoinette" -> "Marie Antoinette")
    - Remove punctuation (except for numbers like "2.5")
    - Strip stopwords
    """
    if not text:
        return ""
    text = text.lower().strip()
    text = text.replace('-', ' ')
    cleaned = []
    for i, char in enumerate(text):
        if char.isalnum() or char.isspace():
            cleaned.append(char)
        elif char == '.' and i > 0 and i < len(text) - 1:
            if text[i-1].isdigit() and text[i+1].isdigit():
                cleaned.append(char)
    text = ''.join(cleaned)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    
    return ' '.join(tokens)


def _extract_numbers(text: str) -> Set[str]:
    """
    Extract both Arabic digits (e.g., "1990", "2.5") and Roman numerals.
    Returns a set of normalized number strings.
    
    Examples:
        "Elizabeth II" -> {"ii"}
        "Louis XIV" -> {"xiv"}
        "In 1990" -> {"1990"}
        "2.5 years" -> {"2.5"}
    """
    if not text:
        return set()
    
    text = text.lower()
    numbers = set()
    arabic_pattern = r'\b\d+(?:\.\d+)?\b'
    for match in re.finditer(arabic_pattern, text):
        numbers.add(match.group())
    roman_pattern = r'\b[ivxlcdm]+\b'
    for match in re.finditer(roman_pattern, text):
        candidate = match.group()
        if _is_valid_roman(candidate):
            numbers.add(candidate)
    
    return numbers


def _is_valid_roman(s: str) -> bool:
    """
    Basic validation to check if a string looks like a valid Roman numeral.
    Prevents false positives from random words like "mix" or "civil".
    """
    if not s:
        return False
    valid_chars = set('ivxlcdm')
    if not set(s).issubset(valid_chars):
        return False
    false_positives = {
        'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x',
        'xi', 'xii', 'xiii', 'xiv', 'xv', 'xvi', 'xvii', 'xviii', 'xix', 'xx',
        'xxi', 'xxii', 'xxiii', 'xxiv', 'xxv', 'xxx', 'xl', 'l', 'lx', 'lxx',
        'lxxx', 'xc', 'c', 'cc', 'ccc', 'cd', 'd', 'dc', 'dcc', 'dccc', 'cm', 'm',
        'mm', 'mmm',         'mmmm'
    }
    if s in false_positives:
        return True
    roman_regex = r'^m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$'
    if re.match(roman_regex, s):
        return True
    if len(s) == 1 and s in 'ivxlcdm':
        return True
    
    return False


# ==============================================================================
# CORE METRICS
# ==============================================================================

def exact_match(pred: str, gold: str) -> float:
    """
    Exact Match binary (0/1) after basic normalization.
    """
    return 1.0 if _normalize_text(pred) == _normalize_text(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    """
    Token-level F1: standard for QA.
    """
    pred_tokens = _normalize_text(pred).split()
    gold_tokens = _normalize_text(gold).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)

    common = sum((pred_counts & gold_counts).values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def soft_match(pred: str, gold: str) -> float:
    """
    Soft Match (SM) metric: handles generative variances while strictly enforcing entity identity.
    
    Algorithm:
    1. Extract numbers (Arabic + Roman) from both pred and gold
    2. If number sets differ -> MISMATCH (0.0)
    3. If number sets match:
       - Normalize text (lowercase, remove punctuation, strip stopwords)
       - Check bidirectional token-level inclusion:
         a) Are all gold tokens contained in pred tokens? (verbose prediction)
         b) Are all pred tokens contained in gold tokens? (verbose gold)
            - Constraint: Only apply Direction B if pred_norm length > 3 chars OR if numbers exist
              (This prevents matching short meaningless words like "it", "he", etc.)
       - If either direction matches -> MATCH (1.0)
    4. Otherwise -> MISMATCH (0.0)
    
    Examples:
        - Gold="2 years" | Pred="2" -> SM: 1.0 (numbers match {"2"}, pred tokens in gold)
        - Gold="Elisabetta I" | Pred="Elisabetta II" -> SM: 0.0 (numbers differ: {i} vs {ii})
        - Gold="Clinton" | Pred="Bill Clinton" -> SM: 1.0 (no numbers, "clinton" in "bill clinton")
        - Gold="Bill Clinton" | Pred="Clinton" -> SM: 1.0 (no numbers, "clinton" in "bill clinton")
        - Gold="Elizabeth" | Pred="Elizabeth II" -> SM: 0.0 (numbers: {} vs {ii})
        - Gold="Louis XIV" | Pred="Louis XVI" -> SM: 0.0 (numbers: {xiv} vs {xvi})
        - Gold="1990" | Pred="In 1990" -> SM: 1.0 (numbers: {1990} == {1990}, "1990" in "in 1990")
    """
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    pred_numbers = _extract_numbers(pred)
    gold_numbers = _extract_numbers(gold)
    if pred_numbers != gold_numbers:
        return 0.0
    pred_norm = _normalize_for_soft_match(pred)
    gold_norm = _normalize_for_soft_match(gold)
    if not pred_norm and not gold_norm:
        return 1.0
    if not pred_norm or not gold_norm:
        return 0.0
    pred_tokens = set(pred_norm.split())
    gold_tokens = set(gold_norm.split())
    if gold_tokens.issubset(pred_tokens):
        return 1.0
    has_numbers = len(pred_numbers) > 0 or len(gold_numbers) > 0
    if (len(pred_norm) > 3 or has_numbers) and pred_tokens.issubset(gold_tokens):
        return 1.0
    
    return 0.0


# ==============================================================================
# BATCH COMPUTATION
# ==============================================================================

def compute_metrics(pairs: List[Tuple[str, str]]) -> dict:
    """
    Computes EM, F1, and SM for a list of (pred, gold) pairs.
    
    Args:
        pairs: List of (prediction, gold_answer) tuples
    
    Returns:
        Dictionary with keys: "em", "f1", "sm"
    """
    if not pairs:
        return {"em": 0.0, "f1": 0.0, "sm": 0.0}

    em_sum = 0.0
    f1_sum = 0.0
    sm_sum = 0.0
    
    for pred, gold in pairs:
        em_sum += exact_match(pred, gold)
        f1_sum += token_f1(pred, gold)
        sm_sum += soft_match(pred, gold)

    n = len(pairs)
    return {
        "em": em_sum / n,
        "f1": f1_sum / n,
        "sm": sm_sum / n,
    }


def compute_em_f1(pairs: List[Tuple[str, str]]) -> Tuple[float, float]:
    """
    Legacy function for backward compatibility.
    Computes only EM and F1.
    
    Args:
        pairs: List of (pred, gold) tuples
    
    Returns:
        Tuple of (EM_avg, F1_avg)
    """
    if not pairs:
        return 0.0, 0.0

    em_sum = 0.0
    f1_sum = 0.0
    for pred, gold in pairs:
        em_sum += exact_match(pred, gold)
        f1_sum += token_f1(pred, gold)

    n = len(pairs)
    return em_sum / n, f1_sum / n
