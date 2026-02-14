from __future__ import annotations

from pathlib import Path


# ==============================================================================
# PROJECT PATHS
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINTS_EN_ACTOR = CHECKPOINTS_DIR / "en_actor_lora"
CHECKPOINTS_IT_ACTOR = CHECKPOINTS_DIR / "it_actor_lora"
CHECKPOINTS_CRITIC = CHECKPOINTS_DIR / "critic"
CHECKPOINTS_TRAINING_LORA = CHECKPOINTS_DIR / "training_lora"


# ==============================================================================
# MODELS
# ==============================================================================

DEV_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_MODEL_NAME_EN = "Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_MODEL_NAME_IT = TRAIN_MODEL_NAME_EN
CRITIC_MODEL_NAME_EN = TRAIN_MODEL_NAME_EN


# ==============================================================================
# GENERATION DEFAULTS
# ==============================================================================

GEN_MAX_NEW_TOKENS = 256
GEN_TEMPERATURE = 0.2
GEN_TOP_P = 0.9


def get_model_name(mode: str = "dev", lang: str = "en", role: str = "actor") -> str:
    """Return HuggingFace model name from mode (dev/train), lang (en/it), and role (actor/critic)."""
    mode = mode.lower()
    lang = lang.lower()
    role = role.lower()

    if mode == "dev":
        return DEV_MODEL_NAME

    if role == "critic":
        return CRITIC_MODEL_NAME_EN

    if lang == "it":
        return TRAIN_MODEL_NAME_IT
    return TRAIN_MODEL_NAME_EN