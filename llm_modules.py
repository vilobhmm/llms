"""
llm_modules.py — importlib loader for digit-prefixed module files
=================================================================
Python cannot import files whose names start with digits using normal
`import` syntax.  This helper uses importlib to register all modules
under clean aliases so any module can do:

    from llm_modules import data_cleaning, transformer_model, ...
"""

import sys
import importlib.util
from pathlib import Path

_ROOT = Path(__file__).parent


def _load(alias: str, filename: str):
    if alias in sys.modules:
        return sys.modules[alias]
    path = _ROOT / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


# ── Public aliases ──────────────────────────────────────────────────────────

data_cleaning    = _load("data_cleaning",    "01_data_cleaning.py")
embeddings       = _load("embeddings",       "02_embeddings.py")
attention        = _load("attention",        "03_attention.py")
transformer      = _load("transformer",      "04_transformer_model.py")
pretraining      = _load("pretraining",      "05_pretraining.py")
sft              = _load("sft",              "06_sft.py")
reward_model_mod = _load("reward_model_mod", "07_reward_model.py")
rlhf             = _load("rlhf",             "08_rlhf_ppo_dpo.py")
visualizations   = _load("visualizations",   "09_visualizations.py")

__all__ = [
    "data_cleaning", "embeddings", "attention", "transformer",
    "pretraining", "sft", "reward_model_mod", "rlhf", "visualizations",
]
