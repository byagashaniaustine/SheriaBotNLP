"""Generative answer model for SheriaBot.

Wraps a fine-tuned flan-t5 (base or small) that produces a natural-language
answer given (retrieved context + intent + user question). This is the
LAYER 3 of the pipeline:

    [BERT intent]  →  [RAG retrieve]  →  [flan-t5 generate]  →  [cite]

Prompt template (kept in sync with the Colab notebook that trained the model):

    intent: <intent> | context: <top-k chunks joined by " | "> | question: <user text>

Model lives on disk at:

    sheriabot_generator/                    (or GENERATOR_MODEL_DIR env var)
        config.json
        pytorch_model.bin (or model.safetensors)
        tokenizer.json  spiece.model  tokenizer_config.json

Runtime behaviour:

  * import — cheap, no model load
  * first .generate() call — cold-loads model (~1 GB) into CPU RAM,
    takes 5-15 seconds
  * subsequent calls — 2-5 seconds each on CPU

If the model directory is missing OR transformers isn't installed OR the
load fails for any reason, generate() returns None and the answer engine
falls back to the canned answer_bank. This graceful-degradation contract
means the bot always ships — it just gets nicer answers once the generator
is in place.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import ROOT

log = logging.getLogger("sheriabot.generator")

# ------------------------------------------------------------------
# Where the fine-tuned generator lives
# ------------------------------------------------------------------
DEFAULT_MODEL_DIR = ROOT / "sheriabot_generator"
MODEL_DIR = Path(os.getenv("GENERATOR_MODEL_DIR", str(DEFAULT_MODEL_DIR)))

# Fallback base model if MODEL_DIR is missing. If set (via env or code),
# transformers will fetch this from HF the first time. Useful for a dev
# machine that doesn't yet have the fine-tuned copy.
FALLBACK_HF_MODEL_ID = os.getenv("GENERATOR_HF_ID", "")  # empty = no fallback

MAX_INPUT_TOKENS  = 512
MAX_OUTPUT_TOKENS = 256


class _Generator:
    """Owns the tokenizer + model. Singleton. Loads lazily on first call."""

    def __init__(self) -> None:
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._source: str = ""
        self._tried_load: bool = False

    # ------------------------------------------------------------------
    # lazy load
    # ------------------------------------------------------------------
    def _try_load(self) -> None:
        """Attempt to load the model. Sets self._model on success.

        Called on first .generate(). If it fails we set a sentinel so we
        don't try again every request.
        """
        if self._tried_load:
            return
        self._tried_load = True

        try:
            import torch
            # AutoTokenizer picks up the fast tokenizer.json we saved with the
            # fine-tuned model. This avoids the protobuf dependency that
            # T5Tokenizer (slow, SentencePiece-based) requires.
            from transformers import T5ForConditionalGeneration, AutoTokenizer
        except ImportError as e:
            log.warning(
                "Generator dependencies missing (%s). "
                "Install with: pip install 'transformers>=4.30' torch --extra-index-url "
                "https://download.pytorch.org/whl/cpu",
                e,
            )
            return

        # Prefer the local fine-tuned copy.
        source: Optional[str] = None
        if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
            source = str(MODEL_DIR)
        elif FALLBACK_HF_MODEL_ID:
            source = FALLBACK_HF_MODEL_ID
            log.warning(
                "Fine-tuned model dir %s empty — falling back to HF model %s. "
                "Generation quality will be lower until you fine-tune.",
                MODEL_DIR, FALLBACK_HF_MODEL_ID,
            )
        else:
            log.warning(
                "Generator model dir %s missing and no fallback set. "
                "Generation disabled — answer engine will use answer_bank.",
                MODEL_DIR,
            )
            return

        log.info("Loading generator from %s ...", source)
        t0 = time.time()

        # Tokenizer — try strategies in order until one loads WITH the model's
        # own vocabulary. Order matters:
        #   1. Load tokenizer.json directly (bypasses tokenizer_config.json version issues)
        #   2. AutoTokenizer with the local folder
        #   3. T5TokenizerFast with the local folder
        # If ALL local strategies fail, refuse to load — an HF-base fallback
        # produces GIBBERISH because token IDs don't align with the fine-tuned weights.
        tokenizer = None
        tokenizer_errors: list[str] = []
        source_path = Path(source)
        local_tokenizer_json = source_path / "tokenizer.json"

        strategies = ["pretrained_fast_from_tokenizer_json", "auto_fast_local", "t5_fast_local"]
        for strategy in strategies:
            try:
                if strategy == "pretrained_fast_from_tokenizer_json":
                    if not local_tokenizer_json.exists():
                        raise FileNotFoundError(f"no tokenizer.json at {local_tokenizer_json}")
                    from transformers import PreTrainedTokenizerFast
                    tokenizer = PreTrainedTokenizerFast(
                        tokenizer_file=str(local_tokenizer_json),
                        model_max_length=MAX_INPUT_TOKENS,
                        pad_token="<pad>",
                        eos_token="</s>",
                        unk_token="<unk>",
                        additional_special_tokens=[f"<extra_id_{i}>" for i in range(100)],
                    )
                elif strategy == "auto_fast_local":
                    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
                elif strategy == "t5_fast_local":
                    from transformers import T5TokenizerFast
                    tokenizer = T5TokenizerFast.from_pretrained(source)
                log.info("Tokenizer loaded via strategy: %s", strategy)
                break
            except Exception as e:
                tokenizer_errors.append(f"{strategy}: {e}")
                tokenizer = None

        if tokenizer is None:
            log.error(
                "All LOCAL tokenizer strategies failed. Refusing to fall back to "
                "the HF-base tokenizer because token IDs would not align with the "
                "fine-tuned model — output would be gibberish. Errors: %s",
                tokenizer_errors,
            )
            return

        try:
            self._tokenizer = tokenizer
            self._model = T5ForConditionalGeneration.from_pretrained(source)
            self._model.eval()
        except Exception as e:
            log.error("Failed to load generator model weights: %s", e)
            self._tokenizer = None
            self._model = None
            return
        self._source = source
        log.info(
            "Generator ready (source=%s, loaded in %.1fs, params=%.0fM)",
            source, time.time() - t0,
            sum(p.numel() for p in self._model.parameters()) / 1e6,
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def ready(self) -> bool:
        return self._model is not None

    def source(self) -> str:
        return self._source

    def build_prompt(
        self,
        user_text: str,
        intent: str,
        chunks: List[Dict[str, Any]],
        lang: str = "en",
    ) -> str:
        """Assemble the flan-t5 input string. MUST match the format used
        during Colab fine-tuning (SheriaBot_Generator_Colab.ipynb)."""
        ctx_parts = []
        for c in chunks[:5]:
            citation = c.get("citation", "")
            text = c.get("text", "")
            if not text:
                continue
            snippet = text[:400]
            ctx_parts.append(f"[{citation}] {snippet}" if citation else snippet)
        context = " | ".join(ctx_parts) if ctx_parts else "no retrieved statute"

        lang_hint = "Answer in English." if lang == "en" else "Jibu kwa Kiswahili."
        return (
            f"{lang_hint}\n"
            f"intent: {intent} | "
            f"context: {context} | "
            f"question: {user_text}"
        )

    def generate(
        self,
        user_text: str,
        intent: str,
        chunks: List[Dict[str, Any]],
        lang: str = "en",
    ) -> Optional[str]:
        """Produce a natural-language answer or None if the model isn't loaded."""
        if not self._tried_load:
            self._try_load()
        if self._model is None or self._tokenizer is None:
            return None

        prompt = self.build_prompt(user_text, intent, chunks, lang=lang)

        import torch
        # T5 doesn't use token_type_ids — PreTrainedTokenizerFast emits them
        # by default, and .generate() then complains. Explicitly disable.
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
            return_token_type_ids=False,
        )
        # Defensive: strip token_type_ids in case a tokenizer ignores the flag.
        inputs.pop("token_type_ids", None)
        with torch.no_grad():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=MAX_OUTPUT_TOKENS,
                num_beams=4,
                do_sample=False,
                repetition_penalty=1.2,
                early_stopping=True,
            )
        return self._tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()


# --- singleton --------------------------------------------------------------
_gen: Optional[_Generator] = None


def get_generator() -> _Generator:
    global _gen
    if _gen is None:
        _gen = _Generator()
    return _gen


def generate(
    user_text: str,
    intent: str,
    chunks: List[Dict[str, Any]],
    lang: str = "en",
) -> Optional[str]:
    """Public helper — equivalent to get_generator().generate(...)."""
    return get_generator().generate(user_text, intent, chunks, lang=lang)


def ready() -> bool:
    """Force-load and report whether the generator is usable."""
    g = get_generator()
    if not g._tried_load:
        g._try_load()
    return g.ready()
