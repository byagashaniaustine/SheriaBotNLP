"""Turn a classified intent (+ the user's original question + session context)
into a natural-language answer suitable for sending back over WhatsApp.

Runtime dependency: ONLY artifacts/answer_bank.json — produced once at
training time by bake_answers.py or the Colab notebook. The runtime bot
never opens a CSV; if answer_bank.json is missing the engine refuses to
start rather than silently degrading to something else.
"""
from __future__ import annotations

import json
import re
from typing import Dict, Any, Optional

from config import ARTIFACTS

ANSWER_BANK_JSON = ARTIFACTS / "answer_bank.json"

_SWAHILI_MARKERS = {
    "na","ya","wa","ni","kwa","za","la","ku","katika","hii","hiyo","nini",
    "gani","lini","wapi","vipi","je","nimefutwa","nifanye","nataka",
    "sijui","siwezi","sitaki","tafadhali","asante","habari","jambo","mambo",
    "nimechoka","nimeumia","mimi","wewe","yeye","sisi","wao","kazi","mshahara",
    "mkataba","likizo","ugonjwa","mimba","mtoto","mwajiri","mfanyakazi",
    "ndio","hapana","sawa","haki","sheria","kesi","cma",
}

CONFIDENCE_FLOOR = 0.35


class AnswerEngine:
    """Loads the answer bank once and answers questions given a classified intent."""

    def __init__(self) -> None:
        self._bank: Dict[str, Dict[str, Dict[str, str]]] = {}
        self._source: str = ""
        self._load_bank()

    def _load_bank(self) -> None:
        if not ANSWER_BANK_JSON.exists():
            raise FileNotFoundError(
                f"{ANSWER_BANK_JSON} not found. This file is a training-time "
                "artifact produced by bake_answers.py (local) or the Colab "
                "training notebook. The runtime bot cannot start without it — "
                "deploy the trained artifacts/ folder alongside the code."
            )
        self._bank = json.loads(ANSWER_BANK_JSON.read_text())
        self._source = "answer_bank.json"

    # ------------------------------------------------------------------
    # language detection — light heuristic sufficient for en/sw switch
    # ------------------------------------------------------------------
    @staticmethod
    def detect_lang(text: str) -> str:
        tokens = re.findall(r"[a-zA-ZÀ-ſ]+", text.lower())
        if not tokens:
            return "en"
        sw_hits = sum(1 for t in tokens if t in _SWAHILI_MARKERS)
        return "sw" if sw_hits >= 1 else "en"

    # ------------------------------------------------------------------
    # small-talk classifiers
    # ------------------------------------------------------------------
    _GREETINGS = {
        "hi", "hello", "hey", "hallo", "yo", "howdy",
        "habari", "mambo", "jambo", "salama", "shikamoo", "vipi", "hodi",
    }
    _THANKS = {
        "thanks", "thank", "ta",
        "asante", "asanteni", "shukrani",
    }
    _AFFIRMATIVE = {"yes", "yeah", "yep", "ok", "okay", "sure", "ndio", "ndiyo", "sawa"}
    _NEGATIVE    = {"no", "nope", "hapana", "la"}
    _GOODBYE     = {"bye", "goodbye", "kwaheri", "baadaye", "tutaonana"}

    @classmethod
    def is_greeting(cls, text: str) -> bool:
        t = text.strip().lower()
        if not t:
            return False
        first = re.split(r"[\s,.!?]", t, maxsplit=1)[0]
        return first in cls._GREETINGS or t in cls._GREETINGS

    @classmethod
    def is_thanks(cls, text: str) -> bool:
        t = text.strip().lower().rstrip("!.")
        return t in cls._THANKS or any(t.startswith(k + " ") for k in cls._THANKS)

    @classmethod
    def is_affirmative(cls, text: str) -> bool:
        return text.strip().lower().rstrip("!.?") in cls._AFFIRMATIVE

    @classmethod
    def is_negative(cls, text: str) -> bool:
        return text.strip().lower().rstrip("!.?") in cls._NEGATIVE

    @classmethod
    def is_goodbye(cls, text: str) -> bool:
        t = text.strip().lower().rstrip("!.")
        return t in cls._GOODBYE

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------
    def answer(
        self,
        user_text: str,
        intent: str,
        confidence: Optional[float] = None,
        session: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        session = session or {}
        lang = session.get("lang") or self.detect_lang(user_text)
        name = (session.get("profile") or {}).get("name")

        # ---- small-talk short-circuits (no legal lookup needed) ----
        if self.is_greeting(user_text):
            return self._reply(self._greeting_text(lang, name, session.get("turn_count", 0)),
                               intent="greeting", lang=lang, source="greeting")

        if self.is_thanks(user_text):
            return self._reply(self._thanks_reply(lang, name),
                               intent="thanks", lang=lang, source="thanks")

        if self.is_goodbye(user_text):
            return self._reply(self._goodbye_reply(lang, name),
                               intent="goodbye", lang=lang, source="goodbye")

        # ---- classifier low-confidence → clarifying question ----
        if confidence is not None and confidence < CONFIDENCE_FLOOR:
            return self._reply(self._low_confidence_reply(lang, name),
                               intent=intent, lang=lang, source="low_confidence_fallback")

        entry = self._bank.get(intent, {}).get(lang) or self._bank.get(intent, {}).get("en")
        if entry is None:
            return self._reply(self._unknown_intent_reply(lang, name),
                               intent=intent, lang=lang, source="unknown_intent_fallback")

        text = entry["response"]
        citation = entry.get("citation", "")
        if citation and citation.lower() not in text.lower():
            text = f"{text}\n\n_Ref: {citation}_"

        if name and not text.startswith(name):
            prefix = f"{name}, " if lang == "en" else f"{name}, "
            text = prefix + text[0].lower() + text[1:] if text else prefix

        # Disclaimer only on substantive legal answers, not small-talk / fallbacks.
        text = f"{text}\n\n{self._disclaimer(lang)}"

        return self._reply(text, intent=intent, lang=lang, source="answer_bank",
                           citation=citation)

    @staticmethod
    def _disclaimer(lang: str) -> str:
        if lang == "sw":
            return ("_Ni taarifa ya jumla, si ushauri wa kisheria. "
                    "Kwa kesi yako maalum, wasiliana na wakili au CMA._")
        return ("_This is general information, not legal advice. "
                "For your specific case, consult a lawyer or CMA._")

    # ------------------------------------------------------------------
    # reply helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _reply(text: str, **extra: Any) -> Dict[str, Any]:
        r = {"text": text}
        r.update(extra)
        return r

    @staticmethod
    def _greeting_text(lang: str, name: Optional[str], turn_count: int) -> str:
        # First contact vs returning user in the same session.
        if turn_count > 0:
            if lang == "sw":
                return (f"Karibu tena{', ' + name if name else ''}. "
                        "Una swali gani la sheria ya ajira?")
            return (f"Welcome back{', ' + name if name else ''}. "
                    "What employment question can I help with?")
        if lang == "sw":
            return ("Habari! Mimi ni Sheria-Bot, msaidizi wako wa sheria ya ajira Tanzania. "
                    "Ninaweza kukusaidia vipi?")
        return ("Hi! I'm Sheria-Bot, your Tanzania employment-law assistant. "
                "What employment question can I help with?")

    @staticmethod
    def _thanks_reply(lang: str, name: Optional[str]) -> str:
        if lang == "sw":
            return f"Karibu{', ' + name if name else ''}. Uko na swali lingine?"
        return f"You're welcome{', ' + name if name else ''}. Any other question?"

    @staticmethod
    def _goodbye_reply(lang: str, name: Optional[str]) -> str:
        if lang == "sw":
            return f"Kwaheri{', ' + name if name else ''}. Kuwa salama."
        return f"Goodbye{', ' + name if name else ''}. Take care."

    @staticmethod
    def _low_confidence_reply(lang: str, name: Optional[str]) -> str:
        if lang == "sw":
            return (f"{name + ', ' if name else ''}"
                    "Sijaelewa vizuri. Tafadhali eleza zaidi — nini kimetokea, lini, "
                    "na unatarajia matokeo gani?")
        return (f"{name + ', ' if name else ''}"
                "I want to make sure I answer correctly — could you tell me a bit more? "
                "What happened, when, and what outcome are you hoping for?")

    @staticmethod
    def _unknown_intent_reply(lang: str, name: Optional[str]) -> str:
        if lang == "sw":
            return (f"{name + ', ' if name else ''}"
                    "Ninaweza kusaidia kuhusu masuala ya sheria ya ajira Tanzania. "
                    "Tafadhali eleza swali lako kwa undani zaidi.")
        return (f"{name + ', ' if name else ''}"
                "I can help with Tanzania employment-law issues. "
                "Please tell me more about your situation.")


# module-level singleton
_engine: Optional[AnswerEngine] = None


def get_engine() -> AnswerEngine:
    global _engine
    if _engine is None:
        _engine = AnswerEngine()
    return _engine
