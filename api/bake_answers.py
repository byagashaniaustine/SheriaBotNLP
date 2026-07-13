"""One-off script: read 05_NLU_Knowledge_Map.csv and produce artifacts/answer_bank.json.

Rationale: at runtime we only need one canned response per (intent, lang) — not
the 2,700-row NLU map. Baking to JSON means the deployed WhatsApp bot only
needs artifacts/ on disk; you can delete data/ entirely and the bot still works.

Run once after any change to the NLU map:
    python bake_answers.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE      = Path(__file__).resolve().parent
DATA_DIR  = HERE.parent / "data"
ARTIFACTS = HERE / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

NLU_CANDIDATES = [
    DATA_DIR / "05_nlu_knowledge_mapping.csv",   # 5M dataset
    DATA_DIR / "05_NLU_Knowledge_Map.csv",       # small dataset
]
OUT = ARTIFACTS / "answer_bank.json"


def main() -> None:
    nlu_csv = next((p for p in NLU_CANDIDATES if p.exists()), None)
    if nlu_csv is None:
        raise FileNotFoundError(
            f"None of {[p.name for p in NLU_CANDIDATES]} found in {DATA_DIR}"
        )
    print(f"Reading {nlu_csv}")
    df = pd.read_csv(nlu_csv)
    bank: dict[str, dict[str, dict[str, str]]] = {}
    for (intent, lang), grp in df.groupby(["parsed_intent", "lang"]):
        first = grp.iloc[0]
        bank.setdefault(intent, {})[lang] = {
            "response": str(first["solution_response"]).strip(),
            "citation": str(first.get("citation", "")).strip(),
        }
    OUT.write_text(json.dumps(bank, ensure_ascii=False, indent=2))
    n_intents = len(bank)
    n_entries = sum(len(v) for v in bank.values())
    print(f"Wrote {OUT}")
    print(f"  intents: {n_intents}")
    print(f"  entries: {n_entries}")
    print(f"  size:    {OUT.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
