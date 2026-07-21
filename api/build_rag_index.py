"""Build the RAG index for SheriaBot.

Reads two sources of legal knowledge, deduplicates, embeds every chunk with
the multilingual sentence-transformer, and writes two artifacts:

    api/artifacts/rag_index.npz     — float32 embeddings [n, 384]  (L2-normalised)
    api/artifacts/rag_chunks.json   — the chunks aligned to those rows

SOURCE 1 (required)
    api/artifacts/law_chunks_seed.json
        The hand-curated ~40 statute-section chunks. Each has text_en,
        text_sw, statute, section, topic, citation.

SOURCE 2 (optional — for a bigger corpus)
    data/02_legal_terms.csv
        The 400k-row generated corpus. This script downsamples it and
        deduplicates by (term, definition) so we don't drown the seed
        chunks in near-duplicates. Skip by passing --no-csv.

The embed model:
    sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
        Small (118 MB), multilingual (50+ langs incl. Swahili), 384-dim
        vectors. First run downloads it from HF; subsequent runs use cache.

Usage:
    cd "sheriabot NLP/api"
    pip install sentence-transformers pandas numpy
    python build_rag_index.py                      # seed + a CSV sample
    python build_rag_index.py --no-csv             # seed only
    python build_rag_index.py --csv-sample 20000   # a bigger CSV sample

After it runs, api/artifacts/rag_index.npz + rag_chunks.json exist and
api/rag.py loads them at bot startup.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARTIFACTS = HERE / "artifacts"
SEED_JSON = ARTIFACTS / "law_chunks_seed.json"
CSV_PATH  = ROOT / "data" / "02_legal_terms.csv"
OUT_NPZ   = ARTIFACTS / "rag_index.npz"
OUT_JSON  = ARTIFACTS / "rag_chunks.json"

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# --------------------------------------------------------------------------
# Chunk loaders
# --------------------------------------------------------------------------
def load_seed_chunks() -> List[Dict[str, Any]]:
    """Load the hand-curated statute chunks."""
    if not SEED_JSON.exists():
        raise FileNotFoundError(
            f"{SEED_JSON} not found. This is the required minimum RAG "
            "knowledge base — do not delete it before running this script."
        )
    payload = json.loads(SEED_JSON.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    for c in chunks:
        c.setdefault("source", "seed")
    print(f"[build] loaded {len(chunks)} seed chunks from {SEED_JSON.name}")
    return chunks


def load_csv_chunks(sample_n: int) -> List[Dict[str, Any]]:
    """Load a deduplicated sample from data/02_legal_terms.csv."""
    if not CSV_PATH.exists():
        print(f"[build] {CSV_PATH.name} not found — skipping CSV augmentation.")
        return []

    try:
        import pandas as pd
    except ImportError:
        print("[build] pandas not installed — skipping CSV augmentation.")
        return []

    print(f"[build] reading {CSV_PATH.name} (this can take a moment)...")
    df = pd.read_csv(CSV_PATH)
    print(f"[build]   full CSV rows: {len(df):,}")

    # Detect the columns the 5M dataset uses. Typical field names:
    #   term, definition, statute_citation, lang
    # Fall back gracefully if column names differ.
    text_col = next((c for c in ["definition", "explanation", "description",
                                 "solution_response", "text"] if c in df.columns), None)
    key_col  = next((c for c in ["term", "concept", "label", "topic"]
                     if c in df.columns), None)
    cite_col = next((c for c in ["statute_citation", "citation", "source"]
                     if c in df.columns), None)
    lang_col = "lang" if "lang" in df.columns else None
    if text_col is None:
        print(f"[build] could not find a text column in CSV — columns={list(df.columns)[:8]}")
        return []
    print(f"[build]   using columns: key={key_col}, text={text_col}, "
          f"cite={cite_col}, lang={lang_col}")

    df = df[df[text_col].notna() & (df[text_col].str.len() > 40)]

    # Deduplicate by (key, first 80 chars of text) so we don't get 400k
    # near-identical rows.
    df["_dedup_key"] = df[text_col].str.slice(0, 80)
    if key_col:
        df["_dedup_key"] = df[key_col].astype(str) + "|" + df["_dedup_key"]
    df = df.drop_duplicates(subset="_dedup_key")
    print(f"[build]   after dedup: {len(df):,} unique rows")

    if len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42)
        print(f"[build]   downsampled to {len(df):,} rows")

    chunks: List[Dict[str, Any]] = []
    for i, row in df.iterrows():
        lang = str(row[lang_col]).lower() if lang_col else "en"
        text = str(row[text_col]).strip()
        entry: Dict[str, Any] = {
            "id":     f"csv_{i}",
            "topic":  str(row[key_col]) if key_col else "",
            "citation": str(row[cite_col]) if cite_col and pd.notna(row[cite_col]) else "",
            "source": "csv",
        }
        entry[f"text_{lang}" if lang in ("en", "sw") else "text_en"] = text
        chunks.append(entry)
    print(f"[build] loaded {len(chunks)} CSV chunks")
    return chunks


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------
def make_texts_for_embedding(chunks: List[Dict[str, Any]]) -> List[str]:
    """For each chunk build ONE embedding-input string that combines every
    available language + topic + citation. That way a query in SW retrieves
    an EN chunk about the same statute, and vice versa.
    """
    out = []
    for c in chunks:
        parts: List[str] = []
        if c.get("topic"):     parts.append(str(c["topic"]))
        if c.get("statute"):   parts.append(str(c["statute"]))
        if c.get("section"):   parts.append(str(c["section"]))
        if c.get("text_en"):   parts.append(str(c["text_en"]))
        if c.get("text_sw"):   parts.append(str(c["text_sw"]))
        if c.get("citation"):  parts.append(str(c["citation"]))
        out.append(" | ".join(parts))
    return out


def embed_chunks(chunks: List[Dict[str, Any]], batch_size: int = 32) -> np.ndarray:
    """Embed every chunk with the multilingual sentence-transformer."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is not installed. Run:\n"
            "  pip install sentence-transformers\n"
        )
    print(f"[build] loading {EMBED_MODEL_NAME}...")
    t0 = time.time()
    model = SentenceTransformer(EMBED_MODEL_NAME)
    print(f"[build]   loaded in {time.time() - t0:.1f}s")

    texts = make_texts_for_embedding(chunks)
    print(f"[build] embedding {len(texts):,} chunks (batch={batch_size})...")
    t0 = time.time()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,          # L2-normalise so dot == cosine
        show_progress_bar=True,
    ).astype(np.float32)
    print(f"[build]   embedded in {time.time() - t0:.1f}s "
          f"— shape={vecs.shape}, dtype={vecs.dtype}")
    return vecs


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Build the SheriaBot RAG index.")
    ap.add_argument("--no-csv", action="store_true",
                    help="Skip the CSV augmentation; embed only seed chunks.")
    ap.add_argument("--csv-sample", type=int, default=5000,
                    help="Max rows to sample from 02_legal_terms.csv (default: 5000).")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Sentence-transformer batch size (default: 32).")
    args = ap.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    chunks = load_seed_chunks()
    if not args.no_csv:
        chunks.extend(load_csv_chunks(args.csv_sample))

    print(f"\n[build] total chunks to embed: {len(chunks):,}")
    if not chunks:
        print("[build] no chunks — nothing to build.")
        return 1

    vecs = embed_chunks(chunks, batch_size=args.batch_size)

    np.savez_compressed(OUT_NPZ, embeddings=vecs)
    OUT_JSON.write_text(json.dumps(chunks, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n[build] wrote {OUT_NPZ.relative_to(HERE.parent)} "
          f"({OUT_NPZ.stat().st_size/1024:.1f} KB)")
    print(f"[build] wrote {OUT_JSON.relative_to(HERE.parent)} "
          f"({OUT_JSON.stat().st_size/1024:.1f} KB)")
    print("\n[build] Done.  api/rag.py will pick these up on next boot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
