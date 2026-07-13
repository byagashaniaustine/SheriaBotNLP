"""Graph generation — renders PNGs into the graphs/ folder.

Each function targets one NLP stage from the pipeline.  The main FastAPI app
exposes them behind /graphs/generate/{stage}.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk

from config import GRAPHS_DIR
from pipeline import STATE, RANDOM_SEED, get_hf_tokenizer, clean_text

sns.set_style("whitegrid")


def _save(fig_name: str) -> Path:
    p = GRAPHS_DIR / f"{fig_name}.png"
    plt.tight_layout()
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    return p


# -----------------------------------------------------------
# Stage 4 — EDA (4 graphs)
# -----------------------------------------------------------
def gen_stage04() -> Dict[str, str]:
    df = STATE["df"]
    out = {}

    # 4.1 class distribution
    plt.figure(figsize=(11, 6))
    df["intent"].value_counts().plot(kind="barh", color="steelblue", edgecolor="black")
    plt.title("Sheria-Bot Intent Distribution")
    plt.xlabel("count"); plt.ylabel("intent"); plt.gca().invert_yaxis()
    out["class_distribution"] = str(_save("stage04_class_distribution"))

    # 4.2 language pie
    plt.figure(figsize=(6, 5))
    df["lang"].value_counts().plot(kind="pie", autopct="%1.1f%%",
                                   colors=["#080","#c00","#c60"], startangle=90)
    plt.title("Language Distribution"); plt.ylabel("")
    out["language_distribution"] = str(_save("stage04_language_distribution"))

    # 4.3 length by language
    plt.figure(figsize=(8, 4))
    sns.boxplot(x="lang", y="length", data=df, palette=["steelblue","darkorange","teal"])
    plt.title("Utterance Length by Language"); plt.xlabel("language"); plt.ylabel("characters")
    out["length_by_language"] = str(_save("stage04_length_by_language"))

    # 4.4 data quality
    stats = pd.Series({
        "missing_text":   df["text"].isna().sum(),
        "missing_intent": df["intent"].isna().sum(),
        "duplicates":     df.duplicated(subset=["clean"]).sum(),
        "clean_rows":     len(df),
    })
    plt.figure(figsize=(8, 3.5))
    stats.plot.bar(color=["#c00","#c00","#c60","#080"], edgecolor="black")
    plt.title("Data Quality Overview"); plt.ylabel("count"); plt.xticks(rotation=15)
    out["data_quality"] = str(_save("stage04_data_quality"))

    return out


# -----------------------------------------------------------
# Stage 11 — POS
# -----------------------------------------------------------
def gen_stage11() -> Dict[str, str]:
    df = STATE["df"]
    tags = []
    for text in df["clean"].sample(500, random_state=RANDOM_SEED):
        tags.extend(t for _, t in pos_tag(word_tokenize(text)))
    top = pd.Series(tags).value_counts().head(15)

    plt.figure(figsize=(10, 4))
    top.plot.bar(color="teal", edgecolor="black")
    plt.title("Top 15 Part-of-Speech Tags"); plt.xlabel("POS tag"); plt.ylabel("count")
    plt.xticks(rotation=45, ha="right")
    return {"pos_distribution": str(_save("stage11_pos_distribution"))}


# -----------------------------------------------------------
# Stage 12 — NER
# -----------------------------------------------------------
def gen_stage12() -> Dict[str, str]:
    df = STATE["df"]
    counts: Counter = Counter()
    for text in df["text"].sample(400, random_state=RANDOM_SEED):
        try:
            tree = ne_chunk(pos_tag(word_tokenize(str(text))))
        except Exception:
            continue
        for sub in tree:
            if hasattr(sub, "label"):
                counts[sub.label()] += 1
    ent_series = pd.Series(counts).sort_values(ascending=False)

    plt.figure(figsize=(8, 4))
    ent_series.plot.bar(color="purple", edgecolor="black")
    plt.title("Named Entity Types Detected"); plt.xlabel("entity type"); plt.ylabel("count")
    return {"ner_types": str(_save("stage12_ner_types"))}


# -----------------------------------------------------------
# Stage 13 — Text visualisation
# -----------------------------------------------------------
def gen_stage13() -> Dict[str, str]:
    df = STATE["df"]; STOP = STATE["STOP"]
    all_tokens = []
    for txt in df["clean"]:
        all_tokens.extend(w for w in word_tokenize(txt) if w.isalpha() and w not in STOP)

    out = {}

    # 13.1 top 20 words
    top20 = pd.DataFrame(Counter(all_tokens).most_common(20), columns=["word","count"])
    plt.figure(figsize=(11, 4))
    plt.bar(top20["word"], top20["count"], color="steelblue", edgecolor="black")
    plt.title("Top 20 Word Frequencies (bilingual, stop-words removed)")
    plt.xticks(rotation=45, ha="right")
    out["top_words"] = str(_save("stage13_top_words"))

    # 13.2 word cloud
    from wordcloud import WordCloud
    wc = WordCloud(background_color="white", width=1000, height=500,
                   colormap="viridis", max_words=200).generate(" ".join(all_tokens))
    plt.figure(figsize=(11, 5))
    plt.imshow(wc, interpolation="bilinear"); plt.axis("off")
    plt.title("Sheria-Bot Vocabulary — Word Cloud")
    out["wordcloud"] = str(_save("stage13_wordcloud"))

    # 13.3 top words by intent (top 2)
    def top_intent_words(rows, n=15):
        ws = []
        for r in rows:
            ws.extend(w for w in word_tokenize(r) if w.isalpha() and w not in STOP)
        return pd.DataFrame(Counter(ws).most_common(n), columns=["word","count"])

    top_intents = df["intent"].value_counts().head(2).index.tolist()
    a = top_intent_words(df[df.intent == top_intents[0]].clean)
    b = top_intent_words(df[df.intent == top_intents[1]].clean)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].barh(a["word"], a["count"], color="#080", edgecolor="black")
    ax[0].set_title(f"Top Words — {top_intents[0]}"); ax[0].invert_yaxis()
    ax[1].barh(b["word"], b["count"], color="#c00", edgecolor="black")
    ax[1].set_title(f"Top Words — {top_intents[1]}"); ax[1].invert_yaxis()
    out["top_words_by_intent"] = str(_save("stage13_top_words_by_intent"))

    return out


# -----------------------------------------------------------
# Stage 14 — Features
# -----------------------------------------------------------
def gen_stage14() -> Dict[str, str]:
    out = {}
    tfidf_vec = STATE["tfidf_vec"]; X_tfidf = STATE["X_tfidf"]; bow_vec = STATE["bow_vec"]
    if any(v is None for v in (tfidf_vec, X_tfidf, bow_vec)):
        raise RuntimeError("Features not built yet — call /pipeline/features first.")

    tfidf_sum = np.asarray(X_tfidf.sum(axis=0)).ravel()
    top_idx = tfidf_sum.argsort()[::-1][:20]
    terms = np.array(tfidf_vec.get_feature_names_out())[top_idx]
    scores = tfidf_sum[top_idx]

    plt.figure(figsize=(11, 4))
    plt.bar(terms, scores, color="darkred", edgecolor="black")
    plt.title("Top 20 TF-IDF Weighted Terms"); plt.xticks(rotation=45, ha="right")
    out["tfidf_top_terms"] = str(_save("stage14_tfidf_top_terms"))

    plt.figure(figsize=(6, 3.5))
    pd.Series({"BoW  (CountVectorizer)": len(bow_vec.get_feature_names_out()),
               "TF-IDF (TfidfVectorizer)": len(tfidf_vec.get_feature_names_out())}
              ).plot.bar(color=["steelblue","darkred"], edgecolor="black")
    plt.title("Feature-Vocabulary Size"); plt.xticks(rotation=0); plt.ylabel("number of features")
    out["vocab_size"] = str(_save("stage14_vocab_size"))
    return out


# -----------------------------------------------------------
# Stage 15 — Word2Vec 2D PCA
# -----------------------------------------------------------
def gen_stage15() -> Dict[str, str]:
    w2v = STATE["w2v"]; STOP = STATE["STOP"]
    if w2v is None:
        raise RuntimeError("Word2Vec not trained — call /pipeline/train first.")

    words = [w for w in w2v.wv.key_to_index if w.isalpha() and w not in STOP][:100]
    vectors = np.array([w2v.wv[w] for w in words])
    coords  = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(vectors)

    plt.figure(figsize=(12, 9))
    plt.scatter(coords[:, 0], coords[:, 1], s=25, c="steelblue")
    for i, w in enumerate(words):
        plt.annotate(w, (coords[i, 0], coords[i, 1]), fontsize=9,
                     xytext=(3, 3), textcoords="offset points")
    plt.title("Word2Vec Embeddings — 2D PCA (bilingual)")
    plt.xlabel("PCA component 1"); plt.ylabel("PCA component 2")
    return {"word2vec_pca": str(_save("stage15_word2vec_pca"))}


# -----------------------------------------------------------
# Stage 16 — Tokenisation
# -----------------------------------------------------------
def gen_stage16() -> Dict[str, str]:
    df = STATE["df"]
    tok = get_hf_tokenizer()

    word_lens = df["clean"].apply(lambda t: len(word_tokenize(t)))
    enc = tok(df["clean"].tolist(), truncation=False)["input_ids"]
    sub_lens = pd.Series([len(x) for x in enc])

    out = {}

    plt.figure(figsize=(11, 4))
    plt.hist(word_lens, bins=40, alpha=0.65, label="NLTK words",   color="steelblue", edgecolor="black")
    plt.hist(sub_lens,  bins=40, alpha=0.65, label="BERT subword", color="darkorange", edgecolor="black")
    plt.legend(); plt.title("Tokens per Utterance — Words vs Subwords")
    plt.xlabel("tokens per utterance"); plt.ylabel("frequency")
    out["word_vs_subword"] = str(_save("stage16_word_vs_subword"))

    ratio = sub_lens / word_lens.replace(0, np.nan)
    plt.figure(figsize=(9, 4))
    plt.hist(ratio.dropna(), bins=30, color="teal", edgecolor="black")
    plt.axvline(ratio.median(), color="crimson", linestyle="--",
                label=f"median = {ratio.median():.2f}")
    plt.legend(); plt.title("Subword / Word Token Ratio")
    plt.xlabel("subword tokens per NLTK word"); plt.ylabel("utterances")
    out["subword_ratio"] = str(_save("stage16_subword_ratio"))

    return out


# -----------------------------------------------------------
# Convenience — generate everything
# -----------------------------------------------------------
STAGE_MAP = {
    "stage04": gen_stage04,
    "stage11": gen_stage11,
    "stage12": gen_stage12,
    "stage13": gen_stage13,
    "stage14": gen_stage14,
    "stage15": gen_stage15,
    "stage16": gen_stage16,
}


def generate(stage: str) -> Dict[str, str]:
    if stage == "all":
        result = {}
        for name, fn in STAGE_MAP.items():
            try:
                result[name] = fn()
            except Exception as e:
                result[name] = {"error": str(e)}
        return result
    if stage not in STAGE_MAP:
        raise ValueError(f"Unknown stage {stage}; valid: {list(STAGE_MAP)+['all']}")
    return STAGE_MAP[stage]()
