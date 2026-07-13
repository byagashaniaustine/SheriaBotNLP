#!/usr/bin/env python3
"""
SheriaBot NLP — Graphs by Stage (script version)

Runs the same pipeline as SheriaBot_NLP_Graphs.ipynb but as a single .py script.
Saves every graph to ./graphs/ next to this script.

Usage:
    cd "sheriabot NLP"
    python SheriaBot_NLP_Graphs.py
"""

# # Sheria-Bot NLP — Graphs by Stage**Dataset:** Sheria-Bot Curriculum, sheet `04_Intents` — 8,100 bilingual (Swahili + English) Tanzanian employment-law questions labelled with one of 16 intent classes.**Task:** apply the classical NLP pipeline (stages 4, 11, 12, 13, 14, 15, 16) to legal text — analogous to what the sample IMDB notebook does for movie reviews.**Every graph is saved to** `../graphs/` **as a PNG** so you can drop it straight into your Word report.---## Notes for a bilingual corpus- Text is **Swahili + English + code-switched** (some sentences mix both).- English NLTK tools (POS, NER, stemming) will only make sense on English tokens. Swahili tokens will still appear in vocab / frequency graphs but won't have meaningful POS or NER tags.- Word2Vec is language-agnostic — it will learn that Swahili and English translation pairs cluster together, which is a great graph.- The transformer stage uses `distilbert-base-multilingual-cased` (the multilingual version) which handles Swahili natively.

# ## Setup

# Install libraries if missing (uncomment once):
# !pip install nltk wordcloud gensim transformers scikit-learn matplotlib seaborn pandas

import os
os.environ["USE_TF"] = "0"

import warnings
warnings.filterwarnings("ignore")

import re, string
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Paths ---
DATA_DIR   = Path("data")           # relative to this notebook
GRAPHS_DIR = Path("graphs")
GRAPHS_DIR.mkdir(exist_ok=True)

def save_and_show(name):
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  saved -> {GRAPHS_DIR / (name + '.png')}")

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
print("Setup ready. Graphs will save into:", GRAPHS_DIR.resolve())

import nltk
for r in ["punkt", "punkt_tab", "stopwords", "wordnet",
          "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
          "maxent_ne_chunker", "maxent_ne_chunker_tab", "words", "omw-1.4"]:
    nltk.download(r, quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk

# English stop words from NLTK + a small hand-curated Swahili list
STOP_EN = set(stopwords.words("english"))
STOP_SW = {
    "na","ya","wa","ni","kwa","za","la","ku","katika","hii","hiyo","kama",
    "tu","pia","yangu","yako","yake","yenu","yao","ndani","nje","juu","chini",
    "hapa","huko","huku","kabisa","tayari","bado","siku","mwezi","mwaka",
    "mimi","wewe","yeye","sisi","nyinyi","wao","mtu","watu","kitu","vitu",
    "je","nini","gani","lini","wapi","vipi","kiasi","ngapi","nani",
}
STOPWORDS = STOP_EN | STOP_SW
print(f"Stopwords: {len(STOP_EN)} EN + {len(STOP_SW)} SW = {len(STOPWORDS)} total")

# Load the Sheria-Bot Intents CSV
df = pd.read_csv(DATA_DIR / "04_Intents.csv")
print("Rows:", len(df), "Cols:", list(df.columns))

# We use 'utterance' as the text and 'intent_label' as the class
# 'lang' tells us which language the row is in ('sw', 'en', or 'mixed')
df = df.rename(columns={"utterance": "text", "intent_label": "intent"})
df["length"] = df["text"].astype(str).apply(len)
df.head(3)

# Text cleaning — same idea as IMDB but keep letters + Swahili accents
def clean_text(t):
    t = re.sub(r"\[.*?\]", " ", str(t))                # drop [ ... ] annotations
    t = re.sub(r"http\S+|www\S+", " ", t)              # drop URLs
    t = re.sub(r"[0-9]+", " ", t.lower())                # drop digits
    t = re.sub(r"[^a-z\u00c0-\u017f\s]", " ", t)      # keep letters + accented chars
    return re.sub(r"\s+", " ", t).strip()

df["clean"] = df["text"].apply(clean_text)
print("Cleaned. Sample rows:")
df[["text","clean","lang","intent"]].sample(3, random_state=42)

# ---## Stage 4 — Exploratory Data Analysis (EDA)Four graphs:1. Class distribution (all 16 intents)2. Language distribution3. Utterance length by language4. Data quality overview

# Graph 4.1 — Class distribution
intent_counts = df["intent"].value_counts()
plt.figure(figsize=(11, 6))
intent_counts.plot(kind="barh", color="steelblue", edgecolor="black")
plt.title("Sheria-Bot Intent Distribution (16 classes)")
plt.xlabel("count"); plt.ylabel("intent")
plt.gca().invert_yaxis()
save_and_show("stage04_class_distribution")

# Graph 4.2 — Language distribution
lang_counts = df["lang"].value_counts()
plt.figure(figsize=(6, 5))
plt.pie(lang_counts.values, labels=lang_counts.index, autopct="%1.1f%%",
        colors=["#080","#c00","#c60"], startangle=90)
plt.title("Language Distribution")
save_and_show("stage04_language_distribution")

# Graph 4.3 — Length by language (boxplot)
plt.figure(figsize=(8, 4))
sns.boxplot(x="lang", y="length", data=df, palette=["steelblue","darkorange","teal"])
plt.title("Utterance Length by Language (characters)")
plt.xlabel("language"); plt.ylabel("characters")
save_and_show("stage04_length_by_language")

# Graph 4.4 — Data quality overview
stats = pd.Series({
    "missing_text":   df["text"].isna().sum(),
    "missing_intent": df["intent"].isna().sum(),
    "duplicates":     df.duplicated(subset=["clean"]).sum(),
    "clean_rows":     len(df),
})
plt.figure(figsize=(8, 3.5))
stats.plot.bar(color=["#c00","#c00","#c60","#080"], edgecolor="black")
plt.title("Data Quality Overview")
plt.ylabel("count"); plt.xticks(rotation=15)
save_and_show("stage04_data_quality")

# ---## Stage 11 — POS TaggingWe run NLTK's English POS tagger on a random sample. Swahili tokens get imperfect tags (usually classified as `NN` — noun) but the graph still shows the grammatical make-up of the corpus.

# Graph 11 — POS tag distribution
all_tags = []
for text in df["clean"].sample(500, random_state=42):
    all_tags.extend(tag for _, tag in pos_tag(word_tokenize(text)))

top_tags = pd.Series(all_tags).value_counts().head(15)
plt.figure(figsize=(10, 4))
top_tags.plot.bar(color="teal", edgecolor="black")
plt.title("Top 15 Part-of-Speech Tags in Sheria-Bot Corpus")
plt.xlabel("POS tag"); plt.ylabel("count")
plt.xticks(rotation=45, ha="right")
save_and_show("stage11_pos_distribution")

POS_MEANING = {
    "NN":"noun (singular)", "NNS":"noun (plural)", "NNP":"proper noun",
    "JJ":"adjective", "VB":"verb (base)", "VBD":"verb (past)", "VBG":"verb (-ing)",
    "VBN":"verb (past participle)", "VBP":"verb (present)", "VBZ":"verb (3rd sg)",
    "RB":"adverb", "IN":"preposition", "DT":"determiner", "PRP":"pronoun",
    "CC":"conjunction", "CD":"cardinal number", "TO":"to",
}
print("\nWhat the top tags mean:")
for t, count in top_tags.items():
    print(f"  {t:5s} {count:>5}  -> {POS_MEANING.get(t, '?')}")

# ---## Stage 12 — Named Entity Recognition (NER)NLTK's NER only recognises English proper nouns. It will pick out `CMA`, `ELRA`, `Dar es Salaam`, `John`, `Mary` — plus some Swahili proper nouns that happen to be capitalised in the source.

# Graph 12 — Named entity types
entity_counts = Counter()
entity_examples = {}          # keep a few examples of each type

for text in df["text"].sample(400, random_state=42):
    try:
        tree = ne_chunk(pos_tag(word_tokenize(str(text))))
    except Exception:
        continue
    for sub in tree:
        if hasattr(sub, "label"):
            lbl = sub.label()
            entity_counts[lbl] += 1
            if lbl not in entity_examples:
                entity_examples[lbl] = " ".join(w for w, _ in sub.leaves())

entity_series = pd.Series(entity_counts).sort_values(ascending=False)

plt.figure(figsize=(8, 4))
entity_series.plot.bar(color="purple", edgecolor="black")
plt.title("Named Entity Types Detected in Sheria-Bot Corpus")
plt.xlabel("entity type"); plt.ylabel("count")
plt.xticks(rotation=0)
save_and_show("stage12_ner_types")

print("\nExamples of each entity type:")
for lbl, ex in entity_examples.items():
    print(f"  {lbl:15s}  '{ex}'")

# ---## Stage 13 — Text VisualisationThree graphs:1. Top 20 most frequent tokens (bilingual, stop-words removed)2. Word cloud3. Top words by intent for the two most common intents

# Graph 13.1 — Top 20 word frequencies (bilingual)
all_tokens = []
for txt in df["clean"]:
    all_tokens.extend(w for w in word_tokenize(txt) if w.isalpha() and w not in STOPWORDS)

freq = Counter(all_tokens)
top20 = pd.DataFrame(freq.most_common(20), columns=["word", "count"])

plt.figure(figsize=(11, 4))
plt.bar(top20["word"], top20["count"], color="steelblue", edgecolor="black")
plt.title("Top 20 Word Frequencies (bilingual, stop-words removed)")
plt.xlabel("word"); plt.ylabel("count")
plt.xticks(rotation=45, ha="right")
save_and_show("stage13_top_words")

# Graph 13.2 — Word cloud (bilingual)
from wordcloud import WordCloud
text_blob = " ".join(all_tokens)
wc = WordCloud(background_color="white", width=1000, height=500,
               colormap="viridis", max_words=200).generate(text_blob)
plt.figure(figsize=(11, 5))
plt.imshow(wc, interpolation="bilinear"); plt.axis("off")
plt.title("Sheria-Bot Vocabulary — Word Cloud (Swahili + English)")
save_and_show("stage13_wordcloud")

# Graph 13.3 — Top words by intent (compare two most common intents)
def top_words(rows, n=15):
    ws = []
    for r in rows:
        ws.extend(w for w in word_tokenize(r) if w.isalpha() and w not in STOPWORDS)
    return pd.DataFrame(Counter(ws).most_common(n), columns=["word","count"])

top_intents = intent_counts.head(2).index.tolist()
a = top_words(df[df.intent == top_intents[0]].clean)
b = top_words(df[df.intent == top_intents[1]].clean)

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].barh(a["word"], a["count"], color="#080", edgecolor="black")
ax[0].set_title(f"Top Words — {top_intents[0]}")
ax[0].invert_yaxis()
ax[1].barh(b["word"], b["count"], color="#c00", edgecolor="black")
ax[1].set_title(f"Top Words — {top_intents[1]}")
ax[1].invert_yaxis()
save_and_show("stage13_top_words_by_intent")

# ---## Stage 14 — Feature Engineering (BoW + TF-IDF)Two graphs:1. Top 20 TF-IDF weighted terms2. Vocabulary size: BoW vs TF-IDF

# Graph 14.1 — Top TF-IDF terms
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["clean"], df["intent"], test_size=0.2, random_state=42, stratify=df["intent"]
)

bow_vec   = CountVectorizer(max_features=5000)
bow_vec.fit(X_train_text)

tfidf_vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
X_train_tfidf = tfidf_vec.fit_transform(X_train_text)

tfidf_sum = np.asarray(X_train_tfidf.sum(axis=0)).ravel()
top_idx = tfidf_sum.argsort()[::-1][:20]
top_terms  = np.array(tfidf_vec.get_feature_names_out())[top_idx]
top_scores = tfidf_sum[top_idx]

plt.figure(figsize=(11, 4))
plt.bar(top_terms, top_scores, color="darkred", edgecolor="black")
plt.title("Top 20 TF-IDF Weighted Terms (Sheria-Bot Corpus)")
plt.xlabel("term"); plt.ylabel("summed TF-IDF weight")
plt.xticks(rotation=45, ha="right")
save_and_show("stage14_tfidf_top_terms")

# Graph 14.2 — BoW vs TF-IDF vocab
sizes = pd.Series({
    "BoW  (CountVectorizer)":     len(bow_vec.get_feature_names_out()),
    "TF-IDF (TfidfVectorizer)":   len(tfidf_vec.get_feature_names_out()),
})
plt.figure(figsize=(6, 3.5))
sizes.plot.bar(color=["steelblue", "darkred"], edgecolor="black")
plt.title("Feature-Vocabulary Size")
plt.ylabel("number of features")
plt.xticks(rotation=0)
save_and_show("stage14_vocab_size")

# ---## Stage 15 — Word Embeddings (Word2Vec)One graph: 2-D PCA projection of 100 word vectors. Because the corpus is bilingual, translation-equivalent Swahili and English words should cluster near each other (e.g. `kufukuzwa` near `dismissed`, `mshahara` near `wage`).

# Graph 15 — Word2Vec 2D PCA
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

tokenized_corpus = [word_tokenize(t) for t in df["clean"]]
w2v = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5,
               min_count=5, workers=1, seed=42, epochs=15)

words = [w for w in w2v.wv.key_to_index
         if w.isalpha() and w not in STOPWORDS][:100]
vectors = np.array([w2v.wv[w] for w in words])
coords  = PCA(n_components=2, random_state=42).fit_transform(vectors)

plt.figure(figsize=(12, 9))
plt.scatter(coords[:, 0], coords[:, 1], s=25, c="steelblue")
for i, w in enumerate(words):
    plt.annotate(w, (coords[i, 0], coords[i, 1]),
                 fontsize=9, xytext=(3, 3), textcoords="offset points")
plt.title("Word2Vec Embeddings — 2D PCA (Bilingual Sheria-Bot Corpus)")
plt.xlabel("PCA component 1"); plt.ylabel("PCA component 2")
save_and_show("stage15_word2vec_pca")

# Also print a nice bilingual similarity example
print("\nSimilarity examples:")
for w in ["kufukuzwa", "dismissed", "mshahara", "wage", "cma", "elra"]:
    if w in w2v.wv:
        sims = w2v.wv.most_similar(w, topn=5)
        print(f"  '{w}' most similar -> {[(s, round(sc,3)) for s, sc in sims]}")

# ---## Stage 16 — Transformer Tokenisation (multilingual DistilBERT)We use `distilbert-base-multilingual-cased` because our text is bilingual. Two graphs:1. NLTK word tokens vs BERT subword tokens (histogram)2. Subword / word ratio

# Graph 16.1 — Word vs subword tokens (multilingual)
from transformers import AutoTokenizer

MODEL_NAME = "distilbert-base-multilingual-cased"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

word_lens = df["clean"].apply(lambda t: len(word_tokenize(t)))
enc = tok(df["clean"].tolist(), truncation=False)["input_ids"]
sub_lens = pd.Series([len(x) for x in enc])

plt.figure(figsize=(11, 4))
plt.hist(word_lens, bins=40, alpha=0.65, label="NLTK words",   color="steelblue", edgecolor="black")
plt.hist(sub_lens,  bins=40, alpha=0.65, label="BERT subword", color="darkorange", edgecolor="black")
plt.legend()
plt.title(f"Token Count per Utterance — Words vs Subwords\n(model: {MODEL_NAME})")
plt.xlabel("tokens per utterance"); plt.ylabel("frequency")
save_and_show("stage16_word_vs_subword")

# Graph 16.2 — Ratio (subword / word)
ratio = sub_lens / word_lens.replace(0, np.nan)
plt.figure(figsize=(9, 4))
plt.hist(ratio.dropna(), bins=30, color="teal", edgecolor="black")
plt.axvline(ratio.median(), color="crimson", linestyle="--",
            label=f"median = {ratio.median():.2f}")
plt.legend(); plt.title("Subword / Word Token Ratio (Multilingual DistilBERT)")
plt.xlabel("subword tokens per NLTK word"); plt.ylabel("utterances")
save_and_show("stage16_subword_ratio")

print(f"\nOn average, BERT produces {ratio.mean():.2f} subword tokens per NLTK word.")
print(f"Median: {ratio.median():.2f}")

# ---## Recap — all graphs savedRun this cell to list every PNG that was generated.

for f in sorted(GRAPHS_DIR.glob("*.png")):
    print(f)
print(f"\nTotal: {len(list(GRAPHS_DIR.glob('*.png')))} graphs saved in {GRAPHS_DIR.resolve()}")

