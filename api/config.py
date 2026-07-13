"""Central config — paths and hyperparameters for the Sheria-Bot NLP API."""
from pathlib import Path

# The api/ folder lives inside "sheriabot NLP/" and data/ + graphs/ sit next to it.
ROOT       = Path(__file__).resolve().parent.parent      # -> sheriabot NLP/
DATA_DIR   = ROOT / "data"
GRAPHS_DIR = ROOT / "graphs"
ARTIFACTS  = Path(__file__).resolve().parent / "artifacts"

GRAPHS_DIR.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)

# Sheria-Bot Intents CSV is the primary text corpus for NLP.
INTENTS_CSV = DATA_DIR / "04_Intents.csv"

# Feature engineering
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE  = (1, 2)
TFIDF_MIN_DF       = 2

# Modelling
TEST_SIZE     = 0.2
RANDOM_SEED   = 42
CV_FOLDS      = 5

# Multilingual model (Swahili + English)
TRANSFORMER_MODEL = "distilbert-base-multilingual-cased"

# API meta
API_TITLE   = "Sheria-Bot NLP API"
API_VERSION = "1.0.0"
API_DESC    = ("REST API exposing the Sheria-Bot NLP pipeline — EDA, POS, NER, "
               "text visualisation, feature engineering, embeddings, tokenisation, "
               "and intent classification.")
