"""Runtime paths + API metadata for the Sheria-Bot WhatsApp service.

The runtime API is BERT-only: it loads the fine-tuned DistilBERT model from
`../sheriabot_bert_model/` and the baked answer bank from `artifacts/`.
It never reads the raw CSVs in `../data/` and never touches `../graphs/` —
those live independently for training and reporting.
"""
from pathlib import Path

# Load api/.env into os.environ BEFORE anything else imports env vars.
# Silent if python-dotenv isn't installed (production deploys usually inject
# env vars directly via the platform).
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# api/ lives inside "sheriabot NLP/"; the trained BERT model sits next to it.
ROOT           = Path(__file__).resolve().parent.parent   # -> sheriabot NLP/
ARTIFACTS      = Path(__file__).resolve().parent / "artifacts"
BERT_MODEL_DIR = ROOT / "sheriabot_bert_model"

ARTIFACTS.mkdir(exist_ok=True)

# API meta
API_TITLE   = "Sheria-Bot WhatsApp API"
API_VERSION = "2.0.0"
API_DESC    = ("WhatsApp-facing Tanzania employment-law assistant. "
               "Classifies inbound messages with a fine-tuned multilingual "
               "DistilBERT, generates a response from the baked answer bank, "
               "and replies via the Meta WhatsApp Cloud API.")
