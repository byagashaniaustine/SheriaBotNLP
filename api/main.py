"""FastAPI application — the HTTP surface of the Sheria-Bot NLP pipeline.

Run locally:
    cd "sheriabot NLP/api"
    pip install -r requirements.txt
    uvicorn main:app --reload

Then open:
    http://127.0.0.1:8000/docs      # Swagger UI (interactive testing)
    http://127.0.0.1:8000/redoc     # ReDoc alternative
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from config import (
    API_TITLE, API_VERSION, API_DESC,
    GRAPHS_DIR, INTENTS_CSV,
)
import pipeline
import graphs_gen
import whatsapp
from answer_engine import get_engine
from sessions import (
    get_store, extract_profile, is_introduction, is_short_followup,
)
from schemas import (
    ClassifyRequest, ClassifyResponse, HealthResponse, StatsResponse,
    POSResponse, NERResponse, TopWordsResponse, TfIdfResponse,
    EmbeddingResponse, TokenisationResponse, TrainResponse,
)


# --------------------------------------------------------------------------
# Startup / shutdown lifecycle
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # EDA endpoints need the CSV; the webhook does not. If data/ is absent we
    # skip load_state() and only the /stats /nlp/* endpoints degrade — the
    # WhatsApp flow keeps working off artifacts/ alone.
    if INTENTS_CSV.exists():
        print(f"[startup] loading pipeline state from {INTENTS_CSV}")
        pipeline.load_state()
    else:
        print(f"[startup] {INTENTS_CSV} not found — EDA endpoints disabled, "
              f"webhook still works off artifacts/")

    if pipeline.try_load_saved_classifier():
        print("[startup] loaded classifier from artifacts/")
    else:
        print("[startup] WARNING: no trained classifier — /classify + webhook will error. "
              "Run `python train.py` first.")

    get_engine()
    print("[startup] answer engine ready")
    get_store()
    print("[startup] session store ready")
    yield
    print("[shutdown] bye")


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESC,
    lifespan=lifespan,
)

# Serve /graphs/*.png as static files (so an image URL is direct)
app.mount("/graphs", StaticFiles(directory=str(GRAPHS_DIR)), name="graphs")


# --------------------------------------------------------------------------
# Admin API-key dependency
# --------------------------------------------------------------------------
def require_admin_key(x_api_key: str = Header(None, alias="X-API-Key")) -> None:
    """Gate for anything that mutates state or costs CPU (training, feature build,
    graph generation). Reject if ADMIN_API_KEY is unset — secure-by-default so a
    misconfigured deploy can't be trained by strangers.
    """
    expected = os.getenv("ADMIN_API_KEY", "")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="admin endpoints disabled: ADMIN_API_KEY env var not set on the server",
        )
    if not x_api_key or x_api_key != expected:
        raise HTTPException(
            status_code=401,
            detail="missing or invalid X-API-Key header",
        )


# --------------------------------------------------------------------------
# Root and health
# --------------------------------------------------------------------------
@app.get("/", tags=["meta"])
def root() -> Dict[str, Any]:
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "docs":    "/docs",
        "endpoints": [
            "GET  /health",
            "GET  /stats",
            "POST /classify",
            "POST /pipeline/features",
            "POST /pipeline/train",
            "POST /pipeline/word2vec",
            "GET  /nlp/pos",
            "GET  /nlp/ner",
            "GET  /nlp/topwords",
            "GET  /nlp/tfidf",
            "GET  /nlp/embedding/{word}",
            "GET  /nlp/tokenise",
            "POST /graphs/generate/{stage}",
            "GET  /graphs/list",
            "GET  /graphs/{filename.png}",
            "GET  /webhook   (Meta verification handshake)",
            "POST /webhook   (WhatsApp inbound messages)",
        ],
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    df = pipeline.STATE.get("df")
    return HealthResponse(
        status="ok",
        model_loaded=pipeline.STATE.get("classifier") is not None,
        dataset_rows=(len(df) if df is not None else 0),
        intents_known=(df["intent"].nunique() if df is not None else 0),
        graphs_available=len(list(GRAPHS_DIR.glob("*.png"))),
    )


# --------------------------------------------------------------------------
# Stage 4 — EDA
# --------------------------------------------------------------------------
@app.get("/stats", response_model=StatsResponse, tags=["stage 4 - EDA"])
def stats() -> StatsResponse:
    return StatsResponse(**pipeline.eda_stats())


# --------------------------------------------------------------------------
# Stage 11 — POS
# --------------------------------------------------------------------------
@app.get("/nlp/pos", response_model=POSResponse, tags=["stage 11 - POS"])
def pos(sample: int = Query(300, ge=50, le=2000), top_n: int = Query(15, ge=5, le=40)):
    return POSResponse(**pipeline.pos_distribution(sample=sample, top_n=top_n))


# --------------------------------------------------------------------------
# Stage 12 — NER
# --------------------------------------------------------------------------
@app.get("/nlp/ner", response_model=NERResponse, tags=["stage 12 - NER"])
def ner(sample: int = Query(400, ge=50, le=2000)):
    return NERResponse(**pipeline.ner_counts(sample=sample))


# --------------------------------------------------------------------------
# Stage 13 — Top words
# --------------------------------------------------------------------------
@app.get("/nlp/topwords", response_model=TopWordsResponse, tags=["stage 13 - visualisation"])
def topwords(n: int = Query(20, ge=5, le=100)):
    return TopWordsResponse(**pipeline.top_words(n=n))


# --------------------------------------------------------------------------
# Stage 14 — Features
# --------------------------------------------------------------------------
@app.post("/pipeline/features", response_model=TfIdfResponse, tags=["stage 14 - features"],
          dependencies=[Depends(require_admin_key)])
def features() -> TfIdfResponse:
    return TfIdfResponse(**pipeline.build_features())


@app.get("/nlp/tfidf", response_model=TfIdfResponse, tags=["stage 14 - features"])
def tfidf():
    if pipeline.STATE.get("tfidf_vec") is None:
        raise HTTPException(status_code=409, detail="Features not built yet — POST /pipeline/features first")
    tfidf_vec = pipeline.STATE["tfidf_vec"]; X_tfidf = pipeline.STATE["X_tfidf"]
    import numpy as np
    tfidf_sum = np.asarray(X_tfidf.sum(axis=0)).ravel()
    top_idx = tfidf_sum.argsort()[::-1][:20]
    top_terms = [(str(t), float(s)) for t, s in
                 zip(np.array(tfidf_vec.get_feature_names_out())[top_idx],
                     tfidf_sum[top_idx])]
    return TfIdfResponse(
        top_terms=top_terms,
        vocab_size_bow=len(pipeline.STATE["bow_vec"].get_feature_names_out()),
        vocab_size_tfidf=len(tfidf_vec.get_feature_names_out()),
    )


# --------------------------------------------------------------------------
# Stage 15 — Word2Vec
# --------------------------------------------------------------------------
@app.post("/pipeline/word2vec", tags=["stage 15 - embeddings"],
          dependencies=[Depends(require_admin_key)])
def train_word2vec(vector_size: int = Query(100, ge=32, le=300),
                   epochs: int = Query(15, ge=5, le=100)):
    pipeline.build_word2vec(vector_size=vector_size, epochs=epochs)
    return {"status": "ok",
            "vocab_size": len(pipeline.STATE["w2v"].wv),
            "vector_size": vector_size,
            "epochs": epochs}


@app.get("/nlp/embedding/{word}", response_model=EmbeddingResponse, tags=["stage 15 - embeddings"])
def embedding(word: str, top_k: int = Query(5, ge=1, le=20)):
    try:
        return EmbeddingResponse(**pipeline.embedding_for(word.lower(), top_k=top_k))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


# --------------------------------------------------------------------------
# Stage 16 — Tokenisation
# --------------------------------------------------------------------------
@app.get("/nlp/tokenise", response_model=TokenisationResponse, tags=["stage 16 - tokenisation"])
def tokenise(text: str = Query(..., description="Text to tokenise both ways")):
    return TokenisationResponse(**pipeline.tokenise_compare(text))


# --------------------------------------------------------------------------
# Training & inference
# --------------------------------------------------------------------------
@app.post("/pipeline/train", response_model=TrainResponse, tags=["training"],
          dependencies=[Depends(require_admin_key)])
def train() -> TrainResponse:
    if pipeline.STATE.get("X_tfidf") is None:
        raise HTTPException(status_code=409, detail="Features not built yet — POST /pipeline/features first")
    return TrainResponse(**pipeline.train_classifiers())


@app.post("/classify", response_model=ClassifyResponse, tags=["inference"])
def classify(req: ClassifyRequest) -> ClassifyResponse:
    try:
        return ClassifyResponse(**pipeline.classify_text(req.text))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


# --------------------------------------------------------------------------
# Graph generation + serving
# --------------------------------------------------------------------------
@app.post("/graphs/generate/{stage}", tags=["graphs"],
          dependencies=[Depends(require_admin_key)])
def graphs_generate(stage: str):
    """Generate one stage's graphs (stage04, stage11, stage12, stage13,
    stage14, stage15, stage16, or 'all')."""
    try:
        result = graphs_gen.generate(stage)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"stage": stage, "generated": result}


@app.get("/graphs/list", tags=["graphs"])
def graphs_list():
    files = sorted(GRAPHS_DIR.glob("*.png"))
    return {"count": len(files),
            "graphs": [{"name": f.name,
                        "url":  f"/graphs/{f.name}",
                        "size_kb": round(f.stat().st_size / 1024, 1)}
                       for f in files]}


# --------------------------------------------------------------------------
# WhatsApp Cloud API webhook (Meta)
# --------------------------------------------------------------------------
async def _process_whatsapp_message(sender: str, text: str, message_id: str) -> None:
    """Runs after the webhook has already 200'd.

    Conversational flow:
      0. Dedup: skip if we've already handled this Meta message_id (retry protection).
      1. Load this user's session (history, profile, last intent).
      2. If the message looks like an intro ("My name is …"), extract profile.
      3. If it's a short follow-up ("how much?"), splice with the previous user
         message before classifying so the classifier has context.
      4. Classify → answer (personalized with name / prior turns).
      5. Store both turns and send reply.
    """
    engine = get_engine()
    store  = get_store()

    # (0) idempotency — Meta retries any non-2xx or timeout by re-delivering.
    # First-seen returns True and we proceed; already-seen returns False and we drop.
    if not store.mark_processed(message_id, sender):
        print(f"[webhook] duplicate msg_id={message_id} from={sender} — skipping")
        return

    session = store.get(sender)
    turn_count_before = store.turn_count(sender)

    lang_guess = engine.detect_lang(text)

    # (2) introduction → capture profile and short-circuit with an acknowledgment,
    # skipping the classifier entirely (intros don't map to a legal intent).
    if is_introduction(text):
        fields = extract_profile(text)
        store.update_profile(sender, fields)
        name = fields.get("name") or session["profile"].get("name")
        ack = (f"Nice to meet you, {name}. What employment question can I help with?"
               if lang_guess == "en" else
               f"Nimefurahi kukutana nawe, {name}. Una swali gani la sheria ya ajira?")
        store.remember_turn(sender, "user", text, intent="introduction", lang=lang_guess)
        store.remember_turn(sender, "bot",  ack,  intent="introduction_ack", lang=lang_guess)
        print(f"[webhook] from={sender} name={name} intent=introduction src=intro_ack")
        try:
            await whatsapp.send_text(sender, ack)
        except Exception as e:
            print(f"[webhook] failed to send reply: {e}")
        return

    # (3) short follow-ups get spliced with the previous user turn for context
    text_for_classifier = text
    last_user_text = session.get("last_user_text")
    if is_short_followup(text) and last_user_text and turn_count_before > 0:
        text_for_classifier = f"{last_user_text}. {text}"

    try:
        result = pipeline.classify_text(text_for_classifier)
        intent = result["intent"]
        confidence = result.get("confidence")
    except RuntimeError as e:
        print(f"[webhook] classifier unavailable: {e}")
        intent, confidence = "general_help", None

    # Don't re-greet a user we've already greeted in this session.
    session_ctx = {
        "profile":     session["profile"],
        "lang":        session.get("lang") or lang_guess,
        "turn_count":  turn_count_before,
        "last_intent": session.get("last_intent"),
    }
    if engine.is_greeting(text) and store.bot_has_greeted(sender):
        session_ctx["turn_count"] = turn_count_before  # ensures "welcome back" branch

    ans = engine.answer(text, intent, confidence, session=session_ctx)

    # Remember both sides of the exchange.
    store.remember_turn(sender, "user", text, intent=intent, lang=ans["lang"])
    store.remember_turn(sender, "bot",  ans["text"], intent=ans["intent"], lang=ans["lang"])

    profile_preview = session["profile"].get("name", "?")
    print(f"[webhook] from={sender} name={profile_preview} msg_id={message_id} "
          f"intent={intent} conf={confidence} lang={ans['lang']} src={ans['source']}")

    try:
        await whatsapp.send_text(sender, ans["text"])
    except Exception as e:
        print(f"[webhook] failed to send reply: {e}")


@app.get("/webhook", tags=["whatsapp"])
def whatsapp_verify(request: Request):
    """Meta subscription verification handshake.

    Meta calls this once when you subscribe your webhook URL, with these query params:
        hub.mode=subscribe
        hub.verify_token=<your token>
        hub.challenge=<random string to echo back>
    """
    params = request.query_params
    challenge = whatsapp.verify_challenge(
        mode      = params.get("hub.mode"),
        token     = params.get("hub.verify_token"),
        challenge = params.get("hub.challenge"),
    )
    if challenge is None:
        raise HTTPException(status_code=403, detail="verification failed")
    return PlainTextResponse(content=challenge)


@app.post("/webhook", tags=["whatsapp"])
async def whatsapp_receive(request: Request, background: BackgroundTasks):
    """Meta pushes every inbound WhatsApp message here.

    We MUST 200 within ~10s or Meta will retry (and eventually disable the webhook).
    So we schedule the actual work (classify → answer → send) as a background task
    and return immediately.
    """
    payload = await request.json()
    messages = whatsapp.parse_incoming(payload)
    for m in messages:
        if not m["text"] or not m["from"]:
            continue
        background.add_task(
            _process_whatsapp_message, m["from"], m["text"], m["message_id"],
        )
    return {"status": "received", "message_count": len(messages)}
