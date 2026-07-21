
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

import bert_classifier
import whatsapp
from answer_engine import get_engine
from config import API_DESC, API_TITLE, API_VERSION
from schemas import ClassifyRequest, ClassifyResponse, HealthResponse
from sessions import (
    extract_profile, get_store, is_introduction, is_short_followup,
)


# --------------------------------------------------------------------------
# Startup / shutdown lifecycle
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        bert_classifier.load()
        print(f"[startup] BERT loaded from {bert_classifier.source()}")
    except Exception as e:
        print(f"[startup] WARNING: BERT model failed to load ({e!r}). "
              "/classify + /webhook will error until the model is available.")
    get_engine()
    print("[startup] answer engine ready (answer_bank.json)")
    get_store()
    print("[startup] session store ready (SQLite)")
    yield
    print("[shutdown] bye")


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESC,
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Meta
# --------------------------------------------------------------------------
@app.get("/", tags=["meta"])
def root() -> Dict[str, Any]:
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "docs":    "/docs",
        "endpoints": [
            "GET  /health",
            "POST /classify",
            "GET  /webhook   (Meta verification handshake)",
            "POST /webhook   (WhatsApp inbound messages)",
        ],
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    engine = None
    try:
        engine = get_engine()
    except Exception:
        pass
    return HealthResponse(
        status="ok",
        model_loaded=bert_classifier.is_loaded(),
        model_source=bert_classifier.source(),
        num_labels=bert_classifier.num_labels(),
        answer_bank_loaded=engine is not None,
    )


# --------------------------------------------------------------------------
# Classify (also used as a debug endpoint — same code path as the webhook)
# --------------------------------------------------------------------------
@app.post("/classify", response_model=ClassifyResponse, tags=["inference"])
def classify(req: ClassifyRequest) -> ClassifyResponse:
    try:
        pred = bert_classifier.classify(req.text)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"BERT unavailable: {e}")

    engine = get_engine()
    ans = engine.answer(req.text, pred["intent"], pred["confidence"],
                        session={}, top_3=pred["top_3"])
    return ClassifyResponse(
        input      = pred["input"],
        intent     = pred["intent"],
        confidence = pred["confidence"],
        top_3      = pred["top_3"],
        answer     = ans["text"],
        lang       = ans["lang"],
        source     = ans["source"],
    )


# --------------------------------------------------------------------------
# WhatsApp webhook — the primary interface
# --------------------------------------------------------------------------
async def _process_whatsapp_message(sender: str, text: str, message_id: str) -> None:
    
    engine = get_engine()
    store  = get_store()

    if not store.mark_processed(message_id, sender):
        print(f"[webhook] duplicate msg_id={message_id} from={sender} — skipping")
        return

    session = store.get(sender)
    turn_count_before = store.turn_count(sender)
    lang_guess = engine.detect_lang(text)

    # (2) introduction → grab profile fields, ack, skip the classifier
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

    # (3) short follow-up → splice previous user turn
    text_for_classifier = text
    last_user_text = session.get("last_user_text")
    if is_short_followup(text) and last_user_text and turn_count_before > 0:
        text_for_classifier = f"{last_user_text}. {text}"

    # (4) classify
    top_3 = None
    try:
        result = bert_classifier.classify(text_for_classifier)
        intent = result["intent"]
        confidence = result.get("confidence")
        top_3 = result.get("top_3")
    except Exception as e:
        print(f"[webhook] classifier unavailable: {e}")
        intent, confidence = "general_help", None

    session_ctx = {
        "profile":     session["profile"],
        "lang":        session.get("lang") or lang_guess,
        "turn_count":  turn_count_before,
        "last_intent": session.get("last_intent"),
    }
    if engine.is_greeting(text) and store.bot_has_greeted(sender):
        session_ctx["turn_count"] = turn_count_before  # forces "welcome back" branch

    ans = engine.answer(text, intent, confidence, session=session_ctx, top_3=top_3)

    store.remember_turn(sender, "user", text, intent=intent, lang=ans["lang"])
    store.remember_turn(sender, "bot",  ans["text"], intent=ans["intent"], lang=ans["lang"])

    # Log a truncated snippet of the user message so OOS false-positives are
    # debuggable without dumping full PII into log storage.
    preview = text.replace("\n", " ")[:60] + ("..." if len(text) > 60 else "")
    print(f"[webhook] from={sender} name={session['profile'].get('name','?')} "
          f"msg_id={message_id} intent={intent} conf={confidence} "
          f"lang={ans['lang']} src={ans['source']} text={preview!r}")

    try:
        await whatsapp.send_text(sender, ans["text"])
    except Exception as e:
        print(f"[webhook] failed to send reply: {e}")


@app.get("/webhook", tags=["whatsapp"])
def whatsapp_verify(request: Request):
    """Meta subscription verification handshake."""
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

    We MUST 200 within ~10s or Meta retries. Actual work happens in a background
    task so the webhook returns immediately.
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
