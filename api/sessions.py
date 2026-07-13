"""Per-WhatsApp-user conversation memory, backed by SQLite.

The store persists three things:

  sessions            — one row per wa_id: profile JSON, lang, last_intent,
                        last_user_text, created_at, updated_at.
  turns               — the rolling window of {user,bot} messages per wa_id,
                        capped at MAX_TURNS_PER_SESSION.
  processed_messages  — Meta message_ids we've already handled, to dedup the
                        retries Meta fires on any non-2xx response.

Path is configurable via env var SESSION_DB_PATH. Defaults to
artifacts/sessions.db. On Railway, mount a persistent volume (e.g. /data)
and set SESSION_DB_PATH=/data/sessions.db so state survives redeploys.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional

from config import ARTIFACTS

DB_PATH = Path(os.getenv("SESSION_DB_PATH", str(ARTIFACTS / "sessions.db")))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

MAX_TURNS_PER_SESSION       = 20
PROCESSED_RETENTION_SECONDS = 24 * 60 * 60  # Meta retries stop long before 24h

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    wa_id           TEXT PRIMARY KEY,
    profile_json    TEXT NOT NULL DEFAULT '{}',
    lang            TEXT,
    last_intent     TEXT,
    last_user_text  TEXT,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    wa_id   TEXT NOT NULL,
    role    TEXT NOT NULL,
    text    TEXT NOT NULL,
    intent  TEXT,
    lang    TEXT,
    ts      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_wa ON turns(wa_id, ts);
CREATE TABLE IF NOT EXISTS processed_messages (
    message_id    TEXT PRIMARY KEY,
    wa_id         TEXT NOT NULL,
    processed_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pm_ts ON processed_messages(processed_at);
"""


# ---- introduction / profile extraction ----------------------------------
_NAME_PATTERNS = [
    re.compile(r"\bmy name is\s+([A-Z][a-zA-Z'\-]{1,30})", re.I),
    re.compile(r"\bi'?m\s+([A-Z][a-zA-Z'\-]{1,30})", re.I),
    re.compile(r"\bi am\s+([A-Z][a-zA-Z'\-]{1,30})", re.I),
    re.compile(r"\bjina langu ni\s+([A-Z][a-zA-Z'\-]{1,30})", re.I),
    re.compile(r"\bnaitwa\s+([A-Z][a-zA-Z'\-]{1,30})", re.I),
    re.compile(r"\bmimi ni\s+([A-Z][a-zA-Z'\-]{1,30})", re.I),
]

_SECTOR_PATTERN = re.compile(
    r"\bin the\s+([a-z][a-z\s]{2,30}?)\s+sector\b|"
    r"\bin\s+([a-z][a-z\s]{2,30}?)\s+sector\b|"
    r"\bkatika sekta ya\s+([a-z][a-z\s]{2,30})",
    re.I,
)

_REGION_PATTERN = re.compile(
    r"\bbased in\s+([A-Z][a-zA-Z'\- ]{2,30})|"
    r"\bfrom\s+([A-Z][a-zA-Z'\- ]{2,30})|"
    r"\bnina ishi\s+([A-Z][a-zA-Z'\- ]{2,30})",
    re.I,
)


def extract_profile(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for pat in _NAME_PATTERNS:
        m = pat.search(text)
        if m:
            out["name"] = m.group(1).strip().title()
            break
    m = _SECTOR_PATTERN.search(text)
    if m:
        out["sector"] = next(g for g in m.groups() if g).strip().lower()
    m = _REGION_PATTERN.search(text)
    if m:
        out["region"] = next(g for g in m.groups() if g).strip().title()
    return out


def is_introduction(text: str) -> bool:
    return any(p.search(text) for p in _NAME_PATTERNS)


def is_short_followup(text: str) -> bool:
    """True if this looks like a context-dependent follow-up ("how much?", "when?")
    that needs the previous user turn spliced in before classification.
    Heuristic: 3 words or fewer.
    """
    stripped = text.strip()
    if not stripped:
        return False
    words = re.findall(r"[A-Za-zÀ-ſ]+", stripped)
    return len(words) <= 3


# ---- SQLite-backed session store ----------------------------------------
class SessionStore:
    def __init__(self, db_path: Path = DB_PATH):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            isolation_level=None,   # autocommit — every statement commits itself
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)

    # -- session lifecycle -------------------------------------------------
    def _ensure_session(self, wa_id: str) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO sessions(wa_id, created_at, updated_at) "
                "VALUES (?, ?, ?)",
                (wa_id, now, now),
            )

    def get(self, wa_id: str) -> Dict[str, Any]:
        self._ensure_session(wa_id)
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE wa_id = ?", (wa_id,)
            ).fetchone()
            turn_rows = self._conn.execute(
                "SELECT role, text, intent, lang, ts FROM turns WHERE wa_id = ? "
                "ORDER BY ts ASC",
                (wa_id,),
            ).fetchall()
        return {
            "profile":        json.loads(row["profile_json"] or "{}"),
            "lang":           row["lang"],
            "last_intent":    row["last_intent"],
            "last_user_text": row["last_user_text"],
            "created_at":     row["created_at"],
            "updated_at":     row["updated_at"],
            "turns":          [dict(t) for t in turn_rows],
        }

    def remember_turn(
        self,
        wa_id: str,
        role: str,
        text: str,
        intent: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> None:
        self._ensure_session(wa_id)
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO turns(wa_id, role, text, intent, lang, ts) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (wa_id, role, text, intent, lang, now),
            )
            updates = ["updated_at = ?"]
            params: list = [now]
            if lang:
                updates.append("lang = COALESCE(lang, ?)")
                params.append(lang)
            if role == "user":
                updates.append("last_user_text = ?")
                params.append(text)
                if intent:
                    updates.append("last_intent = ?")
                    params.append(intent)
            params.append(wa_id)
            self._conn.execute(
                f"UPDATE sessions SET {', '.join(updates)} WHERE wa_id = ?",
                params,
            )
            # trim old turns beyond the window
            self._conn.execute(
                "DELETE FROM turns WHERE wa_id = ? AND id NOT IN "
                "(SELECT id FROM turns WHERE wa_id = ? ORDER BY ts DESC LIMIT ?)",
                (wa_id, wa_id, MAX_TURNS_PER_SESSION),
            )

    def update_profile(self, wa_id: str, fields: Dict[str, str]) -> None:
        if not fields:
            return
        self._ensure_session(wa_id)
        with self._lock:
            row = self._conn.execute(
                "SELECT profile_json FROM sessions WHERE wa_id = ?", (wa_id,)
            ).fetchone()
            prof = json.loads(row["profile_json"] or "{}") if row else {}
            prof.update({k: v for k, v in fields.items() if v})
            self._conn.execute(
                "UPDATE sessions SET profile_json = ?, updated_at = ? WHERE wa_id = ?",
                (json.dumps(prof, ensure_ascii=False), time.time(), wa_id),
            )

    def turn_count(self, wa_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM turns WHERE wa_id = ?", (wa_id,)
            ).fetchone()
        return int(row["n"])

    def bot_has_greeted(self, wa_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM turns WHERE wa_id = ? AND role='bot' "
                "AND intent='greeting' LIMIT 1",
                (wa_id,),
            ).fetchone()
        return row is not None

    # -- Meta retry deduplication -----------------------------------------
    def mark_processed(self, message_id: str, wa_id: str) -> bool:
        """Atomic first-seen check on a Meta message_id.

        Returns True if this is the first time we've seen it (proceed to process),
        False if it's a Meta retry (skip). Also opportunistically GCs old rows.
        """
        if not message_id:
            return True  # no id → can't dedup, treat as fresh
        now = time.time()
        cutoff = now - PROCESSED_RETENTION_SECONDS
        with self._lock:
            self._conn.execute(
                "DELETE FROM processed_messages WHERE processed_at < ?", (cutoff,)
            )
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO processed_messages(message_id, wa_id, processed_at) "
                "VALUES (?, ?, ?)",
                (message_id, wa_id, now),
            )
        return cur.rowcount == 1


# module-level singleton
_store: Optional[SessionStore] = None


def get_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store
