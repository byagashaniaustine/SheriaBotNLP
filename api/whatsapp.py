"""Meta WhatsApp Cloud API integration.

Reads three env vars:
    WHATSAPP_VERIFY_TOKEN     — arbitrary string; must match what you configure
                                in the Meta App dashboard when subscribing the webhook.
    WHATSAPP_ACCESS_TOKEN     — permanent (system-user) access token from Meta.
    WHATSAPP_PHONE_NUMBER_ID  — the WABA phone-number ID (numeric string).

Docs: https://developers.facebook.com/docs/whatsapp/cloud-api
"""
from __future__ import annotations

import os
from typing import Optional, Dict, Any, List

import httpx

GRAPH_API_VERSION = os.getenv("WHATSAPP_GRAPH_VERSION", "v20.0")

VERIFY_TOKEN     = os.getenv("WHATSAPP_VERIFY_TOKEN", "")
ACCESS_TOKEN     = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
PHONE_NUMBER_ID  = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")

# WhatsApp text-message body has a 4096-char limit.
MAX_BODY_LEN = 4000


def verify_challenge(mode: Optional[str], token: Optional[str], challenge: Optional[str]) -> Optional[str]:
    """Return the challenge string if Meta's verification handshake is valid, else None."""
    if mode == "subscribe" and token and token == VERIFY_TOKEN:
        return challenge
    return None


def parse_incoming(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract [{from, text, message_id}] from Meta's webhook payload.

    Ignores non-text messages and status updates (delivered/read receipts) —
    those still arrive at the same webhook but have no `messages` array.
    """
    out: List[Dict[str, str]] = []
    for entry in payload.get("entry", []) or []:
        for change in entry.get("changes", []) or []:
            value = change.get("value", {}) or {}
            for msg in value.get("messages", []) or []:
                if msg.get("type") != "text":
                    continue
                text = (msg.get("text") or {}).get("body", "")
                out.append({
                    "from":       msg.get("from", ""),
                    "text":       text,
                    "message_id": msg.get("id", ""),
                })
    return out


async def send_text(to: str, body: str) -> Dict[str, Any]:
    """Send a text message via the Graph API. Returns Meta's JSON response."""
    if not (ACCESS_TOKEN and PHONE_NUMBER_ID):
        raise RuntimeError(
            "WhatsApp credentials missing — set WHATSAPP_ACCESS_TOKEN and "
            "WHATSAPP_PHONE_NUMBER_ID env vars."
        )
    if len(body) > MAX_BODY_LEN:
        body = body[: MAX_BODY_LEN - 1] + "…"
    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type":  "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type":    "individual",
        "to":                to,
        "type":              "text",
        "text":              {"preview_url": False, "body": body},
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(url, headers=headers, json=payload)
        try:
            data = resp.json()
        except Exception:
            data = {"raw": resp.text}
        if resp.status_code >= 400:
            print(f"[whatsapp] send failed {resp.status_code}: {data}")
        return data
