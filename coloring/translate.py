"""Spanish (or any language) -> English, with the local Llama model on prism.

FLUX's text encoder understands English far better than Spanish, and she writes her
sheets and scenes in Spanish. Labels become English training captions; scene
descriptions are translated before drawing (the original is kept with the page).
Everything runs on the LAN (Ollama on prism); nothing is sent out. If the model is
unavailable the text is used as it is.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.request

log = logging.getLogger("coloring-translate")
URL = "http://127.0.0.1:11434/api/generate"
MODEL = "llama3.2:latest"


def _ask(prompt: str, timeout: float = 120) -> str | None:
    body = {"model": MODEL, "prompt": prompt, "stream": False, "format": "json",
            "keep_alive": "60s",
            # Ollama here defaults to a 128K context (~10 GB of KV cache); a label needs 4K
            "options": {"temperature": 0, "num_ctx": 4096}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r).get("response")
    except Exception as exc:
        log.warning("translation unavailable: %s", exc)
        return None


def labels_to_english(labels: list[str], keep: list[str] = ()) -> list[str]:
    """Short sheet labels ('Vista de tres cuartos', 'Guiño') -> English, same order."""
    todo = [l for l in dict.fromkeys(labels) if l.strip()]
    if not todo:
        return list(labels)
    names = ", ".join(keep) or "none"
    out = _ask(
        "These are short labels from a children's cartoon character reference sheet. Translate each "
        "into short, natural English as used on such sheets (e.g. 'Vista frontal' -> 'Front View', "
        "'Guiño' -> 'Winking', 'Sentado' -> 'Sitting'). Keep labels that are already English unchanged. "
        f"Never translate these names: {names}. Reply as JSON: "
        '{"items": [{"src": "<label>", "en": "<English>"}]}. Labels:\n'
        + "\n".join(f"- {l}" for l in todo))
    try:
        items = json.loads(out or "")["items"]
        table = {i["src"].strip(): i["en"].strip() for i in items if i.get("src") and i.get("en")}
    except Exception:
        table = {}
    # the model sometimes drops or echoes labels in a long batch: retry those one by one
    for l in todo:
        if table.get(l.strip(), l).strip().lower() == l.strip().lower() and re.search(r"[a-záéíóúñ]{3}", l, re.I):
            one = _ask("Translate this short label from a children's character sheet into short English. "
                       f"Never translate these names: {names}. Reply as JSON "
                       '{"en": "<English>"}. Label: ' + l)
            try:
                table[l.strip()] = json.loads(one or "")["en"].strip() or l
            except Exception:
                pass
    return [table.get(l.strip(), l) for l in labels]


# Puerto Rican words a general model gets wrong ("amapola" is the hibiscus there, not a poppy).
PR_GLOSSARY = ("amapola = hibiscus flower, flamboyán = flamboyant flame tree, china = orange (the fruit), "
               "guineo = banana, chiringa = kite, guagua = bus, bohío = Taíno thatched hut, "
               "batey = Taíno village plaza, mahones = jeans, pana = friend, zafacón = trash can; "
               "keep place names such as El Morro, El Yunque and Viejo San Juan as they are")


def scene_to_english(scene: str, names: list[str]) -> str:
    """Her scene description -> English for FLUX; names stay exactly as written."""
    out = _ask(
        "Translate this description of a children's coloring-book scene into English. It is Puerto Rican "
        "Spanish: " + PR_GLOSSARY + ". If it is already "
        "English, return it unchanged. Keep these character names exactly as written: "
        f"{', '.join(names) or 'none'}. Do not add anything. Reply as JSON: "
        '{"en": "<English>"}.\n\n' + scene)
    try:
        en = json.loads(out or "")["en"].strip()
        return en or scene
    except Exception:
        return scene
