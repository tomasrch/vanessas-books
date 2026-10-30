"""Characters and pages on the NAS, one folder each with a meta.json.

characters/<slug>/meta.json  sheet.png  crops/NN.png
pages/<page_id>/meta.json    raw.png  page.png  page.pdf
loras/<trigger>.safetensors

meta.json is the source of truth; Redis holds only the training queue, live
progress and login sessions. Writes go through a temp file + rename so a reader
never sees half a file.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import time
import unicodedata
from pathlib import Path

from . import config

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,23}$")
PAGE_ID_RE = re.compile(r"^p\d{14}-[0-9a-f]{6}$")


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


# --- characters --------------------------------------------------------------

def slugify(name: str) -> str:
    # "Coquí puertorriqueño" -> "coqui-puertorriqueno" (accents folded, not dropped)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:24].strip("-")
    return s or "character"


def trigger_for(slug: str) -> str:
    # Rare token the base model has no meaning for; "_" because imagegen LoRA names allow it.
    return "cbk_" + slug.replace("-", "_")


def char_dir(slug: str) -> Path:
    if not SLUG_RE.match(slug):
        raise ValueError("bad character id")
    return config.CHARACTERS / slug


def new_character(name: str, kind: str) -> dict:
    base = slugify(name)
    slug, n = base, 2
    while (config.CHARACTERS / slug).exists():
        slug, n = f"{base[:21]}-{n}", n + 1
    d = config.CHARACTERS / slug
    (d / "crops").mkdir(parents=True)
    meta = {
        "slug": slug, "name": name.strip(), "kind": kind.strip().lower(),
        "trigger": trigger_for(slug), "status": "new", "created": time.time(),
        "crops": 0, "excluded": [], "error": None, "trained_at": None,
    }
    _write_json(d / "meta.json", meta)
    return meta


def get_character(slug: str) -> dict | None:
    return _read_json(char_dir(slug) / "meta.json")


def save_character(meta: dict) -> None:
    _write_json(char_dir(meta["slug"]) / "meta.json", meta)


def list_characters() -> list[dict]:
    out = []
    if config.CHARACTERS.is_dir():
        for d in config.CHARACTERS.iterdir():
            if d.name.startswith("_"):
                continue
            m = _read_json(d / "meta.json")
            if m:
                out.append(m)
    # her own characters (no roster code, e.g. Vanessa) first, then the roster in CB order
    return sorted(out, key=lambda m: (bool(m.get("code")), m.get("code") or "", m["created"]))


def archive_character(slug: str) -> None:
    """Moved aside, never deleted: her artwork is the input and can't be regenerated."""
    src = char_dir(slug)
    dst = config.CHARACTERS / "_archived" / f"{slug}--{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
    dst.parent.mkdir(exist_ok=True)
    shutil.move(str(src), str(dst))
    lora = config.LORAS / f"{trigger_for(slug)}.safetensors"
    if lora.exists():
        shutil.move(str(lora), str(dst / lora.name))


def lora_file(meta: dict) -> Path:
    return config.LORAS / f"{meta['trigger']}.safetensors"


# --- pages -------------------------------------------------------------------

def new_page(meta: dict) -> dict:
    page_id = time.strftime("p%Y%m%d%H%M%S", time.gmtime()) + "-" + secrets.token_hex(3)
    d = config.PAGES / page_id
    d.mkdir(parents=True)
    meta = {**meta, "id": page_id, "status": "queued", "created": time.time(), "error": None}
    _write_json(d / "meta.json", meta)
    return meta


def page_dir(page_id: str) -> Path:
    if not PAGE_ID_RE.match(page_id):
        raise ValueError("bad page id")
    return config.PAGES / page_id


def get_page(page_id: str) -> dict | None:
    return _read_json(page_dir(page_id) / "meta.json")


def save_page(meta: dict) -> None:
    _write_json(page_dir(meta["id"]) / "meta.json", meta)


def list_pages(limit: int = 200) -> list[dict]:
    if not config.PAGES.is_dir():
        return []
    ids = sorted((d.name for d in config.PAGES.iterdir() if PAGE_ID_RE.match(d.name)), reverse=True)
    return [m for m in (get_page(i) for i in ids[:limit]) if m]


# --- casts: characters trained together so they stay distinct in one scene ---------
# casts/<id>.json {id, members: [slugs], status: new|queued|training|ready|failed,
#                  trained_at, error}; LoRA loras/cbcast_<id>.safetensors

def cast_id(slugs) -> str:
    import hashlib
    return hashlib.sha1("+".join(sorted(set(slugs))).encode()).hexdigest()[:10]


def cast_trigger(cid: str) -> str:
    return f"cbcast_{cid}"


def _cast_path(cid: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{10}", cid):
        raise ValueError("bad cast id")
    return config.ROOT / "casts" / f"{cid}.json"


def get_cast(cid: str) -> dict | None:
    return _read_json(_cast_path(cid))


def save_cast(meta: dict) -> None:
    p = _cast_path(meta["id"])
    p.parent.mkdir(exist_ok=True)
    _write_json(p, meta)


def cast_for(slugs) -> dict:
    """The cast record for this group of characters, created on first use."""
    cid = cast_id(slugs)
    m = get_cast(cid)
    if not m:
        m = {"id": cid, "members": sorted(set(slugs)), "status": "new", "created": time.time(),
             "trained_at": None, "error": None}
        save_cast(m)
    return m


def list_casts() -> list[dict]:
    return [m for p in sorted((config.ROOT / "casts").glob("*.json")) if (m := _read_json(p))]


def cast_lora(cid: str) -> Path:
    return config.LORAS / f"{cast_trigger(cid)}.safetensors"


def cast_is_current(m: dict) -> bool:
    """Ready, its LoRA on disk, and trained after every member's own last training."""
    if m.get("status") != "ready" or not cast_lora(m["id"]).is_file():
        return False
    for slug in m["members"]:
        c = get_character(slug)
        if not c or (c.get("trained_at") or 0) > (m.get("trained_at") or 0):
            return False
    return True
