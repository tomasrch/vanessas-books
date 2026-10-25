"""coloring-studio web app: upload character sheets, train them, make coloring pages.

Runs on 127.0.0.1:8095 behind nginx :8090, which admits 10.3.0.0/16 only (minus the
pfSense and portal relays). On top of that, one shared password (scrypt hash in
/etc/coloring-studio.env) and a session cookie kept in Redis.

    python -m coloring.app --hash-password     # prints a COLORING_PASSWORD_HASH line
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import logging
import random
import re
import secrets
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from . import config, page, store
from .sheet import split_sheet
from .trainer import PROGRESS, QUEUE, redis_client

log = logging.getLogger("coloring-studio")
STATIC = Path(__file__).parent / "static"
SESSION_COOKIE = "cs_session"
Image.MAX_IMAGE_PIXELS = 40_000_000    # decompression-bomb cap, same as the PRISM API

app = FastAPI(title="Vanessa's Coloring Books", docs_url=None, redoc_url=None, openapi_url=None)
r = redis_client()
renderer = ThreadPoolExecutor(max_workers=2, thread_name_prefix="render")
# Logo and favicon only - shown on the login screen, so served without a session.
app.mount("/img", StaticFiles(directory=STATIC / "img"), name="img")


# --- auth --------------------------------------------------------------------

def hash_password(pw: str) -> str:
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(pw.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt${salt.hex()}${h.hex()}"


def check_password(pw: str) -> bool:
    try:
        _, salt, want = config.PASSWORD_HASH.split("$")
    except ValueError:
        return False          # no hash configured = nobody gets in
    got = hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt), n=2**14, r=8, p=1)
    return hmac.compare_digest(got.hex(), want)


def require_login(request: Request) -> None:
    token = request.cookies.get(SESSION_COOKIE, "")
    if not (token and re.fullmatch(r"[A-Za-z0-9_-]{43}", token) and r.exists(f"coloring:session:{token}")):
        raise HTTPException(401, "please log in")


class Login(BaseModel):
    password: str = Field(..., max_length=200)


@app.post("/api/login")
def login(body: Login, request: Request, response: Response):
    ip = request.headers.get("x-real-ip") or (request.client.host if request.client else "?")
    key = f"coloring:login-fail:{ip}"
    if int(r.get(key) or 0) >= 10:
        raise HTTPException(429, "too many attempts, try again in 15 minutes")
    if not check_password(body.password):
        r.incr(key)
        r.expire(key, 900)
        raise HTTPException(401, "wrong password")
    token = secrets.token_urlsafe(32)
    r.setex(f"coloring:session:{token}", config.SESSION_DAYS * 86400, "1")
    response.set_cookie(SESSION_COOKIE, token, max_age=config.SESSION_DAYS * 86400,
                        httponly=True, samesite="strict")
    return {"ok": True}


@app.post("/api/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get(SESSION_COOKIE, "")
    if token:
        r.delete(f"coloring:session:{token}")
    response.delete_cookie(SESSION_COOKIE)
    return {"ok": True}


@app.get("/api/me", dependencies=[Depends(require_login)])
def me():
    return {"ok": True}


# --- upload checks -------------------------------------------------------------

def av_scan(data: bytes) -> str | None:
    """Virus signature name, or None. Fail-open if clamd is down (same policy as PRISM)."""
    try:
        import clamd
        res = clamd.ClamdUnixSocket("/run/clamav/clamd.ctl").instream(io.BytesIO(data))
        status, sig = res.get("stream", ("OK", None))
        return sig if status == "FOUND" else None
    except Exception as exc:
        log.warning("AV scan unavailable (%s) - upload allowed", exc)
        return None


def read_image(data: bytes) -> Image.Image:
    if len(data) > config.UPLOAD_MAX_BYTES:
        raise HTTPException(400, "file is larger than 25 MB")
    sig = av_scan(data)
    if sig:
        log.error("infected upload rejected: %s", sig)
        raise HTTPException(400, f"this file was rejected by the virus scanner ({sig})")
    try:
        Image.open(io.BytesIO(data)).verify()
        img = Image.open(io.BytesIO(data))
        if img.format not in ("PNG", "JPEG", "WEBP"):
            raise ValueError(img.format)
        img.load()
    except Exception:
        raise HTTPException(400, "please upload a PNG, JPG or WEBP image")
    if img.mode in ("RGBA", "LA", "P"):    # transparent background -> white
        bg = Image.new("RGB", img.size, "white")
        bg.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[-1])
        img = bg
    return img.convert("RGB")


# --- characters ----------------------------------------------------------------

def _with_progress(m: dict) -> dict:
    p = r.hgetall(PROGRESS + m["slug"]) if m["status"] in ("queued", "training") else {}
    return {**m, "progress": p or None,
            "queue_position": _queue_position(m["slug"]) if m["status"] == "queued" else None}


def _queue_position(slug: str) -> int | None:
    q = r.lrange(QUEUE, 0, -1)
    return q.index(slug) + 1 if slug in q else None


def _char(slug: str) -> dict:
    try:
        m = store.get_character(slug)
    except ValueError:
        m = None
    if not m:
        raise HTTPException(404, "no such character")
    return m


@app.get("/api/characters", dependencies=[Depends(require_login)])
def characters():
    return [_with_progress(m) for m in store.list_characters()]


@app.post("/api/characters", dependencies=[Depends(require_login)])
async def add_character(name: str = Form(..., min_length=1, max_length=40),
                        kind: str = Form(..., min_length=1, max_length=40),
                        sheet: UploadFile = File(...)):
    if not re.fullmatch(r"[A-Za-z][A-Za-z ]{0,39}", kind.strip()):
        raise HTTPException(400, "describe what the character is in plain words, e.g. 'hedgehog' or 'girl'")
    img = read_image(await sheet.read())
    crops = split_sheet(img)
    if not crops:
        raise HTTPException(400, "no drawing found on that sheet - is it dark lines on a white background?")
    m = store.new_character(name, kind)
    d = store.char_dir(m["slug"])
    img.save(d / "sheet.png")
    for i, c in enumerate(crops):
        c.save(d / "crops" / f"{i:02d}.png")
    m["crops"] = len(crops)
    store.save_character(m)
    return m


class Poses(BaseModel):
    excluded: list[int] = Field(default_factory=list, max_length=100)


@app.put("/api/characters/{slug}/poses", dependencies=[Depends(require_login)])
def set_poses(slug: str, body: Poses):
    m = _char(slug)
    if m["status"] in ("queued", "training"):
        raise HTTPException(409, "wait until training finishes")
    m["excluded"] = sorted({i for i in body.excluded if 0 <= i < m["crops"]})
    if len(m["excluded"]) >= m["crops"]:
        raise HTTPException(400, "keep at least one pose")
    store.save_character(m)
    return m


@app.post("/api/characters/{slug}/train", dependencies=[Depends(require_login)])
def start_training(slug: str):
    m = _char(slug)
    if m["status"] in ("queued", "training"):
        return _with_progress(m)
    m.update(status="queued", error=None)
    store.save_character(m)
    r.delete(PROGRESS + slug)
    r.rpush(QUEUE, slug)
    return _with_progress(m)


@app.delete("/api/characters/{slug}", dependencies=[Depends(require_login)])
def remove_character(slug: str):
    m = _char(slug)
    if m["status"] in ("queued", "training"):
        raise HTTPException(409, "wait until training finishes")
    store.archive_character(slug)
    return {"ok": True}


@app.get("/api/characters/{slug}/sheet", dependencies=[Depends(require_login)])
def sheet_image(slug: str):
    _char(slug)
    return FileResponse(store.char_dir(slug) / "sheet.png", media_type="image/png")


@app.get("/api/characters/{slug}/crops/{n}", dependencies=[Depends(require_login)])
def crop_image(slug: str, n: int):
    _char(slug)
    p = store.char_dir(slug) / "crops" / f"{n:02d}.png"
    if not p.is_file():
        raise HTTPException(404, "no such pose")
    return FileResponse(p, media_type="image/png")


# --- pages ---------------------------------------------------------------------

class PageRequest(BaseModel):
    characters: list[str] = Field(..., min_length=1, max_length=config.MAX_CHARACTERS_PER_SCENE)
    scene: str = Field(..., min_length=3, max_length=600)
    orientation: str = Field("portrait", pattern="^(portrait|landscape)$")
    variations: int = Field(2, ge=1, le=config.MAX_VARIATIONS)


def build_prompt(scene: str, chars: list[dict]) -> str:
    """Her names -> trigger + kind ("Pip" -> "cbk_pip hedgehog"). A chosen character the
    description doesn't mention is put up front so it still appears."""
    text, missing = scene.strip(), []
    for c in chars:
        ref = f"{c['trigger']} {c['kind']}"
        pat = re.compile(r"\b" + re.escape(c["name"]) + r"\b", re.IGNORECASE)
        if pat.search(text):
            text = pat.sub(ref, text)
        else:
            missing.append(ref)
    if missing:
        text = f"{' and '.join(missing)}: {text}"
    return f"{text}, {config.STYLE}"


def _imagegen(body: dict) -> bytes:
    def call(url, data=None):
        req = urllib.request.Request(url, data, {"Content-Type": "application/json"} if data else {})
        return urllib.request.urlopen(req, timeout=config.IMAGEGEN_TIMEOUT)

    with call(f"{config.IMAGEGEN_URL}/imagegen", json.dumps(body).encode()) as resp:
        if resp.status == 200 and resp.headers.get("Content-Type", "").startswith("image/"):
            return resp.read()
        job_id = json.load(resp)["job_id"]
    deadline = time.time() + config.IMAGEGEN_TIMEOUT
    while time.time() < deadline:         # queued behind other work: collect when done
        time.sleep(10)
        with call(f"{config.IMAGEGEN_URL}/imagegen/{job_id}?format=json") as resp:
            status = json.load(resp).get("status")
        if status == "completed":
            with call(f"{config.IMAGEGEN_URL}/imagegen/{job_id}") as resp:
                return resp.read()
        if status == "failed":
            raise RuntimeError("the image service could not draw this page")
    raise RuntimeError("timed out waiting for the image service")


def render_page(page_id: str) -> None:
    m = store.get_page(page_id)
    try:
        m["status"] = "drawing"
        store.save_page(m)
        w, h = config.RENDER_PORTRAIT
        if m["orientation"] == "landscape":
            w, h = h, w
        body = {"prompt": m["prompt"], "width": w, "height": h, "seed": m["seed"], "wait": True,
                "loras": [{"name": f"{config.LORA_NS}/{t}", "scale": m["lora_scale"]} for t in m["triggers"]]}
        png = _imagegen(body)
        d = store.page_dir(page_id)
        (d / "raw.png").write_bytes(png)
        pg = page.to_page(Image.open(io.BytesIO(png)), landscape=m["orientation"] == "landscape")
        page.save(pg, d / "page.png", d / "page.pdf")
        (d / "thumb.png").write_bytes(page.thumbnail(pg))
        m.update(status="done", error=None, finished=time.time())
    except Exception as exc:
        log.exception("page %s failed", page_id)
        detail = exc.read().decode(errors="replace")[:300] if isinstance(exc, urllib.error.HTTPError) else str(exc)
        m.update(status="failed", error=detail[:300])
    store.save_page(m)


@app.post("/api/pages", dependencies=[Depends(require_login)])
def make_pages(body: PageRequest):
    chars = [_char(s) for s in dict.fromkeys(body.characters)]
    not_ready = [c["name"] for c in chars if c["status"] != "ready" or not store.lora_file(c).is_file()]
    if not_ready:
        raise HTTPException(400, f"not trained yet: {', '.join(not_ready)}")
    prompt = build_prompt(body.scene, chars)
    out = []
    for _ in range(body.variations):
        m = store.new_page({
            "scene": body.scene, "prompt": prompt, "orientation": body.orientation,
            "characters": [c["slug"] for c in chars], "names": [c["name"] for c in chars],
            "triggers": [c["trigger"] for c in chars], "lora_scale": 1.0,
            "seed": random.randint(0, 2**32 - 1),
        })
        renderer.submit(render_page, m["id"])
        out.append(m)
    return out


@app.get("/api/pages", dependencies=[Depends(require_login)])
def pages():
    return store.list_pages()


def _page_file(page_id: str, name: str, media: str, download: str | None = None):
    try:
        p = store.page_dir(page_id) / name
    except ValueError:
        raise HTTPException(404, "no such page")
    if not p.is_file():
        raise HTTPException(404, "not ready")
    return FileResponse(p, media_type=media, filename=download)


@app.get("/api/pages/{page_id}/thumb", dependencies=[Depends(require_login)])
def page_thumb(page_id: str):
    return _page_file(page_id, "thumb.png", "image/png")


@app.get("/api/pages/{page_id}/pdf", dependencies=[Depends(require_login)])
def page_pdf(page_id: str):
    return _page_file(page_id, "page.pdf", "application/pdf", f"coloring-{page_id}.pdf")


@app.get("/api/pages/{page_id}/png", dependencies=[Depends(require_login)])
def page_png(page_id: str):
    return _page_file(page_id, "page.png", "image/png", f"coloring-{page_id}.png")


@app.delete("/api/pages/{page_id}", dependencies=[Depends(require_login)])
def remove_page(page_id: str):
    try:
        src = store.page_dir(page_id)
    except ValueError:
        raise HTTPException(404, "no such page")
    dst = config.PAGES / "_archived" / page_id
    dst.parent.mkdir(exist_ok=True)
    src.rename(dst)
    return {"ok": True}


# --- page shell ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((STATIC / "index.html").read_text(),
                        headers={"Cache-Control": "no-store", "X-Frame-Options": "DENY"})


@app.on_event("startup")
def resume_unfinished() -> None:
    """Pages cut off by a restart are drawn again."""
    def go():
        for m in store.list_pages():
            if m["status"] in ("queued", "drawing"):
                renderer.submit(render_page, m["id"])
    threading.Thread(target=go, daemon=True).start()


if __name__ == "__main__":
    if "--hash-password" in sys.argv:
        import getpass
        pw = getpass.getpass("password: ")
        if pw != getpass.getpass("again: ") or len(pw) < 8:
            sys.exit("passwords differ or shorter than 8 characters")
        print(f"COLORING_PASSWORD_HASH={hash_password(pw)}")
