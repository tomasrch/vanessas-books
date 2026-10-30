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
import unicodedata
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
from .reader import read_sheet
from .sheet import split_sheet
from .translate import scene_to_english
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


def current_hash() -> str:
    try:
        return config.PASSWORD_FILE.read_text().strip()
    except OSError:
        return config.PASSWORD_HASH


def check_password(pw: str) -> bool:
    try:
        _, salt, want = current_hash().split("$")
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
    key = _check_attempts(request)
    if not check_password(body.password):
        _failed(key)
        raise HTTPException(401, "wrong password")
    token = secrets.token_urlsafe(32)
    r.setex(f"coloring:session:{token}", config.SESSION_DAYS * 86400, "1")
    # Secure over https://vanessasbooks.ivns.lab; the plain-HTTP :8090 fallback can't use it.
    response.set_cookie(SESSION_COOKIE, token, max_age=config.SESSION_DAYS * 86400,
                        httponly=True, samesite="strict",
                        secure=request.headers.get("x-forwarded-proto") == "https")
    return {"ok": True}


def _client_ip(request: Request) -> str:
    return request.headers.get("x-real-ip") or (request.client.host if request.client else "?")


def _check_attempts(request: Request) -> str:
    key = f"coloring:login-fail:{_client_ip(request)}"
    if int(r.get(key) or 0) >= 10:
        raise HTTPException(429, "too many attempts, try again in 15 minutes")
    return key


def _failed(key: str) -> None:
    r.incr(key)
    r.expire(key, 900)


class PasswordChange(BaseModel):
    current: str = Field(..., max_length=200)
    new: str = Field(..., min_length=8, max_length=200)


@app.post("/api/password", dependencies=[Depends(require_login)])
def change_password(body: PasswordChange, request: Request):
    key = _check_attempts(request)
    if not check_password(body.current):
        _failed(key)
        raise HTTPException(401, "the current password is not right")
    config.PASSWORD_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = config.PASSWORD_FILE.with_suffix(".tmp")
    tmp.write_text(hash_password(body.new) + "\n")
    tmp.chmod(0o600)
    tmp.replace(config.PASSWORD_FILE)
    # Everyone else is logged out; this browser stays in.
    mine = f"coloring:session:{request.cookies.get(SESSION_COOKIE, '')}"
    for k in r.scan_iter("coloring:session:*"):
        if k != mine:
            r.delete(k)
    log.info("password changed from %s", _client_ip(request))
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
    try:
        partners = sorted(int(f.name[:2]) for f in (store.char_dir(m["slug"]) / "crops").glob("[0-9][0-9]-line.png"))
    except (OSError, ValueError):
        partners = []
    return {**m, "progress": p or None, "line_partners": partners,
            "queue_position": _queue_position(m["slug"]) if m["status"] == "queued" else None}


def _queue_position(slug: str) -> int | None:
    q = r.lrange(QUEUE, 0, -1)
    return q.index(slug) + 1 if slug in q else None


RATE = "coloring:train:rate:"     # first (time, step) the app saw for a run, for the ETA


def _eta(key: str, p: dict) -> int | None:
    """Seconds left, from the step rate since the app first saw this run. The trainer
    doesn't record a start time, and restarting it would kill a running job."""
    try:
        step, total = int(p.get("step", 0)), int(p.get("total", 0))
    except ValueError:
        return None
    if p.get("stage") != "training" or not total:
        return None
    seen = r.hgetall(RATE + key)
    if not seen or int(seen.get("total", 0)) != total or int(seen.get("step", 0)) > step:
        r.hset(RATE + key, mapping={"t": time.time(), "step": step, "total": total})
        r.expire(RATE + key, 2 * 86400)
        return None
    done, spent = step - int(seen["step"]), time.time() - float(seen["t"])
    return round((total - step) * spent / done) if done >= 20 else None


def _training_on_gpus() -> dict[int, str]:
    """GPU index -> trigger of the training running on it (read from run.py's config path)."""
    import pynvml
    out = {}
    pynvml.nvmlInit()
    for i in config.TRAIN_GPUS:
        for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(pynvml.nvmlDeviceGetHandleByIndex(i)):
            try:
                cmd = Path(f"/proc/{proc.pid}/cmdline").read_bytes().split(b"\0")
            except OSError:
                continue
            if b"run.py" in cmd and cmd.index(b"run.py") + 1 < len(cmd):
                out[i] = Path(cmd[cmd.index(b"run.py") + 1].decode()).stem
    return out


@app.get("/api/training", dependencies=[Depends(require_login)])
def training():
    """What is training or waiting to, each group's state, and room on the training GPUs."""
    import pynvml
    queue = r.lrange(QUEUE, 0, -1)
    chars = store.list_characters()
    names = {m["slug"]: m["name"] for m in chars}
    jobs = [{"key": m["slug"], "group": False, "names": [m["name"]], "trigger": m["trigger"], "status": m["status"]}
            for m in chars if m["status"] in ("queued", "training")]
    groups = []
    for cm in store.list_casts():
        members = [names.get(s, s) for s in cm["members"]]
        groups.append({"members": cm["members"], "names": members,
                       "status": "current" if store.cast_is_current(cm) else cm["status"]})
        if cm["status"] in ("queued", "training"):
            jobs.append({"key": f"cast:{cm['id']}", "group": True, "names": members,
                         "trigger": store.cast_trigger(cm["id"]), "status": cm["status"]})
    try:
        on_gpu = _training_on_gpus()
    except pynvml.NVMLError:
        on_gpu = {}
    for j in jobs:
        p = r.hgetall(PROGRESS + j["key"])
        j.update(stage=p.get("stage"), step=int(p.get("step") or 0), total=int(p.get("total") or 0),
                 eta=_eta(j["key"], p), position=queue.index(j["key"]) + 1 if j["key"] in queue else None,
                 gpu=next((g for g, trig in on_gpu.items() if trig == j["trigger"]), None))
    gpus = []
    try:
        pynvml.nvmlInit()
        for i in config.TRAIN_GPUS:
            mem = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i))
            gpus.append({"index": i, "free_gb": round(mem.free / 2**30, 1),
                         "training": next((j["names"] for j in jobs if j["gpu"] == i), None)})
    except pynvml.NVMLError:
        pass
    jobs.sort(key=lambda j: (j["gpu"] is None, j["status"] != "training", j["position"] or 0))
    return {"jobs": jobs, "groups": groups, "gpus": gpus, "need_gb": config.TRAIN_MIN_FREE_GB}


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
                        sheet: UploadFile = File(...),
                        sheet_bw: UploadFile | None = File(None)):
    if not re.fullmatch(r"[^\W\d_][^\W\d_ ]*( [^\W\d_]+){0,4}", kind.strip()):
        raise HTTPException(400, "describe what the character is in plain words, e.g. 'rana' or 'tortuga'")
    img = read_image(await sheet.read())
    bw = read_image(await sheet_bw.read()) if sheet_bw is not None and sheet_bw.filename else None
    m = store.new_character(name, kind)
    d = store.char_dir(m["slug"])
    img.save(d / "sheet.png")
    if bw is not None:
        bw.save(d / "sheet-bw.png")
    m["status"] = "reading"
    store.save_character(m)
    threading.Thread(target=read_into_character, args=(m["slug"],), daemon=True).start()
    return m


def read_into_character(slug: str) -> None:
    """Cut a freshly uploaded sheet into poses (in the background: OCR + translation
    take ~10-30 s). Her model sheets are read by their labels; a plain sheet of poses
    falls back to the simple splitter."""
    from .reader import pair
    from .translate import labels_to_english
    m = store.get_character(slug)
    d = store.char_dir(slug)
    try:
        img = Image.open(d / "sheet.png").convert("RGB")
        kind_en = labels_to_english([m["kind"]])[0].strip().lower() or m["kind"]
        items = read_sheet(img, keep_names=[m["name"]])
        if len(items) >= 2:
            if (d / "sheet-bw.png").is_file():
                pair(items, read_sheet(Image.open(d / "sheet-bw.png").convert("RGB"), keep_names=[m["name"]]))
            for i, it in enumerate(items):
                it.crop.save(d / "crops" / f"{i:02d}.png")
                if it.line is not None:
                    it.line.save(d / "crops" / f"{i:02d}-line.png")
            m.update(crops=len(items), captions=[i.caption for i in items], labels=[i.label for i in items],
                     excluded=[i for i, it in enumerate(items) if it.skip],
                     skip_reasons={str(i): it.reason for i, it in enumerate(items) if it.skip})
        else:
            crops = split_sheet(img)
            if not crops:
                raise ValueError("no drawings found - is it dark lines on a white background?")
            for i, c in enumerate(crops):
                c.save(d / "crops" / f"{i:02d}.png")
            m.update(crops=len(crops))
        m.update(kind=kind_en, kind_original=m["kind"], status="new", error=None)
    except Exception as exc:
        log.exception("reading sheet for %s failed", slug)
        m.update(status="failed", error=f"couldn't read the sheet: {exc}"[:300])
    store.save_character(m)


class Poses(BaseModel):
    excluded: list[int] = Field(default_factory=list, max_length=100)


@app.put("/api/characters/{slug}/poses", dependencies=[Depends(require_login)])
def set_poses(slug: str, body: Poses):
    m = _char(slug)
    if m["status"] in ("queued", "training", "reading"):
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
    if m["status"] == "reading":
        raise HTTPException(409, "still reading the sheet")
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
    kind: str = Field("page", pattern="^(page|cover)$")    # B&W coloring page or colour cover
    variations: int = Field(2, ge=1, le=config.MAX_VARIATIONS)


def _fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def _names(c: dict) -> list[str]:
    """Every way she may call a character: full name, aliases ("Coquí"), and its English
    kind ("manatee" - the translator turns "Manatí" into that), longest first."""
    return sorted({c["name"], *c.get("aliases", []), c["kind"]}, key=len, reverse=True)


def build_prompt(scene: str, chars: list[dict], style: str = config.STYLE) -> str:
    """Her names -> trigger + kind ("el Coquí" -> "cbk_coqui_puertorriqueno coqui frog"),
    matched without regard to accents or case. A chosen character the description
    doesn't mention is put up front so it still appears."""
    text, missing = scene.strip(), []
    for c in chars:
        ref = f"{c['trigger']} {c['kind']}"
        hit = None
        for n in _names(c):
            # match on the accent-folded text, character by character, so the span maps back
            folded = "".join(_fold(ch) or ch for ch in text)
            m = re.search(r"(?<![a-z0-9])" + re.escape(_fold(n)) + r"(?![a-z0-9])", folded)
            if m and len(folded) == len(text):
                hit = m
                break
        if hit:
            text = text[:hit.start()] + ref + text[hit.end():]
        else:
            missing.append(ref)
    if missing:
        text = f"{' and '.join(missing)}: {text}"
    return f"{text}, {style}"


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
                "loras": [{"name": n, "scale": m["lora_scale"]}
                          for n in m.get("loras") or [f"{config.LORA_NS}/{t}" for t in m["triggers"]]]}
        png = _imagegen(body)
        d = store.page_dir(page_id)
        (d / "raw.png").write_bytes(png)
        render = Image.open(io.BytesIO(png))
        landscape = m["orientation"] == "landscape"
        pg = page.to_cover(render, landscape) if m.get("kind") == "cover" else page.to_page(render, landscape)
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
    names = [n for c in chars for n in _names(c)]
    scene_en = scene_to_english(body.scene, names)       # she writes in Spanish
    prompt = build_prompt(scene_en, chars, config.COVER_STYLE if body.kind == "cover" else config.STYLE)
    loras, cast = [f"{config.LORA_NS}/{c['trigger']}" for c in chars], None
    if len(chars) > 1:
        # Two separately trained characters blend about half the time; a joint "cast"
        # LoRA keeps them apart. Use it when current, otherwise queue it for next time.
        cm = store.cast_for([c["slug"] for c in chars])
        cast = {"id": cm["id"], "used": False}
        if store.cast_is_current(cm):
            loras, cast["used"] = [f"{config.LORA_NS}/{store.cast_trigger(cm['id'])}"], True
        elif cm["status"] not in ("queued", "training"):
            cm.update(status="queued", error=None)
            store.save_cast(cm)
            r.rpush(QUEUE, f"cast:{cm['id']}")
    out = []
    for _ in range(body.variations):
        m = store.new_page({
            "scene": body.scene, "scene_en": scene_en, "prompt": prompt, "orientation": body.orientation,
            "kind": body.kind, "characters": [c["slug"] for c in chars], "names": [c["name"] for c in chars],
            "triggers": [c["trigger"] for c in chars], "loras": loras, "cast": cast, "lora_scale": 1.0,
            "seed": random.randint(0, 2**32 - 1),
        })
        renderer.submit(render_page, m["id"])
        out.append(m)
    return out


@app.post("/api/pages/{page_id}/redo", dependencies=[Depends(require_login)])
def redo_page(page_id: str):
    """Same request, new drawing (new seed); a group gets its cast if that's ready now."""
    try:
        m = store.get_page(page_id)
    except ValueError:
        m = None
    if not m:
        raise HTTPException(404, "no such page")
    return make_pages(PageRequest(characters=m["characters"], scene=m["scene"], orientation=m["orientation"],
                                  kind=m.get("kind", "page"), variations=1))


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
