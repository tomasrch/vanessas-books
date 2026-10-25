"""Turn one character's pose crops into a LoRA training set.

Each crop is placed on white canvases of a few shapes, at several sizes and
positions, so the LoRA learns the character rather than one framing. Captions
name the trigger and class, and only what varies between images: pose, view,
framing, rendering style. Everything that should come from the trigger (the
character's look) is left out of the caption - see the cast LoRA lesson in the
PRISM training README.

A colour crop is used twice: as is, captioned STYLE_COLOR, and traced to line
art, captioned STYLE. The trigger then carries who the character is and the
style words carry how it is drawn, so one LoRA gives colour covers and black and
white coloring pages. A black-line crop is used as line art only.
"""
from __future__ import annotations

import base64
import io
import json
import random
import urllib.request
from pathlib import Path

from PIL import Image

from .sheet import is_color, line_art

STYLE = "black and white line art, children's coloring book page, white background"
STYLE_COLOR = "full color children's book illustration, white background"
CANVASES = [(1024, 1024), (768, 1024), (1024, 768)]
VARIANTS_PER_CROP = 6
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
CAPTION_MODEL = "llava:latest"


def describe_pose(crop: Image.Image, timeout: float = 120) -> str:
    """A few words on pose and view, from the local vision model. '' on failure."""
    buf = io.BytesIO()
    crop.convert("RGB").save(buf, "PNG")
    body = {
        "model": CAPTION_MODEL,
        "prompt": ("Describe this cartoon character's pose and which way it faces in 3 to 8 words, "
                   "for example 'front view, waving' or 'side view, running to the left'. "
                   "Answer with only those words."),
        "images": [base64.b64encode(buf.getvalue()).decode()],
        "stream": False,
        # Unload right away: a resident llava (~9 GB) leaves no room to train on the card.
        "keep_alive": 0,
        "options": {"temperature": 0},
    }
    req = urllib.request.Request(OLLAMA_URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = json.load(r).get("response", "")
    except Exception:
        return ""
    text = text.strip().strip(".").strip("'\"").lower()
    return text if 0 < len(text) <= 80 else ""


def _place(crop: Image.Image, size: tuple[int, int], rng: random.Random) -> tuple[Image.Image, str]:
    w, h = size
    frac = rng.choice([0.9, 0.75, 0.6, 0.45])
    s = min(w * frac / crop.width, h * frac / crop.height)
    img = crop.resize((max(1, int(crop.width * s)), max(1, int(crop.height * s))), Image.LANCZOS)
    x = rng.randint(0, w - img.width)
    y = rng.randint(0, h - img.height)
    canvas = Image.new(img.mode if img.mode in ("L", "RGB") else "RGB", size, "white")
    canvas.paste(img, (x, y))
    framing = "full body, large in frame" if frac >= 0.75 else "full body, small in frame"
    return canvas, framing


def build(crops: list[Image.Image], trigger: str, class_word: str, out_dir: Path,
          poses: list[str] | None = None, seed: int = 0,
          lines: list[Image.Image | None] | None = None) -> int:
    """Write <n>.png + <n>.txt pairs to out_dir. Returns the image count.
    `lines[i]`, when given, is a hand-drawn line version of colour crop i, used
    instead of tracing it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    n = 0
    for i, crop in enumerate(crops):
        pose = (poses[i] if poses and i < len(poses) else "") or ""
        if is_color(crop):
            drawn = lines[i] if lines and i < len(lines) else None
            versions = [("c", crop.convert("RGB"), STYLE_COLOR),
                        ("l", drawn.convert("L") if drawn else line_art(crop), STYLE)]
            per = VARIANTS_PER_CROP // 2
        else:
            versions = [("l", crop.convert("L"), STYLE)]
            per = VARIANTS_PER_CROP
        for tag, img, style in versions:
            for v in range(per):
                canvas, framing = _place(img, CANVASES[v % len(CANVASES)], rng)
                caption = ", ".join(p for p in [f"{trigger} {class_word}", pose, framing, style] if p)
                stem = f"{i:02d}{tag}_{v}"
                canvas.save(out_dir / f"{stem}.png")
                (out_dir / f"{stem}.txt").write_text(caption + "\n")
                n += 1
    return n
