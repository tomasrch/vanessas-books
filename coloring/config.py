"""Coloring studio settings. Everything is env-overridable (/etc/coloring-studio.env)."""
import os
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


# --- storage (NAS dataset Medina/VANESSA-BOOKS, mounted on prism + prod-prism01/02) ---
ROOT = Path(_env("COLORING_ROOT", "/mnt/Medina/VANESSA-BOOKS"))
CHARACTERS = ROOT / "characters"
PAGES = ROOT / "pages"
LORAS = ROOT / "loras"                     # = IMAGEGEN_LORA_EXTRA_DIRS "vb" on every image node
TRAIN = ROOT / "train"
LORA_NS = _env("COLORING_LORA_NS", "vb")    # imagegen addresses our LoRAs as "vb/<trigger>"

# --- image generation (the PRISM imagegen API; FLUX loads on demand, unloads when idle) ---
IMAGEGEN_URL = _env("COLORING_IMAGEGEN_URL", "http://127.0.0.1:8001")
IMAGEGEN_TIMEOUT = int(_env("COLORING_IMAGEGEN_TIMEOUT", "1800"))   # cold loads + queueing are fine

# --- broker: the redis-whisper instance (:6381, LAN + password), own "coloring:*" keyspace ---
REDIS_HOST = _env("COLORING_REDIS_HOST") or _env("WHISPER_REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(_env("COLORING_REDIS_PORT") or _env("WHISPER_REDIS_PORT", "6381"))
REDIS_PASSWORD = _env("COLORING_REDIS_PASSWORD") or _env("WHISPER_REDIS_PASSWORD") or None

# --- login (one shared password; network gate is nginx) ---
PASSWORD_HASH = _env("COLORING_PASSWORD_HASH")     # scrypt$<salt hex>$<hash hex>, see app.hash_password
# Set from the page ("Change password"); when present it replaces PASSWORD_HASH. The unit's
# StateDirectory creates the folder, owned by the service user, mode 700.
PASSWORD_FILE = Path(_env("COLORING_PASSWORD_FILE", "/var/lib/coloring-studio/password"))
SESSION_DAYS = int(_env("COLORING_SESSION_DAYS", "30"))

# --- training (ai-toolkit on prism; same setup as the PRISM cast LoRA) ---
TRAIN_PYTHON = _env("COLORING_TRAIN_PYTHON", "/data/imagegen/train/venv/bin/python")
TRAIN_TOOLKIT = _env("COLORING_TRAIN_TOOLKIT", "/data/imagegen/train/ai-toolkit")
TRAIN_BASE = _env("COLORING_TRAIN_BASE", "/data/imagegen/train/models/schnell-assist")
TRAIN_HF_HOME = _env("COLORING_TRAIN_HF_HOME", "/data/imagegen/train/hf")
TRAIN_STEPS = int(_env("COLORING_TRAIN_STEPS", "1500"))
TRAIN_RANK = int(_env("COLORING_TRAIN_RANK", "16"))
TRAIN_GPUS = [int(x) for x in _env("COLORING_TRAIN_GPUS", "0,1").split(",") if x.strip()]
# A run needs ~12 GB; the card's resident live-call whisper worker holds ~2.5 GB.
TRAIN_MIN_FREE_GB = float(_env("COLORING_TRAIN_MIN_FREE_GB", "12.5"))

# --- pages: US Letter at 300 dpi ---
DPI = 300
PAGE_PX = (2550, 3300)                     # 8.5 x 11 in, portrait
MARGIN_PX = int(float(_env("COLORING_MARGIN_IN", "0.5")) * DPI)
RENDER_PORTRAIT = (1056, 1408)             # 3:4 like the 7.5 x 10 in printable area, multiples of 16
STYLE = ("children's coloring book page, black and white line art, clean bold black outlines, "
         "pure white background, no shading, no grey, no color, no text, no letters, "
         "simple shapes, large open areas to color")
# Covers: the colour half of each character's training ("full color children's book illustration").
COVER_STYLE = ("full color children's book cover illustration, bright cheerful tropical colors, "
               "clean cartoon style, leave the top quarter as simple open sky for the title, "
               "no text, no letters, no words")
MAX_CHARACTERS_PER_SCENE = 4
MAX_VARIATIONS = 4

UPLOAD_MAX_BYTES = 25 * 1024 * 1024
