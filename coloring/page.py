"""Render -> printable coloring page: pure black and white, US Letter, 300 dpi.

FLUX draws line art but leaves grey anti-aliasing, the odd grey fill and a bit
of colour (pink cheeks, red ribbons). The page is upscaled to fill the printable
area first and thresholded after, so lines stay smooth at print size; specks
smaller than SPECK_PX are removed.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from . import config

INK_LEVEL = 170
SPECK_PX = 40          # ink blobs smaller than this (at 300 dpi) are noise


def to_page(render: Image.Image, landscape: bool = False) -> Image.Image:
    pw, ph = config.PAGE_PX
    if landscape:
        pw, ph = ph, pw
    m = config.MARGIN_PX
    area = (pw - 2 * m, ph - 2 * m)
    g = render.convert("L")
    s = min(area[0] / g.width, area[1] / g.height)
    g = g.resize((round(g.width * s), round(g.height * s)), Image.LANCZOS)
    ink = (np.asarray(g) < INK_LEVEL).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    small = np.where(stats[:, cv2.CC_STAT_AREA] < SPECK_PX)[0]
    ink[np.isin(labels, small[small > 0])] = 0
    art = Image.fromarray(np.where(ink == 1, 0, 255).astype(np.uint8))
    page = Image.new("L", (pw, ph), 255)
    page.paste(art, ((pw - art.width) // 2, (ph - art.height) // 2))
    return page.convert("1", dither=Image.NONE)


def to_cover(render: Image.Image, landscape: bool = False) -> Image.Image:
    """Colour render -> full US Letter page, edge to edge (no margin: covers print full bleed).
    The render's 3:4 is cropped to 8.5:11 from the centre."""
    pw, ph = config.PAGE_PX
    if landscape:
        pw, ph = ph, pw
    img = render.convert("RGB")
    s = max(pw / img.width, ph / img.height)
    img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    x, y = (img.width - pw) // 2, (img.height - ph) // 2
    return img.crop((x, y, x + pw, y + ph))


def save(page: Image.Image, png_path, pdf_path) -> None:
    page.save(png_path, "PNG", dpi=(config.DPI, config.DPI))
    page.save(pdf_path, "PDF", resolution=config.DPI)


def thumbnail(page: Image.Image, width: int = 480) -> bytes:
    t = page.convert("RGB" if page.mode == "RGB" else "L")
    t.thumbnail((width, width * 2))
    buf = io.BytesIO()
    t.save(buf, "PNG", optimize=True)
    return buf.getvalue()
