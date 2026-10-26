"""Cut a character sheet into one image per pose.

A sheet is several drawings of the same character on a white background. Ink is
found by threshold, each drawing's strokes are joined by dilating the mask, and
every connected blob big enough to be a drawing becomes a crop. Small blobs
(stray marks, labels) are dropped; overlapping boxes are merged.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

INK_LEVEL = 200          # grey value below which a pixel counts as ink
MIN_AREA_FRAC = 0.01     # a pose's box must cover >= 1% of the sheet
PAD_FRAC = 0.04          # padding around each crop, of the crop's longer side


def is_color(img: Image.Image, frac: float = 0.02) -> bool:
    """True when more than `frac` of the pixels are clearly coloured (not grey)."""
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    return float(((a.max(axis=2) - a.min(axis=2)) > 40).mean()) > frac


def isolate(img: Image.Image, pad: int = 12) -> Image.Image:
    """Keep only the biggest drawing in a crop (drops slivers of neighbouring poses and
    stray marks), on white, tightly cropped with `pad` px of margin."""
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    ink = ((a.min(axis=2) < 225) | ((a.max(axis=2) - a.min(axis=2)) > 40)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(ink, np.ones((7, 7), np.uint8)), 8)
    if n < 2:
        return img.convert("RGB")
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    # Eyes, a mouth or spots drawn inside the outline without touching it are separate
    # components. Keep everything inside the drawing's filled silhouette, and - because a
    # close-up cut off at the panel edge has an open outline that can't be filled - any
    # piece lying within the drawing's box that doesn't touch the crop edge (a sliver of
    # the neighbouring pose always does).
    main = (lab == k).astype(np.uint8)
    flood = np.pad(main, 1)
    cv2.floodFill(flood, None, (0, 0), 2)
    keep = (main > 0) | (flood[1:-1, 1:-1] == 0)
    mx, my, mw, mh = st[k, :4]
    H, W = lab.shape
    for j in range(1, n):
        x, y, w, h = st[j, :4]
        if j != k and x >= mx and y >= my and x + w <= mx + mw and y + h <= my + mh \
                and x > 0 and y > 0 and x + w < W and y + h < H:
            keep |= lab == j
    out = np.where(keep[..., None], a, 255).astype(np.uint8)
    x, y, w, h = st[k, :4]
    out = out[y:y + h, x:x + w]
    out = cv2.copyMakeBorder(out, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return Image.fromarray(out)


def line_art(img: Image.Image, side: int = 1024) -> Image.Image:
    """Colour cartoon -> black line art. Enlarged to `side` first so small sheet drawings
    don't come out blocky; then difference-of-Gaussians edges (the drawing's outlines and
    the borders between colour areas) plus anything near-black (lenses, outlines)."""
    s = side / max(img.size)
    if s > 1:
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    g = np.asarray(img.convert("L")).astype(np.float32)
    k = max(img.size) / 400
    dog = cv2.GaussianBlur(g, (0, 0), 0.8 * k) - 0.98 * cv2.GaussianBlur(g, (0, 0), 1.6 * k)
    out = np.where((dog < -1.5) | (g < 55), 0, 255).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats((out == 0).astype(np.uint8), 8)
    small = np.where(st[:, cv2.CC_STAT_AREA] < int(6 * k * k))[0]
    out[np.isin(lab, small[small > 0])] = 255
    return Image.fromarray(out)


def to_line_art(img: Image.Image, level: int = INK_LEVEL) -> Image.Image:
    """Pure black lines on white: drops grey shading and any colour."""
    g = np.asarray(img.convert("L"))
    return Image.fromarray(np.where(g < level, 0, 255).astype(np.uint8))


def _boxes(mask: np.ndarray) -> list[list[int]]:
    h, w = mask.shape
    k = max(3, int(min(h, w) * 0.015)) | 1
    joined = cv2.dilate(mask, np.ones((k, k), np.uint8), iterations=2)
    n, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        if bw * bh >= MIN_AREA_FRAC * w * h:
            boxes.append([x, y, x + bw, y + bh])
    # merge overlaps until stable
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    # reading order: rows (by top edge, in bands of 1/5 sheet height), then left to right
    band = max(1, h // 5)
    return sorted(boxes, key=lambda b: (b[1] // band, b[0]))


def split_sheet(img: Image.Image) -> list[Image.Image]:
    """One line-art crop per pose, in reading order."""
    art = to_line_art(img)
    mask = (np.asarray(art) == 0).astype(np.uint8) * 255
    crops = []
    for x0, y0, x1, y1 in _boxes(mask):
        pad = int(max(x1 - x0, y1 - y0) * PAD_FRAC)
        box = (max(0, x0 - pad), max(0, y0 - pad), min(art.width, x1 + pad), min(art.height, y1 + pad))
        crops.append(art.crop(box))
    return crops
