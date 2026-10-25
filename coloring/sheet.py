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
