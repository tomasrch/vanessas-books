#!/usr/bin/env python3
"""One-off repair (2026-10-26): re-cut kept crops with room around them, so drawings cut at the region edge come back whole.
   expand.py [slug ...] [--write] [--preview DIR]"""
import json, shutil, sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image
sys.path.insert(0, "/data/projects/coloring-studio")
from coloring.sheet import isolate
ROOT = Path("/mnt/Medina/VANESSA-BOOKS/characters"); PAD = 12; E = 110
args = [a for a in sys.argv[1:] if not a.startswith("--") and "/" not in a]
write = "--write" in sys.argv
pv = Path(sys.argv[sys.argv.index("--preview") + 1]) if "--preview" in sys.argv else None
if pv: pv.mkdir(exist_ok=True)
changed = {}


def erase_frames(reg):
    ink = (cv2.cvtColor(reg, cv2.COLOR_RGB2GRAY) < 128).astype(np.uint8)
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, 160), np.uint8)) | \
            cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((120, 1), np.uint8))
    reg[cv2.dilate(lines, np.ones((3, 3), np.uint8)) > 0] = 255
    return reg


def main_box(reg):
    """Box of the largest drawing in a region (frames erased), as isolate() would pick it."""
    a = erase_frames(reg.copy()).astype(np.int16)
    ink = ((a.min(axis=2) < 225) | ((a.max(axis=2) - a.min(axis=2)) > 40)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(ink, np.ones((7, 7), np.uint8)), 8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = st[k, :4]
    return x, y, x + w, y + h
for d in sorted(p for p in ROOT.iterdir() if not p.name.startswith("_")):
    if (d / "crops-pre-manual").exists() or d.name == "vanessa" or (args and d.name not in args):
        continue
    m = json.loads((d / "meta.json").read_text())
    sheet = np.asarray(Image.open(d / "sheet.png").convert("RGB"))
    gray = cv2.cvtColor(sheet, cv2.COLOR_RGB2GRAY)
    for i in range(m["crops"]):
        if i in m.get("excluded", []):
            continue
        p = d / "crops" / f"{i:02d}.png"
        old = np.asarray(Image.open(p).convert("RGB")); inner = old[PAD:-PAD, PAD:-PAD]
        r = cv2.matchTemplate(gray, cv2.cvtColor(inner, cv2.COLOR_RGB2GRAY), cv2.TM_SQDIFF_NORMED)
        v, _, (x, y), _ = cv2.minMaxLoc(r)
        if v > 0.2:
            continue
        h, w = inner.shape[:2]
        bounds = [x, y, x + w, y + h]                 # left, top, right, bottom (bottom never moves)
        moved = False
        for side in ("top", "left", "right"):
            k = {"left": 0, "top": 1, "right": 2}[side]
            for step in range(4, E + 1, 4):
                b = bounds.copy()
                b[k] = b[k] - step if side in ("left", "top") else b[k] + step
                b[0] = max(0, b[0]); b[1] = max(0, b[1]); b[2] = min(sheet.shape[1], b[2])
                bb = main_box(sheet[b[1]:b[3], b[0]:b[2]])
                if bb is None:
                    break
                bx0, by0, bx1, by1 = bb
                rw, rh = b[2] - b[0], b[3] - b[1]
                touching = {"left": bx0 <= 0, "top": by0 <= 0, "right": bx1 >= rw}[side]
                if not touching:
                    if step > 4:                      # closed after growing: keep the tight side
                        bounds[k] = b[0] + bx0 - 6 if side == "left" else (b[1] + by0 - 6 if side == "top" else b[0] + bx1 + 6)
                        moved = True
                    break
                if b[k] in (0, sheet.shape[1]) or step == E:
                    break                             # never closes (a background): leave it
        if not moved:
            continue
        x0, y0, x1, y1 = max(0, bounds[0]), max(0, bounds[1]), min(sheet.shape[1], bounds[2]), bounds[3]
        reg = erase_frames(sheet[y0:y1, x0:x1].copy())
        new = isolate(Image.fromarray(reg), pad=PAD)
        na = np.asarray(new)
        nh, nw = na.shape[0] - 2 * PAD, na.shape[1] - 2 * PAD
        if nh * nw <= h * w * 1.01 or nh > h * 1.5 or nw > w * 1.5:
            continue
        rr = cv2.matchTemplate(cv2.cvtColor(na, cv2.COLOR_RGB2GRAY), cv2.cvtColor(inner, cv2.COLOR_RGB2GRAY), cv2.TM_SQDIFF_NORMED)
        vv, _, (ox, oy), _ = cv2.minMaxLoc(rr)
        sub = na[oy:oy + h, ox:ox + w].min(axis=2) < 160; oi = inner.min(axis=2) < 160
        if (sub & oi).sum() < 0.97 * oi.sum():
            continue
        grow = nh * nw / (h * w) - 1
        print(f"{d.name} {i:02d} {m['captions'][i][:34]!r} +{grow:.0%} area")
        changed.setdefault(d.name, []).append(i)
        if pv:
            a, b = Image.fromarray(old), new
            pair = Image.new("RGB", (a.width + b.width + 10, max(a.height, b.height)), "white")
            pair.paste(a, (0, 0)); pair.paste(b, (a.width + 10, 0)); pair.save(pv / f"{d.name}-{i:02d}.png")
        if write:
            bak = d / "crops-pre-expand"; bak.mkdir(exist_ok=True)
            if not (bak / p.name).exists(): shutil.copy2(p, bak / p.name)
            new.save(p)
print(json.dumps(changed))
