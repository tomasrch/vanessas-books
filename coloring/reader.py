"""Read a character reference sheet: find each drawing, its label, and whether it
is the character at all.

Her sheets look like a designer's model sheet: framed panels with titles
("Turnaround Views", "Facial Expressions", "Props & Environment Elements"), a
label under every drawing ("3/4 Back View", "Thoughtful (Hand on Chin)"), a
colour palette, a text summary, close-ups and props. The plain splitter sees one
shape (the frames join everything), so:

1. OCR (tesseract) finds the real text lines: titles and labels.
2. Frames, title bars and text are erased from the ink mask; what remains are
   the drawings, found as connected shapes.
3. Each drawing gets the label printed under it as its caption, and the panel
   title above it as its section.
4. Drawings under a props / palette / summary title, and drawings the local
   vision model says are not the character, are suggested for leaving out. She
   reviews every crop on the Characters tab before training.
5. With a black-and-white version of the same sheet, drawings are paired by
   label, so one LoRA learns both colour (covers) and line art (pages).
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from .sheet import isolate

log = logging.getLogger("coloring-reader")

MIN_AREA_FRAC = 0.002        # a drawing covers >= 0.2% of the sheet (drops icons, bullets)
# Sections that aren't the character (English after translation; Spanish as a fallback).
SKIP_SECTION = re.compile(r"prop|environment|palette|colou?r|swatch|trait|summary|personality|hex|"
                          r"associated|element|elementos|asociad|paleta|rasgos|resumen", re.I)
VISION_URL = "http://127.0.0.1:11434/api/generate"
VISION_MODEL = "minicpm-v:latest"   # reads small pictures better than llava


@dataclass
class Item:
    box: tuple[int, int, int, int]
    crop: Image.Image
    label: str = ""
    section: str = ""
    skip: bool = False
    reason: str = ""
    caption: str = ""
    line: Image.Image | None = field(default=None, repr=False)


# --- text ------------------------------------------------------------------------

def _ocr_lines(img: Image.Image) -> list[dict]:
    """Real text lines {box, text, h}. Drawings produce OCR noise ('Ge', 'LEA'), so a
    line counts only if it has a confident word of 3+ letters. Tesseract can join two
    labels that sit side by side into one line; a wide gap splits them again."""
    import pytesseract
    from pytesseract import Output

    # Pills and frames around labels make tesseract skip the text inside, so it reads
    # a copy with straight runs (pill tops/bottoms, frame lines) painted out.
    g = np.asarray(img.convert("L")).copy()
    ink = (g < 200).astype(np.uint8)
    runs = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, 45), np.uint8)) | \
        cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((30, 1), np.uint8))
    g[cv2.dilate(runs, np.ones((3, 3), np.uint8)) > 0] = 255
    s = 2                                   # sheet labels are ~12 px tall; OCR wants more
    big = Image.fromarray(g).resize((img.width * s, img.height * s), Image.LANCZOS)
    d = pytesseract.image_to_data(big, lang="eng+spa", output_type=Output.DICT, config="--psm 11")
    raw: dict[tuple, list] = {}
    for i, t in enumerate(d["text"]):
        t = t.strip()
        if not t or float(d["conf"][i]) < 60:
            continue
        x, y, w, h = (d[k][i] // s for k in ("left", "top", "width", "height"))
        raw.setdefault((d["block_num"][i], d["par_num"][i], d["line_num"][i]), []).append(
            (x, y, w, h, t, float(d["conf"][i])))
    groups = []
    for ws in raw.values():
        ws.sort(key=lambda w: w[0])
        cur = [ws[0]]
        for w in ws[1:]:
            prev = cur[-1]
            if w[0] - (prev[0] + prev[2]) > 1.5 * max(prev[3], w[3]):
                groups.append(cur)
                cur = [w]
            else:
                cur.append(w)
        groups.append(cur)
    out = []
    for ws in groups:
        # pill ends read as "(" ")" and make the line look taller than a label
        ws = [w for w in ws if re.search(r"[A-Za-z0-9\u00c0-\u017f]", w[4])]
        if not ws:
            continue
        if not any(len(re.sub(r"[^A-Za-z\u00c0-\u017f]", "", t)) >= 4 and c >= 80 for *_, t, c in ws):
            if not any(re.fullmatch(r"#[0-9A-Fa-f]{6}", t) for *_, t, _c in ws):
                continue
        x0 = min(w[0] for w in ws); y0 = min(w[1] for w in ws)
        x1 = max(w[0] + w[2] for w in ws); y1 = max(w[1] + w[3] for w in ws)
        out.append({"box": (x0, y0, x1, y1), "text": " ".join(w[4] for w in ws), "h": y1 - y0})
    return out


def _labels(text: list[dict], label_h: float) -> list[dict]:
    """Label = a short line that isn't a heading. A two-line label ("Happy" over
    "(Smile)") becomes one, anchored at its first line."""
    cands = sorted((t for t in text if t["h"] < label_h * 1.2 and t["box"][2] - t["box"][0] < 260),
                   key=lambda t: (t["box"][1], t["box"][0]))
    used, out = set(), []
    for i, a in enumerate(cands):
        if i in used:
            continue
        ax0, ay0, ax1, ay1 = a["box"]
        lab = {"box": list(a["box"]), "text": a["text"]}
        for j in range(i + 1, len(cands)):
            b = cands[j]
            bx0, by0, bx1, by1 = b["box"]
            if j not in used and 0 <= by0 - lab["box"][3] <= 0.8 * label_h and \
                    abs((bx0 + bx1) / 2 - (ax0 + ax1) / 2) < 0.6 * max(ax1 - ax0, bx1 - bx0):
                lab["box"] = [min(lab["box"][0], bx0), lab["box"][1], max(lab["box"][2], bx1), by1]
                lab["text"] += " " + b["text"]
                used.add(j)
        out.append(lab)
    return out


# --- drawings ----------------------------------------------------------------------

def _ink(img: Image.Image) -> np.ndarray:
    """Dark or clearly coloured pixels (tinted panel backgrounds don't count)."""
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    lo, hi = a.min(axis=2), a.max(axis=2)
    return ((lo < 215) | ((hi - lo) > 60) & (lo < 235)).astype(np.uint8)


def _frame_lines(img: Image.Image):
    """Long straight lines (panel frames): horizontal as (y, x0, x1), vertical as (x, y0, y1)."""
    ink = _ink(img)
    h, w = ink.shape
    hor, ver = [], []
    # Panel frames are long (300+ px on a 1536 sheet); a drawing's smile or a label pill isn't.
    hm = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(120, w // 6)), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(hm, 8)
    hor = [(y + bh // 2, x, x + bw) for x, y, bw, bh, _ in st[1:]]
    vm = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((max(80, h // 8), 1), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(vm, 8)
    ver = [(x + bw // 2, y, y + bh) for x, y, bw, bh, _ in st[1:]]
    return hor, ver


def _regions(img: Image.Image, text: list[dict], labels: list[dict]):
    """For each label, the box above it where its drawing is: up to the nearest text or
    panel frame above, sideways to halfway to the neighbouring labels of the same row
    and never past a panel's side frame."""
    hor, ver = _frame_lines(img)
    ink = _ink(img)
    for lab in labels:
        x0, y0, x1, y1 = lab["box"]
        cx = (x0 + x1) / 2
        row = sorted(((l["box"][0] + l["box"][2]) / 2 for l in labels if abs(l["box"][1] - y0) <= 12))
        i = row.index(cx)
        gaps = [cx - row[i - 1]] if i > 0 else []
        gaps += [row[i + 1] - cx] if i + 1 < len(row) else []
        half_l = (cx - row[i - 1]) / 2 if i > 0 else None
        half_r = (row[i + 1] - cx) / 2 if i + 1 < len(row) else None
        default = (min(gaps) / 2) if gaps else max(80.0, (x1 - x0) * 1.2)
        # a little past the midpoint: isolate() drops the neighbour's slivers
        rx0 = int(cx - 1.15 * (half_l or default))
        rx1 = int(cx + 1.15 * (half_r or default))
        above = [t["box"][3] for t in text
                 if t["box"][3] <= y0 - 4 and t["box"][0] < rx1 and t["box"][2] > rx0]
        # frame above: skip the label's own pill (its top edge is just above the text)
        above += [y for y, lx0, lx1 in hor if y <= y0 - 25 and lx0 <= cx <= lx1]
        ry0 = max(above) + 4 if above else 0
        # stop above the label's own pill: an ink row just over the text, as wide as the text
        ry1 = y0 - 2
        for y in range(y0 - 2, max(ry0, y0 - 25), -1):
            if ink[y, x0:x1].mean() > 0.6:
                ry1 = y - 3
                break
        sides = [x for x, ly0, ly1 in ver if ly0 <= y0 and ly1 >= ry0 + 20]
        left = [x for x in sides if x < x0 - 5]
        right = [x for x in sides if x > x1 + 5]
        if left:
            rx0 = max(rx0, max(left) + 4)
        if right:
            rx1 = min(rx1, min(right) - 4)
        yield lab, (max(0, rx0), max(0, ry0), min(img.width, rx1), max(0, ry1))


def _section_for(box, text, label_h: float, hor, ver) -> str:
    """The panel title: the nearest heading (text taller than a label) above the drawing,
    with no panel frame between the two - above it or to the side."""
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2
    # the heading can sit inside the region (between the panel frame and the drawings),
    # so measure against the region's bottom, i.e. the drawing's label
    heads = [t for t in text if t["h"] >= label_h * 1.2 and t["box"][3] <= y1
             and t["box"][0] - 60 <= cx
             and not any(t["box"][3] < y < y1 and lx0 <= cx <= lx1 for y, lx0, lx1 in hor)
             and not any(t["box"][2] < x < cx and ly0 <= t["box"][1] <= ly1 for x, ly0, ly1 in ver)]
    return max(heads, key=lambda t: t["box"][3])["text"] if heads else ""


# Her sheets tag every detail "(característica)" and every prop "(hábitat)",
# "(alimento)", "(naturaleza)" - the surest sign a drawing isn't a pose.
NOT_POSE = re.compile(r"caracter|h[aá]bitat|alimento|naturaleza|paisaje|characteristic|habitat|"
                      r"\bfood\b|\bnature\b|landscape|"
                      # body-part close-ups and props whose tag OCR lost
                      r"\b(ojos|eyes|plumas|feathers?|plumage|cola|tail|piel|skin|fur|pelaje|alas?|wings?|"
                      r"aletas?|fins?|p[eé]talos|petals|estambre|stamen|tronco|trunk|semillas|seeds|pico|beak|"
                      r"pecho|chest|caparaz\w*|shell|garras|claws|n[eé]ctar|nectar|forest|bosque|night|noche|"
                      r"nocturn\w*|nest\w*|nido|insect\w*|h[aá]bitos|habits|capullo|bud)\b", re.I)
EXPRESSION = re.compile(r"\b(happy|wink\w*|surprised?|content\w*|seren\w*|smil\w*|sad|angry|"
                        r"sleepy|laugh\w*|excited|joyful|cheerful|shy|scared|curious face)\b", re.I)


def caption_from_label(label: str, section: str) -> str:
    c = label.lower()
    c = re.sub(r"\b3\s*/\s*4\b", "three-quarter", c)
    c = re.sub(r"[()]", ", ", c).replace("&", "and").replace("/", ", ")
    c = re.sub(r"[^a-z0-9,' -]", "", c)
    c = re.sub(r"\s*,\s*(,\s*)*", ", ", c).strip(" ,-")
    sec = section.lower()
    if not c:
        return ""
    if c.endswith(" view") or c.endswith(" views"):
        return f"{c}, full body"
    if "express" in sec or EXPRESSION.search(c) and len(c.split()) <= 3:
        return f"head and shoulders, {c}"
    if "close" in sec or "detail" in sec:
        return f"close-up, {c}"
    return c


def read_sheet(img: Image.Image, keep_names: list[str] = ()) -> list[Item]:
    """Drawings found by their labels; [] when the sheet has no labels (use split_sheet).
    Labels and section titles are translated to English for the captions; the
    original label is kept for pairing and for showing her."""
    rgb = img.convert("RGB")
    text = _ocr_lines(rgb)
    if not text:
        return []
    label_h = float(np.median([t["h"] for t in text]))
    items = []
    hor, ver = _frame_lines(rgb)
    for lab, (rx0, ry0, rx1, ry1) in _regions(rgb, text, _labels(text, label_h)):
        if ry1 - ry0 < 40 or rx1 - rx0 < 30:
            continue
        crop = isolate(rgb.crop((rx0, ry0, rx1, ry1)))
        cw, ch = crop.size
        if (cw - 24) * (ch - 24) < MIN_AREA_FRAC * img.width * img.height:
            continue                       # empty space, an icon, a heart bullet
        it = Item(box=(rx0, ry0, rx1, ry1), crop=crop, label=lab["text"],
                  section=_section_for((rx0, ry0, rx1, ry1), text, label_h, hor, ver))
        if re.fullmatch(r"#[0-9a-f]{6}", it.label.strip().lower()):
            it.skip, it.reason = True, "colour swatch"
        elif SKIP_SECTION.search(it.section):
            it.skip, it.reason = True, f"in \u201c{it.section}\u201d"
        items.append(it)
    from .translate import labels_to_english
    srcs = [i.label for i in items] + [i.section for i in items]
    en = dict(zip(srcs, labels_to_english(srcs, keep=list(keep_names))))
    for it in items:
        lab_en, sec_en = en.get(it.label, it.label), en.get(it.section, it.section)
        it.caption = caption_from_label(lab_en, sec_en)
        if not it.skip and (NOT_POSE.search(it.label) or NOT_POSE.search(lab_en)):
            it.skip, it.reason = True, "detail or prop"
        elif not it.skip and SKIP_SECTION.search(sec_en):
            it.skip, it.reason = True, f"in \u201c{it.section}\u201d"
        # Detail panels: a face/head/hair close-up teaches the character; an eye,
        # a toe pad or a pair of sunglasses on its own doesn't.
        elif not it.skip and re.search(r"detail|close", sec_en, re.I) and \
                not re.search(r"face|head|hair|profile", lab_en, re.I):
            it.skip, it.reason = True, "detail close-up"
    band = max(1, img.height // 8)           # reading order
    items.sort(key=lambda i: (i.box[3] // band, i.box[0]))
    log.info("sheet: %d text lines, %d labelled drawings", len(text), len(items))
    return items


# --- is it the character? ------------------------------------------------------------

def is_character(crop: Image.Image, name: str, kind: str, timeout: float = 120) -> bool | None:
    """Local vision model's yes/no; None if it can't be asked."""
    buf = io.BytesIO()
    c = crop.convert("RGB")
    c.thumbnail((512, 512))
    c.save(buf, "PNG")
    body = {
        "model": VISION_MODEL, "stream": False, "keep_alive": "30s", "options": {"temperature": 0, "num_ctx": 4096},
        "prompt": (f"This picture is cut from a character sheet for {name}, a cartoon {kind}. "
                   f"Does it show {name} - the whole {kind}, the upper body, or the face/head? "
                   f"Answer 'no' for objects, accessories on their own, other animals or text. "
                   f"Answer with one word: yes or no."),
        "images": [base64.b64encode(buf.getvalue()).decode()],
    }
    req = urllib.request.Request(VISION_URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ans = json.load(r).get("response", "").strip().lower()
    except Exception as exc:
        log.warning("vision check unavailable: %s", exc)
        return None
    if ans.startswith("yes"):
        return True
    if ans.startswith("no"):
        return False
    return None


def classify(items: list[Item], name: str, kind: str) -> None:
    for it in items:
        if it.skip:
            continue
        verdict = is_character(it.crop, name, kind)
        if verdict is False:
            it.skip, it.reason = True, f"doesn't look like {name}"


# --- colour + black and white -----------------------------------------------------

def _key(label: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", label.lower()))


def pair(colour: list[Item], bw: list[Item]) -> int:
    """Give each colour item the black-and-white drawing with the same label.
    Unlabelled or unmatched drawings stay unpaired. Returns the pair count."""
    by_label: dict[str, list[Item]] = {}
    for b in bw:
        if b.label:
            by_label.setdefault(_key(b.label), []).append(b)
    n = 0
    for c in colour:
        cands = by_label.get(_key(c.label)) if c.label else None
        if cands:
            c.line = cands.pop(0).crop.convert("L")
            n += 1
    return n
