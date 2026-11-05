"""Her feedback on finished pages, and what it changes.

A page's feedback lives in its meta.json:
    "feedback": {"rating": "good" | "bad", "problems": [tag, ...], "who": [slug, ...],
                 "note": "...", "at": <unix time>}

What it changes, per character or per exact group of characters (a group = the
characters on the page):
- "good" pages become extra training examples the next time that character or
  group trains (a LoRA learns from good examples only, never from "this is wrong").
- problems on "bad" pages made since the last training become prompt fixes for
  new pages of the same character or group ("with both eyes and a mouth clearly
  drawn"). Retraining starts the count again.
"""
from __future__ import annotations

from PIL import Image, ImageOps

from . import config, store
from .dataset import STYLE as TRAIN_STYLE, STYLE_COLOR as TRAIN_STYLE_COLOR

# tag -> applies to one character ("char") or to the group / page as a whole ("page")
TAGS = {
    "eyes_face": "char", "extra_parts": "char", "missing_parts": "char", "wrong_look": "char",
    "blended": "page", "char_missing": "page", "not_bw": "page", "text": "page",
    "scene_wrong": "page", "other": "page",
}
# Tags with no prompt fix: only better training examples help those.
NO_FIX = {"wrong_look", "scene_wrong", "other"}
MAX_TRAINING_PAGES = int(config._env("COLORING_FEEDBACK_MAX_PAGES", "20"))


def group_of(page: dict) -> tuple[str, ...]:
    return tuple(sorted(set(page.get("characters") or [])))


def group_key(group) -> str:
    return "+".join(group)


def trained_at(group: tuple[str, ...]) -> float:
    """When this character, or this group's joint LoRA, last finished training (0 = never)."""
    if len(group) == 1:
        c = store.get_character(group[0])
        return (c or {}).get("trained_at") or 0
    cm = store.get_cast(store.cast_id(group))
    return (cm or {}).get("trained_at") or 0


def _fb(page: dict) -> dict | None:
    fb = page.get("feedback")
    return fb if page.get("status") == "done" and fb and fb.get("rating") in ("good", "bad") else None


def summary(pages: list[dict]) -> dict:
    """Per character and group: approved pages (all / since the last training), problems
    reported on pages drawn since the last training, and her notes."""
    out: dict[str, dict] = {}
    since: dict[tuple, float] = {}
    for p in pages:
        fb = _fb(p)
        if not fb:
            continue
        g = group_of(p)
        if not g:
            continue
        if g not in since:
            since[g] = trained_at(g)
        s = out.setdefault(group_key(g), {"members": list(g), "approved": 0, "approved_new": 0,
                                          "problem_pages": 0, "problems": {}, "notes": []})
        if fb["rating"] == "good":
            s["approved"] += 1
            s["approved_new"] += (fb.get("at") or 0) > since[g]
        elif (p.get("created") or 0) > since[g]:        # drawn by the current training
            s["problem_pages"] += 1
            for tag in fb.get("problems") or []:
                s["problems"][tag] = s["problems"].get(tag, 0) + 1
        else:
            continue                 # a problem the last training was meant to fix: history only
        if fb.get("note"):
            s["notes"].append({"page": p["id"], "rating": fb["rating"], "note": fb["note"], "at": fb.get("at")})
    for s in out.values():
        s["notes"].sort(key=lambda n: n["at"] or 0, reverse=True)
        s["notes"] = s["notes"][:10]
    return out


def _ref(c: dict) -> str:
    return f"{c['trigger']} {c['kind']}"


def fixes_for(chars: list[dict], pages: list[dict]) -> tuple[list[str], list[str]]:
    """(tags, prompt phrases) from problems she reported on pages of exactly these
    characters, drawn since their last training."""
    g = tuple(sorted(c["slug"] for c in chars))
    since = trained_at(g)
    who_by_tag: dict[str, set[str]] = {}
    for p in pages:
        fb = _fb(p)
        if not fb or fb["rating"] != "bad" or group_of(p) != g or (p.get("created") or 0) <= since:
            continue
        for tag in fb.get("problems") or []:
            if tag in TAGS and tag not in NO_FIX:
                who = set(fb.get("who") or []) & set(g)
                who_by_tag.setdefault(tag, set()).update(who or g)
    by_slug = {c["slug"]: c for c in chars}
    phrases: list[str] = []
    for tag in TAGS:                       # stable order
        if tag not in who_by_tag:
            continue
        who = [by_slug[s] for s in g if s in who_by_tag[tag]]
        if tag == "eyes_face":
            phrases += [f"{_ref(c)} with a complete face, both eyes and its mouth clearly drawn" for c in who]
        elif tag == "extra_parts":
            phrases += [f"{_ref(c)} with correct anatomy, no extra legs, arms or heads" for c in who]
        elif tag == "missing_parts":
            phrases += [f"{_ref(c)} drawn whole, every leg and arm visible" for c in who]
        elif tag == "blended" and len(g) > 1:
            phrases.append("every character clearly separate and distinct, never merged together")
        elif tag == "char_missing" and len(g) > 1:
            phrases.append(f"all {len(g)} characters fully visible: " + " and ".join(_ref(c) for c in chars))
        elif tag == "not_bw":
            phrases.append("only clean black outlines on white, no grey fill")
        elif tag == "text":
            phrases.append("no writing, letters or numbers anywhere")
    return sorted(who_by_tag), phrases


def scene_prompt(page: dict) -> str:
    """The page's prompt without the style words (and without any fixes)."""
    if page.get("scene_prompt"):
        return page["scene_prompt"]
    prompt = page.get("prompt") or ""
    for style in (config.STYLE, config.COVER_STYLE):
        if prompt.endswith(", " + style):
            return prompt[: -len(style) - 2]
    return prompt


def _page_image(page: dict) -> Image.Image | None:
    d = store.page_dir(page["id"])
    cover = page.get("kind") == "cover"
    # Covers: the render itself (the saved cover has a frosted title band on top).
    path = d / ("raw.png" if cover else "page.png")
    if not path.is_file():
        return None
    img = Image.open(path)
    if cover:
        img = img.convert("RGB")
    else:
        img = img.convert("L")
        box = ImageOps.invert(img).getbbox()       # trim the print margin around the drawing
        if box:
            pad = int(0.04 * max(img.size))
            img = img.crop((max(0, box[0] - pad), max(0, box[1] - pad),
                            min(img.width, box[2] + pad), min(img.height, box[3] + pad)))
    img.thumbnail((1024, 1024), Image.LANCZOS)
    return img


def training_pages(group) -> list[tuple[Image.Image, str]]:
    """Approved pages of exactly this character or group, newest first, as (image, caption)."""
    g = tuple(sorted(group))
    out = []
    for p in store.list_pages(limit=10**6):
        fb = _fb(p)
        if not fb or fb["rating"] != "good" or group_of(p) != g:
            continue
        img = _page_image(p)
        if img is None:
            continue
        style = TRAIN_STYLE_COLOR if p.get("kind") == "cover" else TRAIN_STYLE
        out.append((img, f"{scene_prompt(p)}, {style}"))
        if len(out) >= MAX_TRAINING_PAGES:
            break
    return out


def write_training_pages(group, out_dir) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, (img, caption) in enumerate(training_pages(group)):
        img.save(out_dir / f"approved_{i:02d}.png")
        (out_dir / f"approved_{i:02d}.txt").write_text(caption + "\n")
        n += 1
    return n
