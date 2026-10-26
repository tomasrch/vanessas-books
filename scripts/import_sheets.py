#!/usr/bin/env python3
"""Import a folder of character model sheets + the roster workbook into the studio.

    import_sheets.py <folder> --roster <Catalogo.xlsx> [--skip FILE ...] [--dry-run]

Each sheet's code ("Código del personaje: CB-01") is read from its info box and
looked up in the roster (Código | Personaje | Nombre científico | Grupo | ...).
Drawings are cut and captioned by coloring.reader; details and props panels are
left out. Characters are created as "new" (not trained) for review.
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import openpyxl                                     # noqa: E402
from PIL import Image                               # noqa: E402

from coloring import store                          # noqa: E402
from coloring.reader import _ocr_lines, read_sheet  # noqa: E402

# What each roster character is, in plain English for the model ("<trigger> <kind>").
# Unknown codes fall back to the roster's species name translated.
KIND = {
    "CB-01": "coqui frog", "CB-02": "manatee", "CB-03": "parrot", "CB-04": "owl",
    "CB-05": "iguana", "CB-06": "tody bird", "CB-07": "flower", "CB-08": "ceiba tree",
    "CB-09": "land crab", "CB-10": "butterfly", "CB-11": "bat", "CB-12": "sea turtle",
    "CB-13": "pelican", "CB-14": "seahorse", "CB-15": "dolphin", "CB-16": "warbler bird",
    "CB-17": "flamingo", "CB-18": "woodpecker", "CB-19": "hawk", "CB-20": "toad",
    "CB-25": "hutia rodent", "CB-26": "freshwater turtle", "CB-27": "queen conch snail",
    "CB-28": "sea snail", "CB-29": "hawksbill sea turtle", "CB-30": "puppy dog",
    "CB-31": "parrotfish", "CB-32": "snapper fish", "CB-33": "barracuda fish", "CB-34": "mahi-mahi fish",
}
# Short names she'll use in scene descriptions ("el coquí y el manatí ...").
ALIASES = {
    "CB-01": ["Coquí"], "CB-02": ["Manatí"], "CB-03": ["Cotorra"], "CB-04": ["Múcaro"],
    "CB-05": ["Iguana"], "CB-06": ["San Pedrito"], "CB-07": ["Flor de maga", "Maga"],
    "CB-08": ["Ceiba"], "CB-09": ["Juey"], "CB-10": ["Mariposa"], "CB-11": ["Murciélago"],
    "CB-12": ["Tortuga"], "CB-13": ["Pelícano"], "CB-14": ["Caballito de mar", "Caballito"],
    "CB-15": ["Delfín"], "CB-16": ["Reinita"], "CB-17": ["Flamenco"], "CB-18": ["Carpintero"],
    "CB-19": ["Guaraguao"], "CB-20": ["Sapo concho", "Sapo"],
    "CB-25": ["Jutía"], "CB-26": ["Jicotea"], "CB-27": ["Carrucho"], "CB-28": ["Burgao"],
    "CB-29": ["Carey"], "CB-30": ["Perrito taíno", "Perrito"], "CB-31": ["Pez loro"],
    "CB-32": ["Pargo criollo", "Pargo"], "CB-33": ["Picúa", "Barracuda"], "CB-34": ["Dorado"],
}


def roster(path: Path) -> dict[str, dict]:
    ws = openpyxl.load_workbook(path, data_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    head = next(i for i, r in enumerate(rows) if r and r[0] == "Código")
    cols = [str(c) for c in rows[head]]
    out = {}
    for r in rows[head + 1:]:
        if r and r[0] and re.fullmatch(r"CB-\d+", str(r[0])):
            out[str(r[0])] = dict(zip(cols, r))
    return out


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", s)


def sheet_code(img: Image.Image, cat: dict[str, dict]) -> str | None:
    """Roster code for a sheet: by the species in its info box ("Especie: Trichechus
    manatus"), which is unique - one sheet's own code was a typo (manatee said CB-18) -
    falling back to the printed code."""
    lines = [t["text"] for t in _ocr_lines(img.crop((img.width * 2 // 3, 0, img.width, img.height // 6)))]
    for ln in lines:
        if "especie" in ln.lower():
            sp = _norm(ln.split(":", 1)[-1])
            for code, info in cat.items():
                if sp and _norm(str(info.get("Nombre científico", ""))) == sp:
                    return code
    for ln in lines:
        m = re.search(r"CB\s*-\s*(\d{1,3})", ln)
        if m:
            return f"CB-{int(m.group(1)):02d}"
    return None


def slug_for(name: str) -> str:
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return store.slugify(plain)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--roster", type=Path, required=True)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--replace", action="store_true",
                    help="re-import codes that already exist (the old version is archived, not deleted)")
    ap.add_argument("--only", nargs="*", default=None, help="limit to these codes (CB-25 ...)")
    a = ap.parse_args()

    cat = roster(a.roster)
    existing = {m.get("code") for m in store.list_characters()}
    for path in sorted(a.folder.glob("*.png")):
        if path.name in a.skip:
            print(f"skip  {path.name}")
            continue
        img = Image.open(path).convert("RGB")
        named = re.match(r"(CB-\d{2})_", path.name)       # the zip names files CB-XX_Ficha_Maestra.png
        code = named.group(1) if named else sheet_code(img, cat)
        if a.only is not None and code not in a.only:
            continue
        info = cat.get(code or "")
        if not info:
            print(f"??    {path.name}: code {code!r} not in the roster - left alone")
            continue
        if code in existing and not a.replace:
            print(f"have  {code} {info['Personaje']} - already imported")
            continue
        name = str(info["Personaje"])
        items = read_sheet(img, keep_names=ALIASES.get(code, [name]) + [name])
        sci = _norm(str(info.get("Nombre científico", "")))
        for it in items:                     # the italic species line under the title
            if sci and sci in _norm(it.label):
                it.skip, it.reason = True, "species name"
        kept = [i for i in items if not i.skip]
        print(f"{code} {name}: {len(items)} drawings, {len(kept)} kept "
              f"({', '.join(i.caption for i in kept)})")
        if a.dry_run:
            continue
        for old in [c for c in store.list_characters() if c.get("code") == code]:
            if old["status"] in ("queued", "training"):
                print(f"      {code}: old version is {old['status']} - left alone")
                break
            store.archive_character(old["slug"])
            print(f"      {code}: previous version archived ({old['slug']})")
        else:
            old = None
        if old is not None and old["status"] in ("queued", "training"):
            continue
        m = store.new_character(name, KIND.get(code, name))
        # ascii slug (store.new_character keeps accents out of the folder name already)
        d = store.char_dir(m["slug"])
        img.save(d / "sheet.png")
        for n, it in enumerate(items):
            it.crop.save(d / "crops" / f"{n:02d}.png")
        m.update(
            crops=len(items), captions=[i.caption for i in items],
            labels=[i.label for i in items],
            excluded=[n for n, i in enumerate(items) if i.skip],
            skip_reasons={str(n): i.reason for n, i in enumerate(items) if i.skip},
            code=code, scientific=info.get("Nombre científico"), group=info.get("Grupo"),
            habitat=info.get("Entorno sugerido"), aliases=ALIASES.get(code, []),
            source=f"incoming/{path.name}",
        )
        store.save_character(m)
        existing.add(code)
    return 0


if __name__ == "__main__":
    sys.exit(main())
