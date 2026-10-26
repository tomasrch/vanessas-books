#!/usr/bin/env python3
"""Cut a character's drawings by hand-measured boxes, for sheets the automatic reader
gets wrong (a reef or forest drawn behind every pose joins the drawings together).

    hand_cut.py <slug> <preview_dir>              cut + contact.png to look at
    hand_cut.py <slug> <preview_dir> --install    replace crops/ (old ones kept in crops-pre-manual/),
                                                  set labels/captions, mark queued

Boxes are in a 1400-px-wide copy of the sheet. Queue it afterwards (the page's Train button).
pez-loro was cut the same way on 2026-10-26."""
import json, os, shutil, sys
from pathlib import Path
from PIL import Image
CUTS = {
 "carey": [
  ("Vista frontal", "front view, full body, coral reef background", (25,125,410,385)),
  ("Vista lateral", "side view, full body, coral reef background", (385,140,735,390)),
  ("Vista de tres cuartos", "three-quarter view, full body, coral reef background", (720,160,1065,395)),
  ("Vista posterior", "back view, full body, shell from behind", (1075,170,1385,395)),
  ("Alegre", "head and shoulders, happy", (30,465,205,625)),
  ("Curiosa", "head and shoulders, curious", (195,465,370,625)),
  ("Sorprendida", "head and shoulders, surprised", (370,465,545,625)),
  ("Contenta", "head and shoulders, content, winking", (545,465,715,625)),
  ("Nadando en el mar", "swimming in the sea", (35,720,215,845)),
  ("Explorando el arrecife", "exploring the coral reef", (215,695,395,845)),
  ("Comiendo esponjas y otros organismos", "eating sponges on the reef", (420,700,615,860)),
  ("Descansando en el fondo marino", "resting on the sea floor, eyes closed", (635,735,830,855))],
 "picua-barracuda": [
  ("Vista frontal", "front view, full body, sharp teeth grin, coral reef background", (10,150,395,390)),
  ("Vista lateral", "side view, full body, coral reef background", (390,150,775,385)),
  ("Vista de tres cuartos", "three-quarter view, full body, coral reef background", (760,150,1135,385)),
  ("Vista posterior", "back view, full body, forked tail", (1133,180,1385,385)),
  ("Alegre", "head close-up, happy toothy grin", (30,475,185,625)),
  ("Curiosa", "head close-up, curious", (155,470,320,625)),
  ("Sorprendida", "head close-up, surprised, mouth open", (290,470,450,625)),
  ("Segura", "head close-up, confident", (420,465,595,625)),
  ("Traviesa", "head close-up, mischievous, winking", (560,485,735,625)),
  ("Nadando en mar abierto", "swimming in open water", (35,735,225,860)),
  ("Cazando su alimento", "hunting, mouth open", (220,710,415,860)),
  ("Explorando su entorno", "exploring its surroundings", (430,730,630,830)),
  ("Nadando rápidamente", "swimming fast, leaping upward with a splash", (640,680,830,860))],
}
CUTS["pargo-criollo"] = [
  ("Vista frontal", "front view, full body, coral reef background", (25,120,305,390)),
  ("Vista lateral", "side view, full body, coral reef background", (315,125,705,385)),
  ("Vista de tres cuartos", "three-quarter view, full body, coral reef background", (720,125,1030,385)),
  ("Vista posterior", "back view, full body, tail fin", (1140,163,1375,385)),
  ("Alegre", "head close-up, happy", (25,470,172,630)),
  ("Curioso", "head close-up, curious", (170,475,312,630)),
  ("Sorprendida", "head close-up, surprised, mouth open", (305,470,445,630)),
  ("Contento", "head close-up, content, eyes closed smiling", (440,470,585,630)),
  ("Juguetón", "head close-up, playful, eyes closed laughing", (578,475,725,630)),
  ("Nadando en el arrecife", "swimming in the coral reef", (40,710,225,850)),
  ("Explorando su entorno", "exploring its surroundings", (250,730,420,850)),
  ("Buscando alimento entre las rocas", "looking for food among the rocks", (435,700,600,830)),
  ("Nadando en grupo", "swimming in a group with another snapper", (610,690,820,840))]
CUTS["dorado-dolphin-fish-mahi"] = [
  ("Vista frontal", "front view, full body, ocean background", (10,115,342,390)),
  ("Vista lateral", "side view, full body, ocean background", (345,135,728,385)),
  ("Vista de tres cuartos", "three-quarter view, full body, ocean background", (730,140,1110,385)),
  ("Vista posterior", "back view, full body, forked tail", (1133,180,1385,385)),
  ("Alegre", "head close-up, happy", (25,465,170,625)),
  ("Curioso", "head close-up, curious", (160,465,300,625)),
  ("Sorprendido", "head close-up, surprised, mouth open", (295,460,440,625)),
  ("Contento", "head close-up, content, winking", (430,460,580,625)),
  ("Juguetón", "head close-up, playful", (580,460,725,625)),
  ("Nadando en manada", "swimming in a school of fish", (30,730,220,855)),
  ("Saltando sobre el agua", "leaping out of the water with a splash", (230,680,420,855)),
  ("Nadando junto a otras", "swimming together with others", (435,700,610,855)),
  ("Buscando alimento", "looking for food", (595,700,795,830))]
WHITE = {("picua-barracuda", 2): [(1005, 150, 1135, 182)],
         ("pargo-criollo", 2): [(1010, 120, 1030, 160)],
         ("dorado-dolphin-fish-mahi", 2): [(1000, 140, 1110, 172)]}
slug, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
d = Path("/mnt/Medina/VANESSA-BOOKS/characters") / slug
s = Image.open(d / "sheet.png").convert("RGB"); k = s.width / 1400
cuts = CUTS[slug]
ims = []
for i, (_, _, b) in enumerate(cuts):
    c = s.crop(tuple(int(v * k) for v in b))
    for wx0, wy0, wx1, wy1 in WHITE.get((slug, i), []):     # the info box corner
        c.paste((255, 255, 255), (int((wx0 - b[0]) * k), int((wy0 - b[1]) * k), int((wx1 - b[0]) * k), int((wy1 - b[1]) * k)))
    c.save(out / f"{i:02d}.png")
    t = c.copy(); t.thumbnail((260, 260)); ims.append(t)
sheet = Image.new("RGB", (6 * 270, 3 * 270), "white")
for i, t in enumerate(ims): sheet.paste(t, ((i % 6) * 270, (i // 6) * 270))
sheet.save(out / "contact.png")
if "--install" in sys.argv:
    bak = d / "crops-pre-manual"
    if not bak.exists(): shutil.move(str(d / "crops"), str(bak))
    else: shutil.rmtree(d / "crops")
    (d / "crops").mkdir()
    for i in range(len(cuts)): shutil.copy2(out / f"{i:02d}.png", d / "crops" / f"{i:02d}.png")
    m = json.loads((d / "meta.json").read_text())
    m.update(crops=len(cuts), labels=[c[0] for c in cuts], captions=[c[1] for c in cuts], excluded=[],
             skip_reasons={}, status="queued", error=None)
    m.setdefault("notes", []).append("2026-10-26: drawings cut by hand (reef background confused the cutter). Old crops in crops-pre-manual/")
    tmp = d / "meta.json.tmp"; tmp.write_text(json.dumps(m, indent=2)); os.replace(tmp, d / "meta.json")
    print("installed", slug)
