# Coloring Studio — Vanessa's Coloring Books

A private web page for making children's coloring-book pages with your own characters.

1. **Characters:** upload a character sheet, the same character drawn in several poses in black
   lines on white. The sheet is cut into poses, and a LoRA for that character is trained in
   about an hour.
2. **Make a page:** pick one or more trained characters and describe the scene in plain words,
   using their names. Each version comes back as a pure black-and-white US Letter page
   (8.5 × 11 in, 300 dpi) as PDF and PNG.

**https://vanessasbooks.ivns.lab** (just type `vanessasbooks.ivns.lab`; http redirects to https).
Internal only: 10.3.0.0/16 plus one shared password. Fallback by IP: `http://10.3.1.240:8090/`.

- **DNS:** A record `vanessasbooks.ivns.lab -> 10.3.1.240` in the ivns.lab AD zone. **No AAAA**:
  nginx listens and allowlists on IPv4 only.
- **TLS:** self-signed (no internal CA), `/etc/nginx/ssl/vanessasbooks.{crt,key}`, SAN
  `vanessasbooks.ivns.lab`, **expires 2029-01-27** (825 days, Apple's maximum). Each device warns
  once; trusting the cert on her devices removes the warning. Renew with the same `openssl req -x509`
  command (see `deploy/nginx-coloring-studio.conf`) before then.

## How it fits on the cluster

```
browser ──> nginx :443 vanessasbooks.ivns.lab / :8090 (10.3.0.0/16 only, minus 10.3.1.254 / 10.3.1.242 relays)
              └─> coloring-studio  (FastAPI, 127.0.0.1:8095, password + Redis session)
                    ├─ upload: AV scan (clamd) -> image checks -> sheet split -> NAS
                    ├─ train:  RPUSH coloring:train:queue ──> coloring-trainer (prism)
                    │            pose captions (Ollama llava, unloaded right after)
                    │            -> dataset -> waits for a GPU with 12.5 GB free -> ai-toolkit
                    │            -> loras/<trigger>.safetensors
                    └─ page:   POST :8001/imagegen {loras:[{name:"vb/<trigger>"}...]}
                                 -> PRISM imagegen queue (Redis :6381) -> any FLUX worker
                                 (loads on demand, unloads after 10 min idle)
                               -> threshold + speck removal -> Letter PNG/PDF on the NAS
```

- **Model:** FLUX.1-schnell (Apache-2.0, so commercial use of the pages is fine), with one LoRA
  per character, trained by the same ai-toolkit setup as the PRISM cast LoRA
  (`/data/imagegen/train`).
- **Storage:** NAS dataset `Medina/VANESSA-BOOKS`, mounted read-write at
  `/mnt/Medina/VANESSA-BOOKS` on prism (LAN) and prod-prism01/02 (SAN). Every image worker reads
  the LoRAs from there as `vb/<name>` (`IMAGEGEN_LORA_EXTRA_DIRS` in `/etc/rch-imagegen.env`).
  ```
  characters/<slug>/{meta.json, sheet.png, crops/NN.png}   (+ _archived/)
  pages/<id>/{meta.json, raw.png, page.png, page.pdf, thumb.png}   (+ _archived/)
  loras/cbk_<slug>.safetensors
  train/{datasets,configs,output}/
  ```
  Nothing is deleted from the page: "Remove" moves a character or page to `_archived/`.
- **One training at a time**, on prism GPU 0 or 1, whichever has room. Running several GPUs at
  full load at once overloaded the rack's power strip on 2026-10-21.

## Deploy (prism)

```bash
sudo install -d -o tomasrch -g tomasrch /opt/coloring-studio
rsync -a --delete --exclude __pycache__ coloring /opt/coloring-studio/
sudo cp deploy/coloring-studio.service deploy/coloring-trainer.service /etc/systemd/system/
sudo cp deploy/nginx-coloring-studio.conf /etc/nginx/sites-available/coloring-studio
sudo cp deploy/nginx-snippet-allow.conf /etc/nginx/snippets/coloring-studio-allow.conf
sudo cp deploy/nginx-snippet-app.conf /etc/nginx/snippets/coloring-studio-app.conf
# cert (once, and to renew):
sudo openssl req -x509 -newkey rsa:2048 -nodes -days 825 -subj "/CN=vanessasbooks.ivns.lab" \
  -addext "subjectAltName=DNS:vanessasbooks.ivns.lab" -addext "extendedKeyUsage=serverAuth" \
  -keyout /etc/nginx/ssl/vanessasbooks.key -out /etc/nginx/ssl/vanessasbooks.crt
sudo ln -sf /etc/nginx/sites-available/coloring-studio /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo systemctl daemon-reload && sudo systemctl restart coloring-studio coloring-trainer
```

**Password:** she changes it on the page (**Change password**, top right). It needs the current
password, is at least 8 characters, and logs out every other browser. It is stored as an scrypt hash
in `/var/lib/coloring-studio/password` (service user, mode 600; the unit's `StateDirectory`).
**If it's forgotten:** delete that file and the login falls back to `COLORING_PASSWORD_HASH` in
`/etc/coloring-studio.env` (root 600). To set that one:
`cd /opt/coloring-studio && /data/rch-shared-env/bin/python -m coloring.app --hash-password`.
To log everyone out: delete the `coloring:session:*` keys in Redis :6381.

**Logs:** `journalctl -t coloring-studio -f`, `journalctl -t coloring-trainer -f`.

## Tips for character sheets

- Black lines on a white (or transparent) background, poses separated by white space.
- 4–8 poses: front, side, back, plus a couple of actions. Click a pose on the Characters tab
  to leave a bad crop out of training.
- One character per sheet. "What it is" (hedgehog, girl, dragon) helps the model place it.
