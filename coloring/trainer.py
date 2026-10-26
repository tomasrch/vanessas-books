"""coloring-trainer: trains one character LoRA at a time from the Redis queue.

    coloring:train:queue            list of character slugs (RPUSH by the web app)
    coloring:train:progress:<slug>  hash {stage, step, total} for the progress bar

Per job: line-art pose crops (cut at upload) -> pose captions from the local
vision model -> dataset -> wait for a prism GPU with room -> ai-toolkit ->
loras/<trigger>.safetensors, which every image worker reads as "vb/<trigger>".
One run at a time on purpose: several GPUs at full load at once overloaded the
rack's power strip on 2026-10-21.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import redis
from PIL import Image

from . import config, store
from .dataset import build, build_composites, describe_pose
from .sheet import is_color, line_art

log = logging.getLogger("coloring-trainer")
QUEUE = "coloring:train:queue"
PROGRESS = "coloring:train:progress:"
OOM_RETRIES = "coloring:train:oom"
STEP_RE = re.compile(rb"(\d+)/(\d+) \[")

CONFIG_TEMPLATE = """\
# Written by coloring-trainer for {name}; do not edit, it is rebuilt each run.
job: extension
config:
  name: "{trigger}"
  process:
    - type: 'sd_trainer'
      training_folder: "{output}"
      device: cuda:0
      network: {{type: "lora", linear: {rank}, linear_alpha: {rank}}}
      save: {{dtype: float16, save_every: 250, max_step_saves_to_keep: 2, push_to_hub: false}}
      datasets:
        - folder_path: "{dataset}"
          caption_ext: "txt"
          caption_dropout_rate: 0.0
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: [ 512, 768, 1024 ]
      train:
        batch_size: 1
        steps: {steps}
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        lr: 1e-4
        cache_text_embeddings: true
        disable_sampling: true
        ema_config: {{use_ema: true, ema_decay: 0.99}}
        dtype: bf16
      model:
        name_or_path: "{base}"
        is_flux: true
        quantize: true
        qtype: "uint4"
        low_vram: true
      sample:
        sampler: "flowmatch"
        sample_every: 250
        width: 1024
        height: 1024
        prompts: ["{trigger}"]
"""


def redis_client() -> redis.Redis:
    return redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, password=config.REDIS_PASSWORD,
                       decode_responses=True, socket_timeout=60)


_gpu_lock = threading.Lock()
_gpus_in_use: set[int] = set()


def claim_gpu() -> int | None:
    """A training GPU with TRAIN_MIN_FREE_GB free that no other slot is using, most free
    first; None if none has room. Released with release_gpu()."""
    import pynvml
    pynvml.nvmlInit()
    with _gpu_lock:
        best = None
        for i in config.TRAIN_GPUS:
            if i in _gpus_in_use:
                continue
            free = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).free / 2**30
            if free >= config.TRAIN_MIN_FREE_GB and (best is None or free > best[1]):
                best = (i, free)
        if best:
            _gpus_in_use.add(best[0])
        return best[0] if best else None


def release_gpu(gpu: int) -> None:
    with _gpu_lock:
        _gpus_in_use.discard(gpu)


def set_progress(r: redis.Redis, key: str, stage: str, step: int = 0, total: int = 0) -> None:
    r.hset(PROGRESS + key, mapping={"stage": stage, "step": step, "total": total})
    r.expire(PROGRESS + key, 7 * 86400)


def member_images(meta: dict, describe: bool = True):
    """A character's kept crops, their drawn line partners and pose captions."""
    crop_dir = store.char_dir(meta["slug"]) / "crops"
    files = sorted(crop_dir.glob("[0-9][0-9].png"))      # NN-line.png = optional drawn partner
    kept = [i for i in range(len(files)) if i not in set(meta.get("excluded", []))]
    crops = [Image.open(files[i]) for i in kept]
    lines = [Image.open(p) if (p := files[i].with_name(f"{i:02d}-line.png")).is_file() else None
             for i in kept]
    given = meta.get("captions") or []      # from the sheet's labels or a reference guide
    poses = [given[i] if i < len(given) and given[i] else (describe_pose(c) if describe else "")
             for i, c in zip(kept, crops)]
    return crops, lines, poses


def run_toolkit(r: redis.Redis, key: str, name: str, trigger: str, dataset: Path, n: int) -> Path:
    """Write the config, wait for a GPU, train; returns the finished LoRA file."""
    output = config.TRAIN / "output"
    # A previous run's checkpoints would make ai-toolkit resume instead of starting over.
    shutil.rmtree(output / trigger, ignore_errors=True)
    steps = min(3000, max(config.TRAIN_STEPS, 20 * n))   # ~20 steps per image
    cfg = config.TRAIN / "configs" / f"{trigger}.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(CONFIG_TEMPLATE.format(
        name=name.replace('"', "'"), kind="", trigger=trigger, output=output,
        dataset=dataset, rank=config.TRAIN_RANK, steps=steps, base=config.TRAIN_BASE))
    log.info("%s: %d training images, %d steps", key, n, steps)

    set_progress(r, key, "waiting for a free GPU")
    while (gpu := claim_gpu()) is None:
        time.sleep(30)
    try:
        log.info("%s: training on GPU %d", key, gpu)
        set_progress(r, key, "training", 0, steps)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "HF_HOME": config.TRAIN_HF_HOME,
               "PYTORCH_ALLOC_CONF": "expandable_segments:True"}
        proc = subprocess.Popen([config.TRAIN_PYTHON, "run.py", str(cfg)], cwd=config.TRAIN_TOOLKIT,
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        tail, last = b"", 0.0
        while chunk := proc.stdout.read1(65536):
            tail = (tail + chunk)[-8192:]
            done = STEP_RE.findall(chunk)
            if done and time.time() - last > 10:
                set_progress(r, key, "training", int(done[-1][0]), steps)
                last = time.time()
        if proc.wait() != 0:
            msg = tail.decode(errors="replace").replace("\r", "\n").strip().splitlines()[-3:]
            raise RuntimeError("training failed: " + " | ".join(msg))
    finally:
        release_gpu(gpu)
    set_progress(r, key, "done", steps, steps)
    return output / trigger / f"{trigger}.safetensors"


def install_lora(result: Path, dest: Path) -> None:
    config.LORAS.mkdir(parents=True, exist_ok=True)
    tmp = config.LORAS / f".{dest.name}.tmp"
    shutil.copyfile(result, tmp)
    tmp.chmod(0o644)
    os.replace(tmp, dest)       # workers key adapters by mtime, so no restart needed


def train(r: redis.Redis, slug: str) -> None:
    meta = store.get_character(slug)
    if not meta:
        log.warning("%s: no such character, skipped", slug)
        return
    trigger = meta["trigger"]
    meta.update(status="training", error=None)
    store.save_character(meta)
    set_progress(r, slug, "preparing")
    crops, lines, poses = member_images(meta)
    if not crops:
        raise RuntimeError("no poses selected")
    dataset = config.TRAIN / "datasets" / trigger
    shutil.rmtree(dataset, ignore_errors=True)
    n = build(crops, trigger, meta["kind"], dataset, poses, lines=lines)
    install_lora(run_toolkit(r, slug, meta["name"], trigger, dataset, n), store.lora_file(meta))
    meta = store.get_character(slug)
    meta.update(status="ready", trained_at=time.time(), error=None)
    store.save_character(meta)
    log.info("%s: ready (%s)", slug, store.lora_file(meta))


def train_cast(r: redis.Redis, cid: str) -> None:
    """One LoRA for a group of characters: every member's own images under its own
    trigger, plus composites of them side by side, so they stay distinct together."""
    cast = store.get_cast(cid)
    if not cast:
        return
    cast.update(status="training", error=None)
    store.save_cast(cast)
    key = f"cast:{cid}"
    set_progress(r, key, "preparing")
    trigger = store.cast_trigger(cid)
    dataset = config.TRAIN / "datasets" / trigger
    shutil.rmtree(dataset, ignore_errors=True)
    n, members = 0, []
    for slug in cast["members"]:
        meta = store.get_character(slug)
        if not meta or meta["status"] != "ready":
            raise RuntimeError(f"{slug} must be trained before its cast")
        crops, lines, poses = member_images(meta, describe=False)
        n += build(crops, meta["trigger"], meta["kind"], dataset / slug, poses, lines=lines)
        line_imgs = [l if l is not None else (line_art(c) if is_color(c) else c) for c, l in zip(crops, lines)]
        members.append({"trigger": meta["trigger"], "kind": meta["kind"], "images": line_imgs})
    n += build_composites(members, dataset / "together", count=max(30, 15 * len(members)))
    flat = dataset / "all"                       # ai-toolkit reads one folder
    flat.mkdir()
    for f in dataset.glob("*/*"):
        if f.parent != flat:
            f.rename(flat / f"{f.parent.name}__{f.name}")
    names = ", ".join(store.get_character(s)["name"] for s in cast["members"])
    install_lora(run_toolkit(r, key, names, trigger, flat, n), store.cast_lora(cid))
    cast = store.get_cast(cid)
    cast.update(status="ready", trained_at=time.time(), error=None)
    store.save_cast(cast)
    log.info("cast %s (%s): ready", cid, names)


def work(r: redis.Redis, slot: int) -> None:
    while True:
        try:
            item = r.blpop(QUEUE, timeout=30)
        except Exception as exc:
            log.error("slot %d: broker error %s", slot, exc)
            time.sleep(10)
            continue
        if not item:
            continue
        job = item[1]
        try:
            if job.startswith("cast:"):
                train_cast(r, job[5:])
            else:
                train(r, job)
            r.hdel(OOM_RETRIES, job)
        except Exception as exc:
            log.exception("%s failed", job)
            # Out of memory means something else grabbed the card mid-start (a render, a
            # vision model): not the job's fault, so put it back and try again later.
            if "out of memory" in str(exc).lower() and r.hincrby(OOM_RETRIES, job, 1) <= 3:
                log.warning("%s: GPU out of memory, requeued", job)
                set_progress(r, job, "queued")
                if not job.startswith("cast:") and (meta := store.get_character(job)):
                    meta.update(status="queued")
                    store.save_character(meta)
                time.sleep(120)
                r.rpush(QUEUE, job)
                continue
            r.hdel(OOM_RETRIES, job)
            if job.startswith("cast:"):
                cast = store.get_cast(job[5:])
                if cast:
                    cast.update(status="failed", error=str(exc)[:500])
                    store.save_cast(cast)
            else:
                meta = store.get_character(job)
                if meta:
                    meta.update(status="failed", error=str(exc)[:500])
                    store.save_character(meta)
            set_progress(r, job, "failed")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    r = redis_client()
    queued = r.lrange(QUEUE, 0, -1)
    # Runs cut off by a restart or power loss start again.
    for m in store.list_characters():
        if m["status"] in ("training", "queued") and m["slug"] not in queued:
            r.rpush(QUEUE, m["slug"])
            log.info("%s: requeued after restart", m["slug"])
    casts = config.ROOT / "casts"
    for p in (casts.glob("*.json") if casts.is_dir() else []):
        c = store.get_cast(p.stem)
        if c and c["status"] in ("training", "queued") and f"cast:{c['id']}" not in queued:
            r.rpush(QUEUE, f"cast:{c['id']}")
            log.info("cast %s: requeued after restart", c["id"])
    slots = max(1, config.TRAIN_PARALLEL)
    log.info("coloring-trainer ready (GPUs %s, %d at a time)", config.TRAIN_GPUS, slots)
    threads = [threading.Thread(target=work, args=(r, i), daemon=True) for i in range(slots)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())
