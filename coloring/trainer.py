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
import time
from pathlib import Path

import redis
from PIL import Image

from . import config, store
from .dataset import build, describe_pose

log = logging.getLogger("coloring-trainer")
QUEUE = "coloring:train:queue"
PROGRESS = "coloring:train:progress:"
STEP_RE = re.compile(rb"(\d+)/(\d+) \[")

CONFIG_TEMPLATE = """\
# Written by coloring-trainer for {name} ({kind}); do not edit, it is rebuilt each run.
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


def free_gpu() -> int | None:
    """A training GPU with TRAIN_MIN_FREE_GB free, most free first; None if none has room."""
    import pynvml
    pynvml.nvmlInit()
    best = None
    for i in config.TRAIN_GPUS:
        free = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).free / 2**30
        if free >= config.TRAIN_MIN_FREE_GB and (best is None or free > best[1]):
            best = (i, free)
    return best[0] if best else None


def set_progress(r: redis.Redis, slug: str, stage: str, step: int = 0, total: int = 0) -> None:
    r.hset(PROGRESS + slug, mapping={"stage": stage, "step": step, "total": total})
    r.expire(PROGRESS + slug, 7 * 86400)


def train(r: redis.Redis, slug: str) -> None:
    meta = store.get_character(slug)
    if not meta:
        log.warning("%s: no such character, skipped", slug)
        return
    trigger = meta["trigger"]
    meta.update(status="training", error=None)
    store.save_character(meta)

    set_progress(r, slug, "preparing")
    crop_dir = store.char_dir(slug) / "crops"
    crops = [Image.open(p) for i, p in enumerate(sorted(crop_dir.glob("*.png")))
             if i not in set(meta.get("excluded", []))]
    if not crops:
        raise RuntimeError("no poses selected")
    poses = [describe_pose(c) for c in crops]
    dataset = config.TRAIN / "datasets" / trigger
    output = config.TRAIN / "output"
    shutil.rmtree(dataset, ignore_errors=True)
    # A previous run's checkpoints would make ai-toolkit resume instead of starting over.
    shutil.rmtree(output / trigger, ignore_errors=True)
    n = build(crops, trigger, meta["kind"], dataset, poses)
    cfg = config.TRAIN / "configs" / f"{trigger}.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(CONFIG_TEMPLATE.format(
        name=meta["name"].replace('"', "'"), kind=meta["kind"], trigger=trigger, output=output,
        dataset=dataset, rank=config.TRAIN_RANK, steps=config.TRAIN_STEPS, base=config.TRAIN_BASE))
    log.info("%s: %d poses -> %d training images", slug, len(crops), n)

    set_progress(r, slug, "waiting for a free GPU")
    while (gpu := free_gpu()) is None:
        time.sleep(30)
    log.info("%s: training on GPU %d", slug, gpu)
    set_progress(r, slug, "training", 0, config.TRAIN_STEPS)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "HF_HOME": config.TRAIN_HF_HOME,
           "PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    proc = subprocess.Popen([config.TRAIN_PYTHON, "run.py", str(cfg)], cwd=config.TRAIN_TOOLKIT,
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tail, last = b"", 0.0
    while chunk := proc.stdout.read1(65536):
        tail = (tail + chunk)[-8192:]
        steps = STEP_RE.findall(chunk)
        if steps and time.time() - last > 10:
            set_progress(r, slug, "training", int(steps[-1][0]), config.TRAIN_STEPS)
            last = time.time()
    if proc.wait() != 0:
        msg = tail.decode(errors="replace").replace("\r", "\n").strip().splitlines()[-3:]
        raise RuntimeError("training failed: " + " | ".join(msg))

    result = output / trigger / f"{trigger}.safetensors"
    config.LORAS.mkdir(parents=True, exist_ok=True)
    tmp = config.LORAS / f".{trigger}.tmp"
    shutil.copyfile(result, tmp)
    os.replace(tmp, store.lora_file(meta))   # workers key adapters by mtime, so no restart needed
    meta = store.get_character(slug)
    meta.update(status="ready", trained_at=time.time(), error=None)
    store.save_character(meta)
    set_progress(r, slug, "done", config.TRAIN_STEPS, config.TRAIN_STEPS)
    log.info("%s: ready (%s)", slug, store.lora_file(meta))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    r = redis_client()
    # A run cut off by a restart or power loss starts again.
    for m in store.list_characters():
        if m["status"] in ("training", "queued") and m["slug"] not in r.lrange(QUEUE, 0, -1):
            r.rpush(QUEUE, m["slug"])
            log.info("%s: requeued after restart", m["slug"])
    log.info("coloring-trainer ready (GPUs %s)", config.TRAIN_GPUS)
    while True:
        item = r.blpop(QUEUE, timeout=30)
        if not item:
            continue
        slug = item[1]
        try:
            train(r, slug)
        except Exception as exc:
            log.exception("%s failed", slug)
            meta = store.get_character(slug)
            if meta:
                meta.update(status="failed", error=str(exc)[:500])
                store.save_character(meta)
            set_progress(r, slug, "failed")


if __name__ == "__main__":
    sys.exit(main())
