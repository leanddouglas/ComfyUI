#!/usr/bin/env python3
"""Queue a small Servus-style mascot smoke-test prompt against a local ComfyUI server.

Default server: http://127.0.0.1:8199
Expected model: models/checkpoints/v1-5-pruned-emaonly.safetensors
"""

import json
import pathlib
import sys
import time
import urllib.request
import uuid

SERVER = "http://127.0.0.1:8199"

PROMPT_TEXT = (
    "single full body friendly 3D cartoon male mascot of a professional cleaning service worker, "
    "centered hero pose, warm smile, large expressive eyes, navy blue baseball cap, navy polo uniform "
    "with white collar trim, navy work pants, black belt, bright orange rubber cleaning gloves, white sneakers, "
    "white cleaning towel tucked into belt, confident relaxed pose, one hand on hip, leaning on a large orange "
    "cleaning equipment prop, dark navy studio background, sparkle shine, glossy reflective floor, polished high "
    "quality 3D render, commercial mascot advertising style, smooth textures, navy orange white color palette, "
    "one character only, full body portrait"
)

NEGATIVE_PROMPT = (
    "photorealistic, dirty clothing, clutter, bad hands, extra fingers, distorted face, scary, low quality, "
    "blurry, unreadable text, watermark, wrong colors, multiple characters, animal mascot, concept sheet, "
    "collage, hats only, cropped body"
)


def queue_prompt(server: str = SERVER) -> pathlib.Path:
    prompt = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 424243,
                "steps": 20,
                "cfg": 7.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 768, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT_TEXT, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE_PROMPT, "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "servus_mascot_single_test", "images": ["8", 0]}},
    }
    body = json.dumps({"prompt": prompt, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(server + "/prompt", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as response:
        queued = json.load(response)
    print("QUEUED", json.dumps(queued, indent=2), flush=True)
    prompt_id = queued["prompt_id"]

    start = time.time()
    while True:
        with urllib.request.urlopen(server + "/history/" + prompt_id, timeout=30) as response:
            history = json.load(response)
        if prompt_id in history:
            item = history[prompt_id]
            print("HISTORY_READY", json.dumps(item.get("status", {}), indent=2), flush=True)
            images = []
            for node_output in item.get("outputs", {}).values():
                images.extend(node_output.get("images", []))
            if not images:
                raise RuntimeError("ComfyUI completed without an image output")
            image = images[0]
            output_path = pathlib.Path("output") / image.get("subfolder", "") / image["filename"]
            print("SAVED_PATH", output_path.resolve(), flush=True)
            print("ELAPSED_SECONDS", round(time.time() - start, 2), flush=True)
            return output_path.resolve()
        if time.time() - start > 540:
            raise TimeoutError("Timed out waiting for ComfyUI generation")
        time.sleep(2)


if __name__ == "__main__":
    server = sys.argv[1] if len(sys.argv) > 1 else SERVER
    queue_prompt(server)
