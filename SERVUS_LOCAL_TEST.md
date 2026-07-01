# Servus local ComfyUI test notes

Purpose: local image-generation smoke test for a Servus-style 3D cleaning/service mascot.

## Source image prompt notes

A friendly 3D cartoon mascot of a professional cleaning service worker, young adult male with a warm smile, large expressive eyes, thick eyebrows, clean stylized facial features, navy blue baseball cap with an orange U-style logo, navy polo shirt with white collar and sleeve piping, white lowercase servus-style chest branding, navy work pants, black belt with silver buckle, bright orange rubber cleaning gloves, white sneakers, and a white cleaning towel tucked into his belt with an orange U-style logo. Full-body confident pose, one hand on hip, one arm leaning on a large bright orange cleaning prop, legs crossed casually. Dark navy gradient studio background, small sparkle shine effect, glossy reflective floor. Polished high-quality 3D render, friendly commercial brand mascot style, clean professional lighting, smooth textures, vibrant navy-orange-white color palette.

Negative prompt: photorealistic human, dirty clothing, cluttered background, unreadable text, distorted hands, extra fingers, harsh shadows, low resolution, scary expression, wrong colors.

## Audit summary

Audited upstream ComfyUI at commit `2c935de1b1cf7f03d2412a1d0bf1ed2685157c27`.

Security checks run:

- `pip-audit -r requirements.txt`: no known dependency vulnerabilities found.
- `bandit -r . -lll`: initially flagged SHA1 use in `comfy_api_nodes/util/request_logger.py`; fixed by marking SHA1 as non-security use with `usedforsecurity=False` because the digest is only a short log filename disambiguator, not a security primitive.
- `gitleaks detect --no-git`: one reviewed false positive on `embedding_key='mistral3_24b'` in `comfy/text_encoders/flux.py`; no real secret was present.

