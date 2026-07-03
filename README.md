# XQuant — custom diffusion-model quantizer

Own quantization kernel that writes **byte-exact GGML** (our Q4_0 is byte-for-byte
identical to the `gguf` reference) plus a custom sub-2-bit ternary format (−1/0/+1)
with a ComfyUI loader node. Drag-and-drop a `.safetensors` model, get a compressed
GGUF you can run in ComfyUI.

## Features
- **Own kernel** (`xquant.py`) — Q4_0 / Q3_K / Q2_K written from scratch (not a
  library call), verified byte-identical / round-trip against the GGUF decoder.
- **CLI tool** (`xquant_tool.py`) — auto-detects architecture (FLUX / Qwen / SDXL /
  SD3 / Wan …) and compresses.
- **Drag-and-drop** (`bats/`) — drop a model onto `Ужать_Q2_K.bat` / `Q3_K` / `Q4_0`.
- **ComfyUI node** (`comfyui-node/ComfyUI-XQuant`) — loads the custom ternary format.
- Critical layers (`final_layer`→VAE, `img_in`/`txt_in`, norms) are kept in bf16 so
  the VAE connection is never broken.

## Bitrate switch (pick per your GPU)
- **2 / 3 / 4-bit** → GGUF files, loaded by the standard `UnetLoaderGGUF` node
  (file dropdown: `model-Q2_K / -Q3_K / -Q4_0.gguf`).
- **1.6-bit ternary** → `.xqt`, loaded by our `XQuant Ternary Loader` node.

## Results (FLUX.1-dev, tested by actual generation)
| bits | size | PSNR vs fp16 | verdict |
|---|---|---|---|
| fp16 | 23.8 GB | — | reference |
| 4-bit Q4_0 | 6.4 GB | ~25 dB | perfect, indistinguishable |
| 3-bit Q3_K | ~5 GB | ~18 dB | balanced |
| 2-bit Q2_K | 3.9 GB | ~19 dB | size/quality sweet spot |
| 1.6-bit ternary | — | ~8 dB | too lossy post-hoc; needs QAT/healing |

**Floor of post-hoc quantization ≈ 2 bit.** Below that, quality collapses without
quantization-aware training (BitNet-style).

## Usage
```
python xquant_tool.py <model.safetensors> [Q4_0|Q3_K|Q2_K]
```
Output: `<model>-<qtype>.gguf` next to the input.

## License — AGPL-3.0
This project is licensed under **GNU AGPL-3.0** (see `LICENSE`). Anyone who uses,
modifies, or serves this code over a network **must release their full source under
the same license.** Commercial closed-source use is not permitted.

Uses City96's ComfyUI-GGUF `convert.py` (Apache-2.0) — see `NOTICE`.
