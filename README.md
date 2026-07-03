# XQuant — custom diffusion-model quantizer (zero third-party engine)

Compress diffusion models (FLUX / SDXL / SD1.5 / SD3 / Qwen-Image / Wan …) to
**2 / 3 / 4-bit GGUF** with a quantization engine written **entirely from scratch** —
our own GGML-byte-exact kernel, our own GGUF writer, our own architecture detection.
Drop a `bf16`/`fp16` model, get a compressed `.gguf` that runs in ComfyUI.

## Zero third-party engine
The standalone engine depends on **numpy only** — no `llama.cpp`, no `gguf` library,
no City96 `convert.py`, no torch:
- **`xquant.py`** — quantization kernel from scratch. Our `Q4_0` output is
  **byte-for-byte identical** to the GGUF reference; `Q3_K` / `Q2_K` verified via
  round-trip through the ComfyUI-GGUF decoder.
- **`xgguf.py`** — our own GGUF v3 writer (validated by an independent GGUF reader).
- **`xquant_standalone.py`** — own safetensors reader (manual `bf16`/`fp16`/`fp8`
  decode) + own architecture detection. **No external quantizer code.**

## LLM requantization (GGUF → GGUF, no re-download)
Drop an existing **LLM GGUF** (from LM Studio etc.) and shrink it further —
no need to download the raw 15 GB safetensors. XQuant reads the GGUF, dequantizes
weights in RAM (F16 / BF16 / F32 / Q8_0 source), applies critical-layer protection,
and repacks to 2/3/4/5/6/8-bit — **preserving all metadata and the tokenizer**
verbatim (raw KV passthrough), so the result loads in llama.cpp / LM Studio.
```
XQuant.exe model-F16.gguf Q4_0        # 8-bit/F16 LLM → 4-bit, tokenizer intact
```
(Best source = F16 or Q8_0. Re-quantizing an already-lossy K-quant is skipped.)

## Universal critical-layer protection
Input embeddings, the output projection to the VAE (`final_layer` / `conv_out` /
`proj_out`), and norms are kept in `bf16` for **every** architecture — so the VAE
connection is never broken (the classic sub-4-bit "colour noise").

## Results (FLUX.1-dev, tested by real generation)
| bits | size | PSNR vs fp16 | verdict |
|---|---|---|---|
| fp16 | 23.8 GB | — | reference |
| 4-bit Q4_0 | 6.4 GB | ~25 dB | perfect, indistinguishable |
| 3-bit Q3_K | ~5 GB | ~18 dB | balanced |
| **2-bit Q2_K** | **4.0 GB** | ~19 dB | **size/quality sweet spot** |

**Post-hoc quantization floor ≈ 2-bit.** Below that (ternary 1.6-bit) quality
collapses without quantization-aware training.

## Usage
**Easiest — the standalone `XQuant.exe`** (20 MB, no Python needed, numpy bundled):
drag a `.safetensors` model onto `XQuant.exe` → get `<model>-Q2_K.gguf` next to it.
Or from a terminal: `XQuant.exe <model.safetensors> [Q4_0|Q3_K|Q2_K]`.

With Python instead:
```
python xquant_standalone.py <model.safetensors> [Q4_0|Q3_K|Q2_K]
```
Output: `<model>-<qtype>.gguf` next to the input. Load 2/3/4-bit GGUF with the
`UnetLoaderGGUF` node in ComfyUI.

## License — AGPL-3.0
Licensed under **GNU AGPL-3.0**. Anyone who uses, modifies, or serves this code
over a network **must release their full source under the same license.**
Commercial closed-source use is not permitted.

*(Optional `xquant_tool.py` variant reuses City96's ComfyUI-GGUF `convert.py`,
Apache-2.0 — see `NOTICE`. The standalone engine above uses none of it.)*
