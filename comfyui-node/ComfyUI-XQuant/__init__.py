# -*- coding: utf-8 -*-
"""ComfyUI-XQuant — загрузчик нашего тернарного (−1/0/+1) формата .xqt.safetensors.
Разжимает 1.6-бит base-3 → bf16 и строит diffusion-модель штатной машинерией
ComfyUI (comfy.sd.load_diffusion_model_state_dict). Имитирует UnetLoaderGGUF, но
для нашего формата (GGML тернар нода city96 не читает)."""
import os, json, sys
import torch
import numpy as np
import folder_paths
import comfy.sd
from safetensors import safe_open

# наши unpack-функции
_XQ = r"D:/ComfyBot/comfyui_portable/ComfyUI_windows_portable"
if _XQ not in sys.path: sys.path.insert(0, _XQ)
import xquant as xq


def _list_xqt():
    names = []
    for folder in ("diffusion_models", "unet"):
        try: names += folder_paths.get_filename_list(folder)
        except Exception: pass
    return sorted({n for n in names if n.endswith(".xqt.safetensors")}) or ["(нет .xqt файлов)"]


def _dequant_ternary_file(path):
    """Прочитать .xqt → полный bf16 state_dict (тернар разжат)."""
    f = safe_open(path, framework="pt")
    meta = f.metadata() or {}
    qkeys = set(json.loads(meta.get("xq_keys", "[]")))
    group = int(meta.get("xq_group", "32"))
    pads = json.loads(meta.get("xq_pads", "{}"))
    keys = list(f.keys())
    sd = {}
    # неквантованные — как есть
    for k in keys:
        if "||" not in k:
            sd[k] = f.get_tensor(k)
    # квантованные — разжать
    for k in qkeys:
        packed = f.get_tensor(f"{k}||qpack").numpy()
        scale = f.get_tensor(f"{k}||qscl").float().numpy()
        shp = tuple(int(x) for x in f.get_tensor(f"{k}||qshp").tolist())
        gpad, ppad = pads.get(k, [0, 0])
        n_groups = scale.size
        n_codes = n_groups * group                    # с учётом pad группы
        codes = xq.unpack_tern5(packed, n_codes).astype(np.float32).reshape(n_groups, group)
        deq = (codes * scale.reshape(n_groups, 1)).reshape(-1)
        n_real = int(np.prod(shp))
        deq = deq[:n_real].reshape(shp)
        sd[k] = torch.from_numpy(deq).to(torch.bfloat16)
    return sd


class XQuantTernaryLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unet_name": (_list_xqt(),)}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "XQuant"
    TITLE = "XQuant Ternary Loader (1.6-bit)"

    def load(self, unet_name):
        path = folder_paths.get_full_path("diffusion_models", unet_name) \
            or folder_paths.get_full_path("unet", unet_name)
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"XQuant: не найден {unet_name}")
        print(f"[XQuant] разжимаю тернар {unet_name} ...")
        sd = _dequant_ternary_file(path)
        model = comfy.sd.load_diffusion_model_state_dict(sd)
        if model is None:
            raise RuntimeError("XQuant: comfy не смог построить модель из state_dict")
        print(f"[XQuant] модель собрана ({len(sd)} тензоров)")
        return (model,)


NODE_CLASS_MAPPINGS = {"XQuantTernaryLoader": XQuantTernaryLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"XQuantTernaryLoader": "XQuant Ternary Loader (1.6-bit)"}
