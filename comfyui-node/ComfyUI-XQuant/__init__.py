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
    CATEGORY = "Жматель"
    TITLE = "Жматель Ternary Loader (1.6-bit)"

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


# ═══════════ GGUF-загрузчик (2/3/4/5/6/8-бит, картинка/видео/аудио) ═══════════
# Разжимает НАШ GGUF (из XQuant.exe) в полный state_dict и строит модель штатной
# машинерией ComfyUI. Арх (flux/sd3/sdxl/stable_audio/…) ComfyUI определяет сам
# по ключам — поэтому одна нода тянет и картиночные, и аудио-диффузии, если
# ComfyUI знает эту архитектуру. dec_source жмёт ВСЕ кванты (Q2_K…Q6_K/Q4_0/Q8_0).
from . import xgguf as _xg  # вендорная копия рядом с нодой


def _list_gguf():
    names = []
    for folder in ("diffusion_models", "unet"):
        try: names += folder_paths.get_filename_list(folder)
        except Exception: pass
    return sorted({n for n in names if n.endswith(".gguf")}) or ["(нет .gguf файлов)"]


def _dtype_for(tt):
    if tt == _xg.T.F32:  return torch.float32
    if tt == _xg.T.F16:  return torch.float16
    if tt == _xg.T.BF16: return torch.bfloat16
    return torch.bfloat16   # разжатый квант → bf16 (как ждёт FLUX/SD3)


def _dequant_gguf_file(path):
    """Прочитать наш GGUF → полный state_dict (все кванты разжаты в bf16/fp)."""
    f, ver, raw_meta, n_kv, tinfos, data_start, align = _xg.read_gguf(path)
    fsize = os.path.getsize(path)
    offs = [t[3] for t in tinfos]
    sd = {}
    try:
        for i, (name, dims, tt, off) in enumerate(tinfos):
            start = data_start + off
            end = data_start + offs[i + 1] if i + 1 < len(tinfos) else fsize
            f.seek(start); raw = f.read(end - start)
            shape = tuple(reversed(dims))                 # ggml ne → torch shape
            nelem = int(np.prod(shape)) if shape else 1
            arr = _xg.dec_source(raw, tt, nelem)
            if arr is None:
                raise RuntimeError(f"XQuant: не могу разжать тензор '{name}' (ggml type {tt})")
            arr = np.asarray(arr, np.float32).reshape(shape if shape else (1,))
            sd[name] = torch.from_numpy(arr.copy()).to(_dtype_for(tt))
    finally:
        f.close()
    return sd


class XQuantGGUFLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unet_name": (_list_gguf(),)}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "Жматель"
    TITLE = "Жматель GGUF Loader (2-8bit · image/video/audio)"

    def load(self, unet_name):
        path = folder_paths.get_full_path("diffusion_models", unet_name) \
            or folder_paths.get_full_path("unet", unet_name)
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"XQuant: не найден {unet_name}")
        print(f"[XQuant] разжимаю GGUF {unet_name} ...")
        sd = _dequant_gguf_file(path)
        model = comfy.sd.load_diffusion_model_state_dict(sd)
        if model is None:
            raise RuntimeError(
                "XQuant: ComfyUI не распознал архитектуру из state_dict. "
                "Картиночные (FLUX/SD3/SDXL) грузятся; для аудио нужна поддержка "
                "этой модели самим ComfyUI (нода строит модель его же машинерией)."
            )
        print(f"[XQuant] модель собрана из GGUF ({len(sd)} тензоров)")
        return (model,)


# ═══════════ МУЗЫКАЛЬНЫЙ загрузчик (Stable Audio / ACE-Step) ═══════════
# Отдельная нода под аудио-диффузию: ищет .gguf ещё и в checkpoints/audio-папках,
# разжимает так же, и ПРОВЕРЯЕТ что ComfyUI распознал именно аудио-модель
# (StableAudio*/ACEStep*) — иначе даёт понятную ошибку вместо тихой картиночной
# модели. Ключи-детекторы ComfyUI: Stable Audio — transformer.rotary_pos_emb.inv_freq;
# ACE-Step — genre_embedder.weight (шейпы тензоров читаются из state_dict, который
# наш загрузчик восстанавливает целиком с верными формами).

def _list_gguf_music():
    names = []
    for folder in ("diffusion_models", "unet", "checkpoints", "audio_checkpoints", "audio"):
        try: names += folder_paths.get_filename_list(folder)
        except Exception: pass
    return sorted({n for n in names if n.endswith(".gguf")}) or ["(нет .gguf файлов)"]


def _find_music_path(name):
    for folder in ("diffusion_models", "unet", "checkpoints", "audio_checkpoints", "audio"):
        try:
            p = folder_paths.get_full_path(folder, name)
            if p and os.path.isfile(p): return p
        except Exception: pass
    return None


# как ComfyUI помечает аудио-модели (comfy/supported_models.py)
_AUDIO_MODEL_HINTS = ("stableaudio", "ace_step", "acestep", "ace-step")


class XQuantMusicLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unet_name": (_list_gguf_music(),)}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "Жматель"
    TITLE = "Жматель Music Loader (Stable Audio / ACE-Step, 2-8bit)"

    def load(self, unet_name):
        path = _find_music_path(unet_name)
        if not path:
            raise FileNotFoundError(f"XQuant: не найден {unet_name}")
        print(f"[XQuant] разжимаю музыкальный GGUF {unet_name} ...")
        sd = _dequant_gguf_file(path)
        model = comfy.sd.load_diffusion_model_state_dict(sd)
        if model is None:
            raise RuntimeError(
                "XQuant Music: ComfyUI не распознал арх из state_dict. Это должна быть "
                "аудио-ДИФФУЗИЯ (Stable Audio / ACE-Step). GPT-класс (Bark/MusicGen) "
                "не грузится через diffusion-машинерию — им нужен свой рантайм."
            )
        # мягкая проверка что модель именно аудио — иначе предупреждаем, но не рушим
        try:
            _mc = type(getattr(model, "model", model)).__name__.lower()
            _cfg = str(getattr(getattr(model, "model", None), "model_config", "")).lower()
            _is_audio = any(h in _mc or h in _cfg for h in _AUDIO_MODEL_HINTS)
            if not _is_audio:
                print("[XQuant Music] ВНИМАНИЕ: модель загрузилась, но не опознана как "
                      "Stable Audio / ACE-Step — проверь, что это музыкальная диффузия.")
            else:
                print(f"[XQuant Music] опознана аудио-модель ({_mc})")
        except Exception:
            pass
        print(f"[XQuant] музыкальная модель собрана из GGUF ({len(sd)} тензоров)")
        return (model,)


NODE_CLASS_MAPPINGS = {
    "XQuantTernaryLoader": XQuantTernaryLoader,
    "XQuantGGUFLoader": XQuantGGUFLoader,
    "XQuantMusicLoader": XQuantMusicLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "XQuantTernaryLoader": "XQuant Ternary Loader (1.6-bit)",
    "XQuantGGUFLoader": "XQuant GGUF Loader (2-8bit · image/video/audio)",
    "XQuantMusicLoader": "XQuant Music Loader (Stable Audio / ACE-Step)",
}

# imatrix-сборщик (активационная важность) — отдельный модуль
try:
    from .imatrix import (NODE_CLASS_MAPPINGS as _IM_CLS,
                          NODE_DISPLAY_NAME_MAPPINGS as _IM_DN)
    NODE_CLASS_MAPPINGS.update(_IM_CLS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_IM_DN)
except Exception as _e:
    print(f"[XQuant] imatrix-ноды не загрузились: {_e}")
