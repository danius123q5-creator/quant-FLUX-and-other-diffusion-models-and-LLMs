# -*- coding: utf-8 -*-
"""ComfyUI-ноды сбора imatrix (активационной важности) для Жмателя.

Идея: во время ОБЫЧНОЙ генерации снимаем вход каждого Linear диффузионной модели
и копим sum(act²) по входным каналам. Это и есть imatrix — важность по активациям,
которую XQuant кормит в Q2_K/Q3_K (make_qkx2) → 2/3-бит держат качество при сильном
сжатии (как AWQ). Собирается на GPU, где модель уже крутится — второй torch не нужен.

Пайплайн:
  [загрузчик модели] → XQuantImatrixCapture → KSampler → (LATENT) → XQuantImatrixSave
Capture цепляет хуки и обнуляет накопитель; гоняешь 1-3 генерации на разных промптах;
Save пишет <name>.npy (dict{ключ_веса: 1-D float}) — его указываешь в XQuant.exe (поле
🎯 imatrix) или env XQUANT_IMATRIX. 2026-07-16.
"""
import os
import torch
import numpy as np

# Глобальный накопитель: {weight_key: torch.Tensor[in_features]} (sum act²) + счётчики.
_ACC = {}
_HOOKED = {"model_id": None, "handles": []}
_STEPS = {"n": 0}


def _diffusion_module(model):
    """Достать реальный nn.Module диффузионной сети из ComfyUI-обёртки MODEL."""
    m = model
    for attr in ("model", "diffusion_model"):
        m = getattr(m, attr, m)
    return m


def _mk_hook(key):
    def hook(module, inp):
        try:
            x = inp[0]
            if not torch.is_tensor(x):
                return
            # x: [..., in_features]; копим sum(x²) по всем токенам/батчу → [in_features]
            xf = x.detach().to(torch.float32)
            s = (xf * xf).reshape(-1, xf.shape[-1]).sum(dim=0).cpu()
            if key in _ACC:
                _ACC[key] += s
            else:
                _ACC[key] = s
        except Exception:
            pass
    return hook


def _attach(model):
    dm = _diffusion_module(model)
    mid = id(dm)
    # снять старые хуки если модель сменилась
    for h in _HOOKED["handles"]:
        try: h.remove()
        except Exception: pass
    _HOOKED["handles"] = []
    n = 0
    for name, mod in dm.named_modules():
        if isinstance(mod, torch.nn.Linear):
            key = f"{name}.weight" if name else "weight"
            _HOOKED["handles"].append(mod.register_forward_pre_hook(_mk_hook(key)))
            n += 1
    _HOOKED["model_id"] = mid
    return n


class XQuantImatrixCapture:
    """Ставится МЕЖДУ загрузчиком модели и KSampler. Цепляет хуки на все Linear и
    (опц.) обнуляет накопитель. Пропускает MODEL насквозь."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "reset": ("BOOLEAN", {"default": True}),
        }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "capture"
    CATEGORY = "Жматель"

    def capture(self, model, reset):
        if reset:
            _ACC.clear(); _STEPS["n"] = 0
        n = _attach(model)
        print(f"[XQuant imatrix] хуки на {n} Linear-слоёв; накопитель "
              f"{'обнулён' if reset else 'дополняется'}. Гоняй генерацию.")
        return (model,)


class XQuantImatrixSave:
    """Ставится ПОСЛЕ KSampler (вход latent — чтобы выполниться после сэмплинга).
    Пишет накопленный imatrix в .npy. latent пропускает насквозь."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT",),
            "filename": ("STRING", {"default": "flux1-dev.imatrix.npy"}),
        }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "save"
    CATEGORY = "Жматель"
    OUTPUT_NODE = True

    def save(self, latent, filename):
        if not _ACC:
            print("[XQuant imatrix] ПУСТО — сначала поставь XQuantImatrixCapture перед "
                  "KSampler и прогони генерацию.")
            return (latent,)
        out = {k: v.numpy().astype(np.float32) for k, v in _ACC.items()}
        # путь: если относительный — в папку ComfyUI output
        path = filename
        if not os.path.isabs(path):
            try:
                import folder_paths
                path = os.path.join(folder_paths.get_output_directory(), filename)
            except Exception:
                path = os.path.abspath(filename)
        np.save(path, np.array(out, dtype=object), allow_pickle=True)
        tot = sum(int(v.size) for v in out.values())
        print(f"[XQuant imatrix] сохранён: {path}  ({len(out)} слоёв, {tot} каналов). "
              f"Укажи его в XQuant.exe (🎯 imatrix) или XQUANT_IMATRIX=<путь>.")
        return (latent,)


NODE_CLASS_MAPPINGS = {
    "XQuantImatrixCapture": XQuantImatrixCapture,
    "XQuantImatrixSave": XQuantImatrixSave,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "XQuantImatrixCapture": "XQuant imatrix: Capture (перед KSampler)",
    "XQuantImatrixSave": "XQuant imatrix: Save (после KSampler)",
}
