# -*- coding: utf-8 -*-
"""Собрать imatrix (activation-aware importance) для FLUX — БЕЗ ComfyUI-ноды.

Гоняет несколько прогонов трансформера flux на реальных текст-эмбеддингах,
хуками на каждом Linear копит sum(act^2) по входным каналам и пишет
<model>.imatrix.npy — dict {имя_слоя: np.array[in_features]}.

Этот .npy потом скармливается XQuant.exe (поле 🎯 imatrix) или через
переменную XQUANT_IMATRIX — и Q2_K/Q3_K бережёт активационные выбросы
(AWQ-класс), особенно на attention, где data-free методы бессильны.

Запуск (нужен CUDA-torch + diffusers, напр. python_embeded от ComfyUI):
    python collect_imatrix.py                       # FLUX.1-dev из HF-кэша
    python collect_imatrix.py --model <repo|path> --out my.imatrix.npy --gens 3

⚠️ cpu_offload на новых torch (cu13x) может segfault'ить в связке с хуками —
поэтому грузим текст-энкодеры → эмбеддинги → выгружаем, на GPU только
трансформер (~24ГБ, влезает в 24-32ГБ карту). Планировщик обходим (imatrix
важны активации, а не корректный денойз) — гоняем фикс. таймстепы.
"""
import argparse, time, sys
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="black-forest-labs/FLUX.1-dev",
                    help="HF-repo или локальный путь к flux (diffusers-формат)")
    ap.add_argument("--out", default="flux1-dev.imatrix.npy")
    ap.add_argument("--gens", type=int, default=3, help="сколько прогонов (промптов)")
    ap.add_argument("--res", type=int, default=512)
    args = ap.parse_args()

    from diffusers import FluxPipeline
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("[imatrix] ВНИМАНИЕ: CUDA не найдена — на CPU сбор нереально медленный.")
    torch.manual_seed(0)

    print(f"[imatrix] загрузка {args.model} (bf16)...", flush=True)
    pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)

    prompts = [
        "a cinematic portrait of a woman in a rain-soaked neon city at night",
        "an epic fantasy castle on a cliff at sunrise, dramatic clouds, highly detailed",
        "a cozy cat on a wooden table by a window, soft daylight, photorealistic",
        "a rugged man with a beard in a workshop, warm tungsten light, 50mm",
        "a vast alien desert under two moons, sci-fi concept art, wide shot",
    ][: max(1, args.gens)]

    # 1) эмбеддинги на GPU → выгружаем текст-энкодеры (экономим VRAM, избегаем offload)
    print("[imatrix] энкодю промпты...", flush=True)
    pipe.text_encoder.to(dev); pipe.text_encoder_2.to(dev)
    embeds = []
    with torch.no_grad():
        for p in prompts:
            pe, ppe, tids = pipe.encode_prompt(prompt=p, prompt_2=p, device=dev,
                                               num_images_per_prompt=1, max_sequence_length=256)
            embeds.append((pe, ppe, tids))
    pipe.text_encoder.to("cpu"); pipe.text_encoder_2.to("cpu")
    if dev == "cuda":
        torch.cuda.empty_cache()

    # 2) трансформер на GPU + хуки на все Linear (накапливаем на устройстве x)
    tr = pipe.transformer.to(dev)
    acc, cnt, hooks = {}, {}, []

    def mk(name):
        def hook(mod, inp):
            x = inp[0]
            if x is None:
                return
            with torch.no_grad():
                xf = x.detach().reshape(-1, x.shape[-1]).float()
                s = (xf * xf).sum(0)
                if name in acc:
                    acc[name] += s; cnt[name] += xf.shape[0]
                else:
                    acc[name] = s; cnt[name] = xf.shape[0]
        return hook

    for name, mod in tr.named_modules():
        if isinstance(mod, torch.nn.Linear):
            hooks.append(mod.register_forward_pre_hook(mk(name)))
    print(f"[imatrix] хуков на Linear: {len(hooks)}", flush=True)

    def save():
        imat = {n: (acc[n] / max(cnt[n], 1)).detach().cpu().numpy().astype(np.float32) for n in acc}
        np.save(args.out, imat, allow_pickle=True)
        return imat

    # 3) фиксированные таймстепы (планировщик flux требует mu; для imatrix не нужен)
    H = W = args.res
    gembeds = tr.config.guidance_embeds
    timesteps = [1000., 750., 500., 250.]
    t0 = time.time(); done = 0
    for i, (pe, ppe, tids) in enumerate(embeds, 1):
        print(f"[imatrix] прогон {i}/{len(embeds)} (+{time.time()-t0:.0f}s)", flush=True)
        gen = torch.Generator("cpu").manual_seed(i)
        lat, img_ids = pipe.prepare_latents(1, tr.config.in_channels // 4, H, W,
                                            torch.bfloat16, dev, gen)
        guid = torch.full([1], 3.5, device=dev, dtype=torch.bfloat16) if gembeds else None
        for tval in timesteps:
            ts = torch.full([1], tval, device=dev, dtype=torch.bfloat16)
            with torch.no_grad():
                noise = tr(hidden_states=lat, timestep=ts / 1000, guidance=guid,
                           pooled_projections=ppe, encoder_hidden_states=pe,
                           txt_ids=tids, img_ids=img_ids, return_dict=False)[0]
                lat = lat - 0.2 * noise.to(lat.dtype)
        done += 1
        save()

    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    if done:
        imat = save()
        # маленькая сводка выбросов — чем выше p99/median, тем важнее imatrix для слоя
        peaks = sorted(((float(np.percentile(v, 99) / max(np.median(v), 1e-9)), n)
                        for n, v in imat.items()), reverse=True)[:3]
        print(f"[imatrix] ГОТОВО: {args.out}  ({done} прогонов, {len(imat)} слоёв, {time.time()-t0:.0f}с)", flush=True)
        print("[imatrix] самые «выбросные» слои (p99/median):", flush=True)
        for r, n in peaks:
            print(f"    {n}: {r:.0f}x", flush=True)
    else:
        print("[imatrix] 0 прогонов — что-то пошло не так", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
