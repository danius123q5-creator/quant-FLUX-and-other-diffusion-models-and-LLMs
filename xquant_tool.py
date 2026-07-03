# -*- coding: utf-8 -*-
"""XQUANT — ужиматель моделей. Кидаешь fp16/bf16 .safetensors → отдаёт ужатый GGUF.
Использует НАШЕ ядро (xquant.py) для Q4_0/Q3_K/Q2_K + gguf для Q5_0/Q8_0.
Авто-детект архитектуры (flux/sd3/qwen/wan/sdxl...) через city96 convert.py.

Запуск:  python xquant_tool.py <model.safetensors> [Q4_0|Q3_K|Q2_K|Q5_0|Q8_0]
"""
import os, sys, gguf, torch, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE,"ComfyUI","custom_nodes","ComfyUI-GGUF","tools"))
import xquant as xq
import convert
from tqdm import tqdm

# --- Qwen arch (загрузчик знает, конвертеру добавляем) ---
class ModelQwenImage(convert.ModelTemplate):
    arch="qwen_image"
    keys_detect=[("transformer_blocks.0.attn.add_q_proj.weight",
                  "time_text_embed.timestep_embedder.linear_1.weight")]
convert.arch_list.insert(0, ModelQwenImage)

OUR={  # наши ядра (энкодер, размер блока, GGML-тип)
 "Q4_0":(xq.our_quantize_q4_0, gguf.GGMLQuantizationType.Q4_0, 32),
 "Q3_K":(xq.our_quantize_q3k,  gguf.GGMLQuantizationType.Q3_K, 256),
 "Q2_K":(xq.our_quantize_q2k,  gguf.GGMLQuantizationType.Q2_K, 256),
}

def handle(writer, sd, arch, QNAME):
    nq=nf=0
    for key,data in tqdm(sd.items()):
        od=data.dtype
        if any(x in key for x in arch.keys_ignore): continue
        if od==torch.bfloat16: data=data.to(torch.float32).numpy()
        elif "float8" in str(od): data=data.to(torch.float16).numpy()
        else: data=data.numpy()
        nd=len(data.shape); npm=int(np.prod(data.shape))
        qt = gguf.GGMLQuantizationType.BF16 if od==torch.bfloat16 else gguf.GGMLQuantizationType.F16
        if od in (torch.float32,torch.bfloat16) or "float8" in str(od):
            if nd==1 or npm<=convert.QUANTIZATION_THRESHOLD or any(x in key for x in arch.keys_hiprec):
                qt=gguf.GGMLQuantizationType.F32
        # УНИВЕРСАЛЬНАЯ защита критических слоёв (все архитектуры): вход/выход/
        # эмбеды/нормы → bf16. Иначе рвётся связь с VAE = цветной шум. См. xquant.is_critical.
        _crit = xq.is_critical(key)
        # большие 2D → наш квант
        blkdiv = OUR.get(QNAME,(None,None,32))[2]
        if (not _crit) and nd==2 and npm>convert.QUANTIZATION_THRESHOLD and qt in (gguf.GGMLQuantizationType.BF16,gguf.GGMLQuantizationType.F16) and data.shape[1]%blkdiv==0:
            try:
                if QNAME in OUR:
                    fn,ggtype,bs=OUR[QNAME]
                    q=fn(data).reshape(data.shape[0],-1); qt=ggtype
                else:  # Q5_0/Q8_0 через gguf
                    qt=getattr(gguf.GGMLQuantizationType,QNAME); q=gguf.quants.quantize(data,qt)
                nq+=1
            except Exception:
                qt=gguf.GGMLQuantizationType.F16; q=gguf.quants.quantize(data,qt); nf+=1
        else:
            q=gguf.quants.quantize(data,qt)
        writer.add_tensor(key,q,raw_dtype=qt)
    tqdm.write(f"ужато нашим ядром: {nq} | откат F16: {nf}")

def main():
    if len(sys.argv)<2:
        print("USAGE: xquant_tool.py <model.safetensors> [Q4_0|Q3_K|Q2_K|Q5_0|Q8_0]"); return
    SRC=sys.argv[1].strip('"')
    QNAME=(sys.argv[2] if len(sys.argv)>2 else os.environ.get("QTYPE","Q4_0")).upper()
    if not os.path.isfile(SRC): print("НЕТ ФАЙЛА:",SRC); return
    base=os.path.splitext(SRC)[0]
    DST=f"{base}-{QNAME}.gguf"
    print(f"XQUANT: {os.path.basename(SRC)} → {QNAME}")
    convert.handle_tensors=lambda w,sd,a: handle(w,sd,a,QNAME)
    sz0=os.path.getsize(SRC)/1e9
    convert.convert_file(SRC,DST,interact=False,overwrite=True)
    sz1=os.path.getsize(DST)/1e9
    print(f"\nГОТОВО: {DST}\n  {sz0:.1f}ГБ → {sz1:.1f}ГБ  (×{sz0/sz1:.1f} сжатие)")

if __name__=="__main__": main()
