# -*- coding: utf-8 -*-
"""НАШ GGUF-писатель — с нуля, без внешней gguf-либы. Только numpy + struct.
Пишет валидный GGUF v3, который читает ComfyUI-GGUF. Формат: magic+version+
counts → метадата KV → тензор-инфо → выравнивание → данные тензоров (align 32).

GGML-типы (что нужно): F32=0 F16=1 Q4_0=2 Q2_K=10 Q3_K=11 BF16=30.
Логика dims: ggml-порядок = numpy-shape в ОБРАТНОМ порядке (ne[0]=последняя).
Для квантованных dims = ЛОГИЧЕСКИЕ (элементы), данные = упакованные байты.
"""
import struct, numpy as np

# GGML quant types
class T:
    F32=0; F16=1; Q4_0=2; Q5_0=6; Q8_0=8; Q2_K=10; Q3_K=11; Q6_K=14; BF16=30
# метадата value-types
_U32=4; _STR=8; _U64=10
ALIGN=32
_MAGIC=0x46554747  # "GGUF" LE

def _gstr(s):
    b=s.encode("utf-8"); return struct.pack("<Q",len(b))+b

def _kv_str(k,v): return _gstr(k)+struct.pack("<I",_STR)+_gstr(v)
def _kv_u32(k,v): return _gstr(k)+struct.pack("<I",_U32)+struct.pack("<I",v)

def _pad(nbytes, align=ALIGN):
    r = nbytes % align
    return b"\x00"*((align-r)%align)

def write_gguf(path, arch, tensors):
    """tensors: list of (name, ggml_type, logical_shape_tuple, data_bytes(np.uint8/bytes)).
    logical_shape в numpy-порядке (rows, cols); мы сами развернём в ggml."""
    # --- метадата ---
    meta = b""
    kv = [ _kv_str("general.architecture", arch),
           _kv_u32("general.quantization_version", 2),
           _kv_u32("general.alignment", ALIGN),
           _kv_str("general.name", "xquant") ]
    meta = b"".join(kv)
    n_kv = len(kv)

    # --- тензор-инфо + расчёт оффсетов ---
    infos = b""
    data_blobs = []
    offset = 0
    for name, ttype, shape, data in tensors:
        db = bytes(data) if not isinstance(data, (bytes, bytearray)) else data
        ne = list(reversed([int(x) for x in shape]))  # ggml порядок
        info = _gstr(name) + struct.pack("<I", len(ne))
        for d in ne: info += struct.pack("<Q", d)
        info += struct.pack("<I", ttype) + struct.pack("<Q", offset)
        infos += info
        pad = _pad(len(db))
        data_blobs.append(db + pad)
        offset += len(db) + len(pad)

    header = struct.pack("<I", _MAGIC) + struct.pack("<I", 3) \
           + struct.pack("<Q", len(tensors)) + struct.pack("<Q", n_kv)

    with open(path, "wb") as f:
        f.write(header)
        f.write(meta)
        f.write(infos)
        # выравнивание перед секцией данных
        pos = len(header)+len(meta)+len(infos)
        f.write(_pad(pos))
        for blob in data_blobs:
            f.write(blob)


# ═══════════ ЧТЕНИЕ GGUF (для requantize LLM) — свой ридер ═══════════
# Метадату (вкл. токенайзер) копируем СЫРЫМИ байтами → сохраняется как есть.
_VT_FIXED = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}  # value_type → размер

def _skip_kv_value(f, vt):
    if vt in _VT_FIXED: f.read(_VT_FIXED[vt])
    elif vt == _STR: n=struct.unpack("<Q",f.read(8))[0]; f.read(n)
    elif vt == 9:  # ARRAY: elem_type(u32) + count(u64) + elems
        et=struct.unpack("<I",f.read(4))[0]; cnt=struct.unpack("<Q",f.read(8))[0]
        for _ in range(cnt):
            if et==_STR: n=struct.unpack("<Q",f.read(8))[0]; f.read(n)
            elif et==9: _skip_kv_value(f, 9)
            else: f.read(_VT_FIXED.get(et,4))
    else: raise ValueError(f"unknown KV value_type {vt}")

def read_gguf(path):
    """Вернуть (f, version, raw_meta_bytes, n_kv, tensor_infos, data_start, align).
    tensor_infos: list of (name, dims_tuple_ggml, ggml_type, offset)."""
    f = open(path, "rb")
    magic, ver = struct.unpack("<II", f.read(8))
    if magic != _MAGIC: raise ValueError("не GGUF")
    n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
    meta_start = f.tell()
    align = ALIGN
    for _ in range(n_kv):
        kn = struct.unpack("<Q", f.read(8))[0]; key = f.read(kn).decode("utf-8","replace")
        vt = struct.unpack("<I", f.read(4))[0]
        vpos = f.tell()
        if key == "general.alignment" and vt == _U32:
            align = struct.unpack("<I", f.read(4))[0];
        else:
            _skip_kv_value(f, vt)
    meta_end = f.tell()
    f.seek(meta_start); raw_meta = f.read(meta_end - meta_start); f.seek(meta_end)
    tinfos = []
    for _ in range(n_tensors):
        nn = struct.unpack("<Q", f.read(8))[0]; name = f.read(nn).decode("utf-8","replace")
        nd = struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack(f"<{nd}Q", f.read(8*nd))
        tt = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        tinfos.append((name, dims, tt, off))
    after = f.tell()
    data_start = after + ((align - after % align) % align)
    return f, ver, raw_meta, n_kv, tinfos, data_start, align

def write_gguf_raw(path, raw_meta, n_kv, tensors):
    """Записать GGUF с ГОТОВОЙ метадатой (сырые байты) + новыми тензорами.
    tensors: list of (name, ggml_type, ggml_dims_tuple, data_bytes)."""
    infos = b""; blobs = []; offset = 0
    for name, tt, ne, data in tensors:
        db = bytes(data)
        info = _gstr(name) + struct.pack("<I", len(ne))
        for d in ne: info += struct.pack("<Q", int(d))
        info += struct.pack("<I", tt) + struct.pack("<Q", offset)
        infos += info; pad=_pad(len(db)); blobs.append(db+pad); offset += len(db)+len(pad)
    header = struct.pack("<I",_MAGIC)+struct.pack("<I",3)+struct.pack("<Q",len(tensors))+struct.pack("<Q",n_kv)
    with open(path,"wb") as f:
        f.write(header); f.write(raw_meta); f.write(infos)
        f.write(_pad(len(header)+len(raw_meta)+len(infos)))
        for b in blobs: f.write(b)


# ── дековод source-тензоров GGUF (F16/F32/BF16/Q8_0) для реквантизации ──
def dec_source(raw, ggml_type, n):
    if ggml_type == T.F32: return np.frombuffer(raw, np.float32)[:n].astype(np.float32)
    if ggml_type == T.F16: return np.frombuffer(raw, np.float16)[:n].astype(np.float32)
    if ggml_type == T.BF16:
        u=np.frombuffer(raw, np.uint16).astype(np.uint32); return ((u<<16).view(np.float32))[:n]
    if ggml_type == T.Q8_0:  # блок 34б: fp16 d + 32 int8
        b=np.frombuffer(raw, np.uint8).reshape(-1,34)
        d=b[:,0:2].copy().view(np.float16).astype(np.float32).reshape(-1,1)
        q=b[:,2:34].view(np.int8).astype(np.float32)
        return (d*q).reshape(-1)[:n]
    return None  # прочие кванты — не деководим (пропуск)


# ── кодировка простых типов (замена gguf.quants для F32/F16/BF16) ──
def enc_f32(a): return np.ascontiguousarray(a, np.float32).tobytes()
def enc_f16(a): return np.ascontiguousarray(a, np.float32).astype(np.float16).tobytes()
def enc_bf16(a):
    u = np.ascontiguousarray(a, np.float32).view(np.uint32)
    return ((u >> 16) & 0xFFFF).astype(np.uint16).tobytes()
