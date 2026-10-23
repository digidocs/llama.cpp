# For each BF16 tensor: can every 32-value block (along a row) be stored exactly as e4m3 values times one power-of-two scale?
import glob, re, sys, collections
import numpy as np
sys.path.insert(0, '/home/david/llama.cpp/gguf-py')
from gguf import GGUFReader

def block_ok(b):  # b: uint16 bf16 bits, shape (n, 32)
    exp  = ((b >> 7) & 0xFF).astype(np.int32)
    mant = (b & 0x7F).astype(np.int32)
    zero = (exp == 0) & (mant == 0)
    bad_sub = (exp == 0) & (mant != 0)                     # bf16 subnormals: treat as not representable
    eu = np.where(zero, -1000, exp - 127)
    emax = eu.max(axis=1, keepdims=True)
    ok_any = np.zeros(b.shape[0], dtype=bool)
    for shift in (8, 7):                                   # scale 2^(emax-8), or one step up if the max needs it
        e = eu - (emax - shift)
        low4, low5, low6 = (mant & 0xF) == 0, (mant & 0x1F) == 0, (mant & 0x3F) == 0
        normal = (e >= -6) & (e <= 8) & low4 & ~((e == 8) & ((mant >> 4) == 7))
        sub    = ((e == -7) & low5) | ((e == -8) & low6) | ((e == -9) & (mant == 0))
        good = zero | (~bad_sub & (normal | sub))
        ok_any |= good.all(axis=1)
    return ok_any

files = sorted(glob.glob('/home/david/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q8_K_XL/*.gguf'))
res = collections.defaultdict(lambda: [0, 0, 0, 0])  # tensors, tensors fully exact, blocks, bad blocks
for f in files:
    for t in GGUFReader(f).tensors:
        if t.tensor_type.name != 'BF16':
            continue
        kind = re.sub(r'blk\.\d+\.', 'blk.N.', t.name)
        flat = np.asarray(t.data).view(np.uint16).reshape(-1)
        assert flat.size % 32 == 0
        bad = 0
        step = 32 * (1 << 20)
        for i in range(0, flat.size, step):
            blk = flat[i:i + step].reshape(-1, 32)
            bad += int((~block_ok(blk)).sum())
        r = res[kind]; r[0] += 1; r[1] += bad == 0; r[2] += flat.size // 32; r[3] += bad
        print(f"{t.name:45s} bad blocks {bad}", flush=True)
print("\nSUMMARY")
for k, (n, nexact, nb, nbad) in sorted(res.items()):
    print(f"{k:45s} {nexact:3d}/{n:<3d} tensors exact, {nbad}/{nb} bad blocks")
