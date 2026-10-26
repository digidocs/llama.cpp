# decode timeline from an nsys sqlite export (llama-bench -p 0 -n N, --cuda-graph-trace=node)
# usage: nsys_tg.py <trace.sqlite> <n_tokens> <tg t/s>: the window is the last n_tokens / rate seconds of the trace; GPU busy/idle per token, idle gaps, top kernels, memcpy
import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1])
n_tok = int(sys.argv[2]) if len(sys.argv) > 2 else 32
names = dict(db.execute("select id, value from StringIds"))
ev = [(s, e, names.get(n, "?"), "k") for s, e, n in db.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL")]
ev += [(s, e, {1: "H2D", 2: "D2H", 8: "D2D"}.get(k, f"memcpy{k}"), "m", b) for s, e, k, b in db.execute("select start, end, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY")]
ev.sort()
t_end = ev[-1][1]
busy, last = [], None
for s, e, *_ in ev:
    if last and s <= last[1]:
        last[1] = max(last[1], e)
    else:
        last = [s, e]; busy.append(last)
rate = float(sys.argv[3]) if len(sys.argv) > 3 else None
wall = n_tok / rate * 1e9 if rate else (t_end - ev[0][0])
w0 = t_end - wall
sel = [x for x in ev if x[0] >= w0]
gpu_busy = sum(min(e, t_end) - max(s, w0) for s, e in busy if e > w0)
print(f"window {wall/1e6:.1f} ms = {n_tok} tokens, {wall/1e6/n_tok:.2f} ms/token; GPU busy {gpu_busy/1e6/n_tok:.2f} ms/token, idle {(wall-gpu_busy)/1e6/n_tok:.2f}")
idle = [b2[0] - b1[1] for b1, b2 in zip(busy, busy[1:]) if b1[0] >= w0]
for lo, hi in [(0, 20e3), (20e3, 200e3), (200e3, 2e6), (2e6, 1e12)]:
    g = [x for x in idle if lo <= x < hi]
    print(f"  idle gaps {lo/1e3:>6.0f}-{hi/1e3:<6.0f} us: {len(g)/n_tok:6.1f} per token, {sum(g)/1e6/n_tok:6.2f} ms/token")
k = collections.defaultdict(lambda: [0, 0])
for x in sel:
    k[x[2]][0] += x[1] - x[0]; k[x[2]][1] += 1
print("top GPU time per token:")
for name, (t, c) in sorted(k.items(), key=lambda kv: -kv[1][0])[:14]:
    print(f"  {t/1e6/n_tok:7.3f} ms  {c/n_tok:7.1f}x  {name[:70]}")
mb = collections.defaultdict(int)
for x in sel:
    if x[3] == "m":
        mb[x[2]] += x[4]
print("memcpy per token:", ", ".join(f"{k} {v/n_tok/1e6:.2f} MB" for k, v in mb.items()))
