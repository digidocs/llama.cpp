# GPU start latency of CUDA graph launches into an idle GPU during the last request of an nsys capture
# (record with --trace=cuda --cuda-graph-trace=graph; usage: launch_lat.py <export.sqlite> [n_tokens])
import sqlite3, sys, bisect
c = sqlite3.connect(sys.argv[1]); c.text_factory = lambda b: b.decode(errors='replace')
ntok = int(sys.argv[2]) if len(sys.argv) > 2 else 300
names = dict(c.execute("select id, value from StringIds"))
tabs = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
g = c.execute("select start, end, correlationId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE").fetchall() if 'CUPTI_ACTIVITY_KIND_GRAPH_TRACE' in tabs else []
k = c.execute("select start, end, correlationId from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
act = sorted(g + k)
i = len(act) - 1
while i > 0 and act[i][0] - act[i-1][1] < 2e9: i -= 1
act = act[i:]
t0, t1 = act[0][0], act[-1][1]
first = {}
for a, b, cid in act: first.setdefault(cid, a)
ends = sorted(b for a, b, cid in act)
api = c.execute("select start, end, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME where start between ? and ? order by start", (t0, t1)).fetchall()
res = []
for a, b, cid, n in api:
    if not names[n].startswith('cudaGraphLaunch') or cid not in first: continue
    f = first[cid]
    j = bisect.bisect_left(ends, a) - 1
    jj = bisect.bisect_left(ends, f) - 1
    if (ends[j] if j >= 0 else 0) <= a and ends[jj] <= a:
        res.append(((f - a)/1e3, (b - a)/1e3))
res.sort(); n = len(res)
nl = sum(1 for a, b, cid, nm in api if names[nm].startswith('cudaGraphLaunch'))
print(f"graph launches {nl/ntok:.1f}/token; into an idle GPU {n/ntok:.1f}/token: API start -> GPU start median {res[n//2][0]:.1f} us, p90 {res[int(n*.9)][0]:.1f}, total {sum(x[0] for x in res)/ntok/1e3:.2f} ms/token")
