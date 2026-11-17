import sqlite3, sys, collections
c = sqlite3.connect(sys.argv[1])
names = dict(c.execute("select id, value from StringIds"))
ev = []
for s, e, n, tid in c.execute("select start, end, nameId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"):
    nm = names[n]
    if nm.startswith(("cudaGraphLaunch", "cudaMemcpyAsync", "cudaStreamSynchronize", "cudaEventSynchronize", "cudaLaunchKernel", "cudaEventRecord", "cudaStreamWaitEvent", "cudaGraphExecUpdate")):
        ev.append((s, e, "API", nm.split("_")[0], tid))
for s, e in c.execute("select start, end from CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
    ev.append((s, e, "GPU", "graph", 0))
for s, e, k, b in c.execute("select start, end, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY"):
    ev.append((s, e, "GPU", {1: "H2D", 2: "D2H", 8: "D2D"}.get(k, str(k)) + f"[{b}]", 0))
for s, e in c.execute("select start, end from CUPTI_ACTIVITY_KIND_KERNEL") if 0 else []:
    pass
ev.sort()
tids = collections.Counter(t for *_, t in ev if t)
main = tids.most_common(1)[0][0]
# GPU busy fraction and gaps
gpu = sorted((s, e) for s, e, k, *_ in ev if k == "GPU")
t0, t1 = gpu[0][0], gpu[-1][1]
busy, last = 0, t0
for s, e in gpu:
    if e > last:
        busy += e - max(s, last); last = e
print(f"window {(t1-t0)/1e9:.2f} s, GPU busy (graphs+copies) {100*busy/(t1-t0):.1f}%")
# print a slice of the timeline around the middle
mid = len(ev) // 2
start = next(i for i in range(mid, len(ev)) if ev[i][3] == "cudaGraphLaunch")
base = ev[start][0]
for s, e, k, nm, tid in ev[start:start + int(sys.argv[2]) if len(sys.argv) > 2 else start + 60]:
    if k == "API" and tid != main:
        continue
    print(f"{(s-base)/1e3:9.1f} {(e-s)/1e3:8.1f} us  {k} {nm}")
