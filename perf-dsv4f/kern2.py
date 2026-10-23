import sqlite3, sys, collections, re
c = sqlite3.connect(sys.argv[1]); c.text_factory = lambda b: b.decode(errors='replace')
names = dict(c.execute("select id, value from StringIds"))
rows = c.execute("select start, end, demangledName, gridX, gridY, blockX from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
i = len(rows) - 1
while i > 0 and rows[i][0] - rows[i-1][1] < 2e9: i -= 1
rows = rows[i:]
tot = collections.defaultdict(lambda: [0, 0])
for a, b, n, gx, gy, bx in rows:
    k = names[n]
    if 'mul_mat_vec_f' not in k and 'mul_mat_f' not in k: continue
    t = re.sub(r'^void ', '', k)[:110]
    key = (t, gx, gy, bx)
    tot[key][0] += 1; tot[key][1] += b - a
for (t, gx, gy, bx), (n, d) in sorted(tot.items(), key=lambda x: -x[1][1])[:14]:
    print(f"{d/1e6:8.1f} ms {n:6d} {d/n/1e3:7.1f} us grid {gx}x{gy} blk {bx}  {t}")
