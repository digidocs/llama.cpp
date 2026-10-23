# per-kernel GPU time during the decode request (kernels after the last idle gap > 2 s), top entries
import sqlite3, sys, collections
c = sqlite3.connect(sys.argv[1]); c.text_factory = lambda b: b.decode(errors='replace')
names = dict(c.execute("select id, value from StringIds"))
rows = c.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
i = len(rows) - 1
while i > 0 and rows[i][0] - rows[i-1][1] < 2e9: i -= 1
rows = rows[i:]
span = (rows[-1][1] - rows[0][0]) / 1e9
tot = collections.defaultdict(lambda: [0, 0])
for a, b, n in rows:
    k = names[n]; tot[k][0] += 1; tot[k][1] += b - a
busy = sum(v[1] for v in tot.values())
print(f"window {span:.2f} s, GPU busy {busy/1e9:.2f} s")
for k, (n, t) in sorted(tot.items(), key=lambda x: -x[1][1])[:int(sys.argv[2]) if len(sys.argv) > 2 else 12]:
    print(f"{t/1e6:9.1f} ms {n:7d}  {k[:70]}")
