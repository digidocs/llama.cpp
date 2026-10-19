import sqlite3,sys
c=sqlite3.connect(sys.argv[1]); per=int(sys.argv[2]); ntok=int(sys.argv[3])
g=c.execute("select start,end from CUPTI_ACTIVITY_KIND_GRAPH_TRACE order by start").fetchall()
w0=g[-per*ntok][0]; w1=g[-1][1]
iv=[]
for tab in ['CUPTI_ACTIVITY_KIND_GRAPH_TRACE','CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET']:
    rows=c.execute(f"select start,end from {tab} where start>=? and end<=?",(w0,w1)).fetchall()
    s=sum(e-b for b,e in rows); print(f"{tab:40s} n/tok={len(rows)/ntok:7.1f} ms/tok={s/ntok/1e6:7.3f}")
    iv+=rows
iv.sort(); busy=0; cs,ce=iv[0]
for b,e in iv[1:]:
    if b>ce: busy+=ce-cs; cs,ce=b,e
    else: ce=max(ce,e)
busy+=ce-cs
T=(w1-w0)/ntok
print(f"token time {T/1e6:.2f} ms, GPU busy {busy/ntok/1e6:.2f} ms, GPU idle {(T-busy/ntok)/1e6:.2f} ms")
m=c.execute("select copyKind,count(*),sum(bytes),sum(end-start) from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and end<=? group by copyKind",(w0,w1)).fetchall()
for k,n,b,t in m: print("memcpy kind",k,"n/tok",n/ntok,"bytes/tok",b/ntok,"ms/tok",t/ntok/1e6)
# graph durations distribution
d=sorted(e-b for b,e in g[-per*ntok:])
print("graph dur us: min %.0f med %.0f max %.0f sum/tok %.2f ms"%(d[0]/1e3,d[len(d)//2]/1e3,d[-1]/1e3,sum(d)/ntok/1e6))
