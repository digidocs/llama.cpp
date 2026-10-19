import json, sys, urllib.request
prompts = [
  "Write a Python function that parses an ISO 8601 duration string like 'P3DT4H12M' into total seconds, with tests.",
  "Explain how a B-tree insertion works, step by step, including node splits.",
  "Write a C function that computes CRC32 using a lookup table, and explain the table generation.",
  "Summarize the causes and consequences of the 1929 stock market crash in about 300 words.",
  "Refactor this into idiomatic Rust: a function that reads a file line by line, counts word frequencies, and prints the top 10.",
]
tag = sys.argv[1]
tot_n = tot_t = dn = da = 0
for i, p in enumerate(prompts):
    body = {"messages": [{"role": "user", "content": p}], "max_tokens": 700, "seed": 1000 + i,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request("http://127.0.0.1:8011/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    t = r.get("timings", {})
    n, ms = t.get("predicted_n", 0), t.get("predicted_ms", 0)
    tot_n += n; tot_t += ms
    dn += t.get("draft_n", 0) or 0; da += t.get("draft_n_accepted", 0) or 0
    print(f"{tag} p{i}: {n} tok, {n/ms*1000 if ms else 0:.2f} t/s, draft {t.get('draft_n',0)}/{t.get('draft_n_accepted',0)} acc, pp {t.get('prompt_per_second',0):.0f} t/s", flush=True)
print(f"{tag} TOTAL: {tot_n} tok, {tot_n/tot_t*1000:.2f} t/s" + (f", acceptance {da/dn:.2%} ({da}/{dn})" if dn else ""), flush=True)
