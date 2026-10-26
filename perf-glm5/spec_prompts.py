import hashlib, json, os, sys, urllib.request
prompts = [
  "Write a Python function that parses an ISO 8601 duration string like 'P3DT4H12M' into total seconds, with tests.",
  "Explain how a B-tree insertion works, step by step, including node splits.",
  "Write a C function that computes CRC32 using a lookup table, and explain the table generation.",
  "Summarize the causes and consequences of the 1929 stock market crash in about 300 words.",
  "Refactor this into idiomatic Rust: a function that reads a file line by line, counts word frequencies, and prints the top 10.",
]
tag = sys.argv[1]
# reasoning_effort low (GLM has no off switch), GREEDY=1 uses temperature 0, MAX_TOKENS overrides the per-prompt token budget
greedy = os.environ.get("GREEDY") == "1"
max_tokens = int(os.environ.get("MAX_TOKENS") or 700)
tot_n = tot_t = dn = da = 0
for i, p in enumerate(prompts):
    body = {"messages": [{"role": "user", "content": p}], "max_tokens": max_tokens, "seed": 1000 + i,
            "chat_template_kwargs": {"reasoning_effort": "low"}}
    if greedy:
        body["temperature"] = 0.0
    req = urllib.request.Request("http://127.0.0.1:8011/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    t = r.get("timings", {})
    n, ms = t.get("predicted_n", 0), t.get("predicted_ms", 0)
    tot_n += n; tot_t += ms
    dn += t.get("draft_n", 0) or 0; da += t.get("draft_n_accepted", 0) or 0
    m = r["choices"][0]["message"]
    md5 = hashlib.md5(((m.get("reasoning_content") or "") + "\n----\n" + (m.get("content") or "")).encode()).hexdigest()[:12]
    print(f"{tag} p{i}: {n} tok, {n/ms*1000 if ms else 0:.2f} t/s, draft {t.get('draft_n',0)}/{t.get('draft_n_accepted',0)} acc, pp {t.get('prompt_per_second',0):.0f} t/s, md5 {md5}", flush=True)
print(f"{tag} TOTAL: {tot_n} tok, {tot_n/tot_t*1000:.2f} t/s" + (f", acceptance {da/dn:.2%} ({da}/{dn})" if dn else ""), flush=True)
