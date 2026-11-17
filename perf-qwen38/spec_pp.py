# long-prompt pp check for spec_run.sh: ~8K tokens of source code, 32 generated tokens, prompt cache off
import json, sys, urllib.request, os
tag = sys.argv[1]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ggml", "src", "ggml-cpu", "ggml-cpu.c")).read()[:30000]
for i in range(2):
    body = {"messages": [{"role": "user", "content": f"Run {i}. Summarize what this file does in 3 bullet points:\n\n```c\n{src}\n```"}],
            "max_tokens": 32, "cache_prompt": False, "chat_template_kwargs": {"reasoning_effort": "low"}}
    req = urllib.request.Request("http://127.0.0.1:8011/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = json.load(urllib.request.urlopen(req, timeout=3600)).get("timings", {})
    print(f"{tag} pp run {i}: {t.get('prompt_n')} tok, {t.get('prompt_per_second', 0):.1f} t/s", flush=True)
