# near-full-context check: one prompt of N_CHARS chars of llama.cpp sources (~3.6 chars/token), 64 generated tokens
import json, sys, urllib.request, os, glob
tag, n_chars = sys.argv[1], int(sys.argv[2])
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
src = ""
for f in sorted(glob.glob(os.path.join(root, "src", "*.cpp")) + glob.glob(os.path.join(root, "ggml", "src", "ggml-cuda", "*.cu"))):
    src += open(f, errors="replace").read()
    if len(src) >= n_chars:
        break
body = {"messages": [{"role": "user", "content": src[:n_chars] + "\n\nSummarize what this code does in two sentences."}],
        "max_tokens": 64, "cache_prompt": False, "chat_template_kwargs": {"reasoning_effort": "low"}}
req = urllib.request.Request("http://127.0.0.1:8011/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=7200))
t = r["timings"]
print(f"{tag} deep: {t.get('prompt_n')} tok, pp {t.get('prompt_per_second', 0):.1f} t/s, tg {t.get('predicted_per_second', 0):.2f} t/s", flush=True)
