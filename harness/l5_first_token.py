#!/usr/bin/env python3
"""Batched-prefill check (perf reset B1): max_tokens=1, so each answer is the prefill's greedy token.

Ten short prompts under three salts are answered solo (one at a time), then as a burst (all at once:
the batched prefill). A batching bug (wrong rows, positions or cache) flips many first tokens. Batch-shape
numerics (the coordinator GEMMs are not batch-invariant) flip at most a rare near-tie.
PASS: at most 2 of 30 first tokens differ.

usage: l5_first_token.py [http://coordinator:8100/v1]
"""
import json, sys, time, threading, urllib.request
base = sys.argv[1] if len(sys.argv) > 1 else "http://coordinator:8100/v1"
prompts = ["Write a Python function that checks whether a string is a palindrome.",
 "Return a JSON object with keys name, age and city for a fictional person.",
 "Tell a short story about a lighthouse keeper who finds a message in a bottle.",
 "Explain how photosynthesis works in plain language.",
 "What is 17 times 23? Show the working.",
 "A farmer has 3 fields of 12 rows with 8 plants each. How many plants in total? Reason step by step.",
 "Summarise the causes of the French Revolution in five bullet points.",
 "Produce a markdown table of three planets with their radius and number of moons.",
 "Format the following as a numbered list: apples, pears, plums, cherries.",
 "Count from 1 to 40 separated by commas."]
def ask(text, out, i, n=1):
    body = {"model": "mimo-v2.6-flash", "messages": [{"role": "user", "content": text}], "max_tokens": n, "temperature": 0}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    out[i] = json.loads(urllib.request.urlopen(req, timeout=600).read())["choices"][0]["message"]["content"]
diff = total = 0
for s in range(3):
    salt = f"[first {s} {time.time_ns()}] "
    texts = [salt + p for p in prompts]
    solo = [None] * len(texts)
    for i, t in enumerate(texts): ask(t, solo, i)
    burst = [None] * len(texts)
    th = [threading.Thread(target=ask, args=(t, burst, i)) for i, t in enumerate(texts)]
    for x in th: x.start()
    for x in th: x.join()
    for i in range(len(texts)):
        total += 1
        if solo[i] != burst[i]:
            diff += 1
            print(f"salt {s} prompt {i}: solo {solo[i]!r} burst {burst[i]!r}")
ok = diff <= 2
print(f"RESULT: {'PASS' if ok else 'FAIL'} L5 first token (burst vs solo differ in {diff}/{total})")
sys.exit(0 if ok else 1)
