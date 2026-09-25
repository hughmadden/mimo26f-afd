#!/usr/bin/env python3
"""G0 identity for an L5 cell (builder): what is running, where, from which bytes. Read-only.

Collects, over ssh from the dev host:
- coordinator: the coordinator process (argv, exe sha256, the named env vars only), the staged CUDA runtime libraries (sha256),
  nvidia-smi and MemAvailable;
- spark1..4: each rank daemon (argv, exe sha256), its slice manifest sha256, its readback line, nvidia-smi and MemAvailable;
- dev host: the repo HEAD.
Never dumps a whole environment: only the variables named in ENV_KEYS are read.
usage: l5_identity.py OUT.json
"""
from __future__ import annotations

import datetime
import json
import subprocess
import sys
from zoneinfo import ZoneInfo

ENV_KEYS = ("LD_LIBRARY_PATH", "MIMO26_SPARK_ADDRS", "MIMO26F_CUDA_ARCH")
RT_DIR = "/var/tmp/mimo26f-attn/cuda-12.8-rt"
SLICES = "/var/tmp/mimo26f-kernel/slices"


def ssh(host, script, timeout=60):
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, "bash", "-s"], input=script,
                       capture_output=True, text=True, timeout=timeout)
    return p.stdout.strip()


def proc_probe(name):
    keys = " ".join(ENV_KEYS)
    return f"""
p=$(for q in $(pgrep -f {name}); do case "$(readlink /proc/$q/exe 2>/dev/null)" in */{name}|*/{name}-*) echo $q;; esac; done | head -1)  # by exe path (also {name}-<tag> builds, e.g. mimo26-spark-b1): comm is truncated to 15 chars
if [ -z "$p" ]; then echo "proc: none"; else
  echo "proc: $p"
  echo "argv: $(tr '\\0' ' ' < /proc/$p/cmdline)"
  echo "exe_sha256: $(sha256sum /proc/$p/exe 2>/dev/null | cut -d' ' -f1)"
  echo "etime: $(ps -o etime= -p $p | tr -d ' ')"
  for k in {keys}; do v=$(tr '\\0' '\\n' < /proc/$p/environ 2>/dev/null | grep "^$k=" | cut -d= -f2-); [ -n "$v" ] && echo "env $k: $v"; done
fi
echo "mem_available_kb: $(awk '/MemAvailable/{{print $2}}' /proc/meminfo)"
echo "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,clocks.sm --format=csv,noheader 2>/dev/null)"
"""


def parse(text):
    out = {}
    for line in text.splitlines():
        k, _, v = line.partition(": ")
        if k.startswith("env "):
            out.setdefault("env", {})[k[4:]] = v
        elif k:
            out[k] = v
    return out


def main():
    rec = {"collected": datetime.datetime.now(ZoneInfo("Australia/Sydney")).strftime("%Y-%m-%d %H:%M:%S %Z"),
           "repo_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
    coordinator = parse(ssh("coordinator", proc_probe("mimo26-coordinator") +
                      f'for f in {RT_DIR}/*.so*; do [ -f "$f" ] && echo "rt $(basename $f): $(sha256sum $f | cut -d" " -f1)"; done\n'))
    rec["coordinator"] = coordinator
    for r in range(4):
        host = f"spark{r + 1}"
        rec[host] = parse(ssh(host, proc_probe("mimo26-spark") +
                              f'echo "manifest_sha256: $(sha256sum {SLICES}/manifest.json 2>/dev/null | cut -d" " -f1)"\n'
                              f'f=/var/tmp/mimo26f-kernel/sparkd-rank{r}-b1.log; [ -f "$f" ] || f=/var/tmp/mimo26f-kernel/sparkd-rank{r}.log\n'
                              f'echo "readback: $(grep -h "slices match" "$f" 2>/dev/null | tail -1)"\n'))
    json.dump(rec, open(sys.argv[1], "w"), indent=1)
    print(json.dumps({h: {k: rec[h].get(k) for k in ("proc", "exe_sha256", "manifest_sha256")} for h in
                      ("coordinator", "spark1", "spark2", "spark3", "spark4")}, indent=1))


if __name__ == "__main__":
    main()
