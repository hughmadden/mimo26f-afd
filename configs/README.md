# configs/

Coordinator + spark configs. Minimal keys, fail-loud on unknown (ARCHITECTURE §8).

```text
role            = coordinator | spark
model_path      = host path to HF snapshot
expert_hosts    = 4 names or addrs
spark_tp        = 4
kv_pool_bytes   = derived + override
prefill_capacity= 256 | 1024 | 4096
draft           = none | mtp | dflash
aot_sm          = pinned per host class (gate)
```

Dev configs land at I3; fleet configs at I4. No secrets here — path references only.

`build.env` is the compile pin file (SM, build root, scratch ban). Agents edit
it when a path is wrong. They do not pass those flags on the command line.
