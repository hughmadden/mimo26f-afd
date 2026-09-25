# harness/

| Dir | Role | Layer |
|---|---|---|
| `selftests/` | Offline tests of the harness itself (R6) | gate before any model run |
| `stubs/` | LaneSim, fake RPC, draft sim, fault inject | L2 — design only |
| `fleet/` | coherence, COUNT, needle, decode, draft drivers | L4–L6 |

See `../TEST-PLAN.md`. Selftests green before a harness touches a model.
