# scripts/

Agents call `dev.sh`. They do not call `cargo`, `nvcc`, `cmake`, or `pytest`
directly. Skill: `mimo26f-build`. Pins: `configs/build.env`.

| Verb | Script | Status |
|---|---|---|
| `dev.sh doctor` | `preflight.sh` | live |
| `dev.sh check` | `ci-cpu.sh` | live. Missing suite is a fail. |
| `dev.sh retro` | `retro-scan.sh` | live. Required at iteration close (`mimo26f-retro`). |
| `dev.sh spike` | `spike-e2e.sh` | closed until I1 |
| `dev.sh build` | — | closed. One target: `cpu`, `coordinator`, `spark`. |
| `dev.sh test` | — | closed. Named cells only. |
| `dev.sh deploy` | `go-window.sh` | closed until I5 |

A closed or unknown verb exits 2. That is not permission to improvise.
`ci-cpu.sh` must fail loud if a suite is missing — silence is not green.
