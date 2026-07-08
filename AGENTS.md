# AI Swarm Intelligence

Parallel evolutionary optimization of a DCT engine using OpenCode agents that mutate, compete, and reproduce.

## Quick start

```bash
nix develop --command python3 evolver.py
```

Runs one generation: mutates agent genomes, spawns parallel child sandboxes, evaluates fitness, selects winner, commits to git, and opens a GitHub PR.

## Architecture

```
base_code/                       # Seed code for each generation (winner from previous gen)
  src/jpeg_engine/               # Rust JPEG decode pipeline (the organism being evolved)
    src/lib.rs                   # FFI exports + public API
    src/header.rs                # JPEG marker/header parser
    src/huffman.rs               # Huffman decode tables + decoder
    src/dct.rs                   # Forward DCT (8x8)
    src/idct.rs                  # Inverse DCT (8x8)
    src/scaling.rs               # YCbCr color transform, bilinear scale
    src/gpu.rs                   # GPU kernel trait + CPU fallback
    tests/                       # Real JPEG test fixtures
  flake.nix                      # Nix dev shell with cargo, opencode, etc.

generations/gen_N/child_X/       # Per-attempt isolated sandbox (in .gitignore)
  .opencode/agents/dct-evolver.md  # Mutated agent genome
  src/jpeg_engine/                 # Child's working copy of the engine
  fitness.score                    # Weighted median ns/block (lower wins)

logs/
  current_gen.json                # Generation + attempt counter (resumable)
  benchmark_history.json           # Phylogenetic performance ledger
  archived_agents/               # Winner genomes for generational continuity
```

## Key commands

| Command | Purpose |
|---|---|
| `nix develop --command python3 evolver.py` | Run one evolutionary generation |
| `nix develop --command cargo test --release` | Run Rust tests (cwd: base_code/src/jpeg_engine) |
| `nix develop --command cargo run --release --bin bench -- 5000 fitness.score` | Rust benchmark (same dir) |

## Critical gotchas

- **`permission: allow` must be a single line** in any agent `.md` file. Never expand it into granular rules — child agents die from auto-rejected bash commands if this rule breaks.
- **Never do `git checkout` or `rm -rf generations/`.** The evolver manages git state. `git checkout -f main` wipes uncommitted fixes. `generations/` is gitignored but contains the only record of each child's work.

- **Gen 2 is missing from benchmark_history.json** — it was lost during a git reset. That gap is intentional, not a bug to fix.
- **No pytest.** Tests use Python's built-in `unittest`. `pytest` commands will fail unless installed separately.
- **CPU core isolation is hardcoded:** workers on cores 0-2, benchmarks on core 3. Only works on ≥4-core machines.
- **GitHub PRs auto-create but may warn** when a PR already exists for the same generation. This is harmless.

## Agent genome flow

```
base_template.md ──mutate──▶ child_1/.opencode/agents/dct-evolver.md
                   ├─mutate─▶ child_2/.opencode/agents/dct-evolver.md
                   │                   │
                   │          [winner selected]
                   │                   │
                   └──────────────────▶ logs/archived_agents/gen_N_winner.md
                                              │
                                    parent for Gen N+1
```

The root `.opencode/agents/dct-evolver.md` was removed — it's dead configuration. The active genomes are per-child and per-archive.

## Nix environment (optional)

```bash
nix develop
```

Provides `python3`, `git`, `gh`, `bc`, `opencode`, `util-linux` (for `taskset`) in a reproducible shell. Not required to run the evolver if these are already on `$PATH`.
