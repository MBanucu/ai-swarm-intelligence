# AI Swarm Intelligence

Parallel evolutionary optimization of a DCT engine using OpenCode agents that mutate, compete, and reproduce.

## Quick start

```bash
bash evolver.sh
```

Runs one generation: mutates agent genomes, spawns parallel child sandboxes, evaluates fitness, selects winner, commits to git, and opens a GitHub PR.

## Architecture

```
base_code/                  # Seed code for each generation (winner from previous gen)
  src/dct_engine.py         # The organism being evolved
  tests/test_dct_engine.py  # Fitness function (unittest, not pytest)

generations/gen_N/child_X/  # Per-child-isolated sandbox (in .gitignore)
  .opencode/agents/dct-evolver.md  # Mutated agent genome for this child
  src/dct_engine.py                 # Child's working copy
  fitness.score                     # Median ms/iter (lower wins)

logs/
  current_gen.txt           # Generation counter (integer)
  benchmark_history.md      # Phylogenetic performance ledger
  archived_agents/          # Winner genomes for generational continuity
```

## Key commands

| Command | Purpose |
|---|---|
| `bash evolver.sh` | Run one evolutionary generation |
| `python3 -m unittest tests.test_dct_engine -v` | Run fitness tests (uses built-in unittest, not pytest) |
| `python3 -m unittest discover -s base_code/ -p 'test_*.py' -v` | Run tests against base_code copy |

## Critical gotchas

- **`permission: allow` must be a single line** in any agent `.md` file. Never expand it into granular rules — child agents die from auto-rejected bash commands if this rule breaks.
- **Never do `git checkout` or `rm -rf generations/`.** The evolver manages git state. `git checkout -f main` wipes uncommitted fixes. `generations/` is gitignored but contains the only record of each child's work.
- **`root src/` and `tests/` are stale.** They're the original Gen 0 code. The evolver reads from `base_code/`, not the root directories. Don't edit root `src/dct_engine.py` expecting it to affect evolution.
- **Gen 2 is missing from benchmark_history.md** — it was lost during a git reset. That gap is intentional, not a bug to fix.
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
