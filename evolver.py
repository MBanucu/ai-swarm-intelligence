#!/usr/bin/env python3
import os
import sys
from datetime import datetime, timezone
import json
import shutil
import subprocess

from config import (
    ROOT_DIR,
    GEN_FILE,
    BASE_CODE,
    ARCHIVE_DIR,
    BASE_TEMPLATE,
    BENCHMARK_HISTORY,
    MAX_RETRIES,
)

from child import ChildProcess


def _run_baseline():
    engine_dir = os.path.join(BASE_CODE, "src", "jpeg_engine")
    proc = subprocess.Popen(
        ["cargo", "run", "--release", "--features", "gpu", "--bin", "bench", "--", "5000", "/dev/stdout"],
        cwd=engine_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    output_lines = []
    for line in proc.stdout:
        output_lines.append(line)
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        return None
    for line in reversed(output_lines):
        try:
            data = json.loads(line.strip())
            if isinstance(data, dict) and "fitness" in data:
                return float(data["fitness"])
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _collect_failure(child):
    failure = {"attempt": child.attempt, "reason": "unknown"}
    lifecycle_log = os.path.join(child.dir, "lifecycle.log")
    test_log = os.path.join(child.dir, "test_output.log")
    bench_log = os.path.join(child.dir, "benchmark.log")

    if os.path.exists(test_log):
        with open(test_log) as f:
            content = f.read()
        if "FAILED" in content or "error: test failed" in content:
            failure["reason"] = "cargo test failed"
            failure["test_tail"] = _tail_lines(content, 15)
        elif os.path.exists(bench_log):
            with open(bench_log) as f:
                failure["reason"] = "benchmark failed"
                failure["bench_tail"] = _tail_lines(f.read(), 10)
    elif os.path.exists(lifecycle_log):
        with open(lifecycle_log) as f:
            content = f.read()
        failure["reason"] = "lifecycle crashed"
        failure["lifecycle_tail"] = _tail_lines(content, 20)

    return failure


def _save_failure(failure_dir, attempt, failure):
    path = os.path.join(failure_dir, f"attempt_{attempt}.txt")
    with open(path, "w") as f:
        f.write(f"Attempt {attempt} - {failure['reason']}\n")
        f.write("=" * 60 + "\n")
        for key in ("test_tail", "bench_tail", "lifecycle_tail"):
            if key in failure:
                f.write(f"\n--- {key} ---\n")
                f.write(failure[key])
                f.write("\n")


def _tail_lines(text, n):
    lines = text.strip().split("\n")
    return "\n".join(lines[-n:])


def _save_state(gen, attempt, best_score=None):
    state = {
        "generation": gen,
        "attempt": attempt,
    }
    if best_score is not None:
        state["best_score"] = best_score
    with open(GEN_FILE, "w") as f:
        json.dump(state, f, indent=2)
        f.write("\n")


def _log_attempt(gen, attempt, score, status):
    ts = datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()
    entry = {"gen": gen, "attempt": attempt, "score": score, "status": status, "timestamp": ts}
    if os.path.exists(BENCHMARK_HISTORY):
        with open(BENCHMARK_HISTORY) as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    else:
        history = []
    history.append(entry)
    with open(BENCHMARK_HISTORY, "w") as f:
        json.dump(history, f, indent=2)
        f.write("\n")


def main():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "logs"), exist_ok=True)

    if not os.path.exists(GEN_FILE):
        old_file = os.path.join(ROOT_DIR, "logs", "current_gen.txt")
        if os.path.exists(old_file):
            with open(old_file) as f:
                gen = int(f.read().strip())
            _save_state(gen, 1)
            os.remove(old_file)
        else:
            _save_state(1, 1)

    with open(GEN_FILE) as f:
        state = json.load(f)
    gen = state["generation"]
    start_attempt = state.get("attempt", 1)

    prev_best = None
    if gen > 1 and os.path.exists(BENCHMARK_HISTORY):
        with open(BENCHMARK_HISTORY) as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
        for entry in history:
            if entry.get("gen") == gen - 1:
                score = float(entry["score"])
                if prev_best is None or score < prev_best:
                    prev_best = score

    if prev_best is not None:
        print(f"[swarm] Previous generation {gen - 1} best: {prev_best:.3f}ns/block")

    print("[swarm] Running baseline benchmark on parent code...")
    baseline = _run_baseline()
    if baseline is not None:
        print(f"[swarm] Baseline: {baseline:.3f}ns/block")
        if prev_best is None:
            prev_best = baseline
        elif baseline < prev_best:
            print(f"[swarm] Basline {baseline:.6f} < previous {prev_best:.6f} - tightening floor")
            prev_best = baseline
    print()

    gen_dir = os.path.join(ROOT_DIR, "generations", f"gen_{gen}")
    os.makedirs(gen_dir, exist_ok=True)

    parent_agent = (
        BASE_TEMPLATE
        if gen == 1
        else os.path.join(ARCHIVE_DIR, f"gen_{gen - 1}_winner.md")
    )

    print("=" * 80)
    print(f"  EVOLUTIONARY SWARM - Generation {gen}")
    print("=" * 80)
    print()

    best_score = 999.9
    winner_dir = None
    winner_attempt = 0
    sibling_failures = []
    failure_dir = os.path.join(gen_dir, "failures")
    os.makedirs(failure_dir, exist_ok=True)

    for attempt in range(start_attempt, MAX_RETRIES + 1):
        print()
        print("=" * 70)
        print(f"  Generation {gen} - Attempt {attempt} of {MAX_RETRIES}")
        print("=" * 70)

        child = ChildProcess(attempt, gen, gen_dir, parent_agent)
        child.sibling_failures = sibling_failures.copy()
        child.failure_dir = failure_dir

        print(f"\n--- Attempt {attempt} ---", flush=True)
        print(f"    dir: {child.dir}", flush=True)

        if child.run_lifecycle():
            if prev_best is not None and child.score >= prev_best:
                failure = {"attempt": child.attempt,
                           "reason": f"regression ({child.score:.3f}ns/block >= prev {prev_best:.3f}ns/block)"}
                sibling_failures.append(failure)
                _save_failure(failure_dir, attempt, failure)
                _save_state(gen, attempt + 1)
                _log_attempt(gen, attempt, child.score, "regression")
                print()
                print(f"REGRESSION on attempt {attempt}: {child.score:.3f}ns/block >= previous gen {prev_best:.3f}ns/block")
            elif child.score < best_score:
                best_score = child.score
                winner_dir = child.dir
                winner_attempt = attempt
                _save_state(gen, attempt + 1)
                _log_attempt(gen, attempt, child.score, "best")
                print()
                print(f">>> NEW BEST on attempt {attempt}: {best_score:.3f}ns/block")
            else:
                _save_state(gen, attempt + 1)
                _log_attempt(gen, attempt, child.score, "survived")
                print()
                print(f"    Survived on attempt {attempt}: {child.score:.3f}ns/block (best: {best_score:.3f})")
        else:
            failure = _collect_failure(child)
            sibling_failures.append(failure)
            _save_failure(failure_dir, attempt, failure)
            _save_state(gen, attempt + 1)
            _log_attempt(gen, attempt, 999.9, "extinction")

            print()
            print(f"EXTINCTION on attempt {attempt}: {failure['reason']}")
            print("Breeding fresh child...")

    if winner_dir is None:
        _save_state(gen, 1)
        print()
        print(f"FINAL EXTINCTION: All {MAX_RETRIES} attempts failed for Generation {gen}.")
        print("Re-running generation...")
        sys.exit(1)

    print()
    print(f">>> WINNER: Attempt {winner_attempt} - {best_score:.3f}ns/block")

    if prev_best is not None and best_score >= prev_best:
        print()
        print(f"[swarm] REGRESSION: winner {best_score:.3f}ns/block >= previous {prev_best:.3f}ns/block")
        print(f"[swarm] ABORTING generation {gen} - no improvement. Re-run for fresh mutations.")
        sys.exit(1)

    print()

    if os.path.exists(BASE_CODE):
        shutil.rmtree(BASE_CODE)
    shutil.copytree(winner_dir, BASE_CODE, symlinks=True)

    for junk in (
        "fitness.score", "lifecycle.log", "mutation.log",
        "test_output.log", "death_test.log", "benchmark.log",
        "fitness_history.json", "analysis.md", "analysis.log",
    ):
        jp = os.path.join(BASE_CODE, junk)
        if os.path.exists(jp):
            if os.path.isdir(jp):
                shutil.rmtree(jp)
            else:
                os.remove(jp)

    for root, dirs, files in os.walk(BASE_CODE):
        for d in dirs:
            if d in (".git", ".opencode", "node_modules", "target", "__pycache__"):
                shutil.rmtree(os.path.join(root, d))
        for f in files:
            if f == ".gitignore":
                os.remove(os.path.join(root, f))

    shutil.copy2(
        os.path.join(winner_dir, ".opencode", "agents", "dct-evolver.md"),
        os.path.join(ARCHIVE_DIR, f"gen_{gen}_winner.md"),
    )

    subprocess.run(["git", "-C", ROOT_DIR, "add", "-A"], check=False)

    diff_rc = subprocess.run(
        ["git", "-C", ROOT_DIR, "diff", "--cached", "--quiet"],
        check=False,
    ).returncode

    if diff_rc != 0:
        branch = f"evolution/gen-{gen}-winner"
        subprocess.run(
            [
                "git", "-C", ROOT_DIR, "commit", "-m",
                f"evolution(gen-{gen}): winner attempt-{winner_attempt}"
                f" at {best_score:.3f}ns/block",
            ],
            check=False,
        )
        subprocess.run(
            ["git", "-C", ROOT_DIR, "checkout", "-B", branch],
            check=False, stderr=subprocess.DEVNULL,
        )

        remote = subprocess.run(
            ["git", "-C", ROOT_DIR, "remote", "get-url", "origin"],
            capture_output=True, text=True,
        ).stdout.strip()

        if remote:
            push_result = subprocess.run(
                ["git", "-C", ROOT_DIR, "push", "origin", branch],
                check=False,
                capture_output=True, text=True,
            )
            if push_result.returncode == 0:
                subprocess.run(
                    [
                        "gh", "pr", "create",
                        "--head", branch,
                        "--title",
                        f"Evolution Gen {gen} Winner - {best_score:.3f}ns/block",
                        "--body",
                        f"Attempt {winner_attempt} won Generation {gen}"
                        f" with {best_score:.3f}ns/block.",
                        "--base", "main",
                    ],
                    check=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

    _save_state(gen + 1, 1)

    print()
    print(
        f"Generation {gen} consolidated."
        f" Ready for Generation {gen + 1}:"
    )
    print(f"  nix develop --command python3 evolver.py")


if __name__ == "__main__":
    main()
