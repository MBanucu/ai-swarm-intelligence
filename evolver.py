#!/usr/bin/env python3
import os
import sys
from datetime import datetime, timezone
import shutil
import subprocess

MAX_RETRIES = 10

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GEN_FILE = os.path.join(ROOT_DIR, "logs", "current_gen.txt")
BASE_CODE = os.path.join(ROOT_DIR, "base_code")
ARCHIVE_DIR = os.path.join(ROOT_DIR, "logs", "archived_agents")
BASE_TEMPLATE = os.path.join(ROOT_DIR, ".opencode", "agents", "base_template.md")
BENCHMARK_HISTORY = os.path.join(ROOT_DIR, "logs", "benchmark_history.md")


class ChildProcess:
    def __init__(self, attempt, gen, gen_dir, parent_agent_path):
        self.attempt = attempt
        self.gen = gen
        self.dir = os.path.join(gen_dir, f"child_{attempt}")
        self.parent_agent_path = parent_agent_path
        self.score = 999.9
        self.sibling_failures = []
        self.failure_dir = None
        self.mutated_agent = os.path.join(
            self.dir, ".opencode", "agents", "dct-evolver.md"
        )
        self.analysis_path = os.path.join(self.dir, "analysis.md")

    def setup(self):
        if os.path.exists(self.dir):
            shutil.rmtree(self.dir)
        os.makedirs(os.path.join(self.dir, ".opencode", "agents"), exist_ok=True)
        for item in os.listdir(BASE_CODE):
            src = os.path.join(BASE_CODE, item)
            dst = os.path.join(self.dir, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, symlinks=True)
            else:
                shutil.copy2(src, dst)

    def run_lifecycle(self):
        self.setup()
        print(f"[Attempt {self.attempt}] Analyzing past generations...", flush=True)
        print(f"    logs -> {self.dir}/analysis.log", flush=True)
        self._analyze()
        print(f"[Attempt {self.attempt}] Creating mutation agent...", flush=True)
        print(f"    logs -> {self.dir}/mutation.log", flush=True)
        if not self._breed():
            print(f"  [Attempt {self.attempt}] Mutation failed", flush=True)
            return False
        print(f"[Attempt {self.attempt}] Spawning lifecycle...", flush=True)
        print(f"    logs -> {self.dir}/lifecycle.log", flush=True)
        if not self._optimize():
            print(f"  [Attempt {self.attempt}] Lifecycle failed", flush=True)
            return False
        print(f"[Attempt {self.attempt}] Running fitness benchmark...", flush=True)
        return self._benchmark()

    def _analyze(self):
        prompt = (
            f"You are a strategy analyst for Generation {self.gen} of an"
            f" evolutionary swarm optimizing a JPEG decoder engine.\n\n"
            f"Your task: analyze all available data and produce recommendations"
            f" for what mutation strategies to try and what to avoid.\n\n"
            f"SOURCES TO ANALYZE:\n"
            f"1) Benchmark history: '{BENCHMARK_HISTORY}' — performance trends"
            f" across generations.\n"
            f"2) Current codebase: '{BASE_CODE}' — the parent seed code.\n"
            f"   Focus on src/jpeg_engine/src/ and Cargo.toml.\n"
            f"3) Archived winner agents: '{ARCHIVE_DIR}' — strategies used by past winners.\n"
            f"4) Parent agent template: '{self.parent_agent_path}' — the mutation baseline.\n"
        )
        if self.sibling_failures:
            prompt += (
                f"5) Sibling failures from current generation:\n"
            )
            for f in self.sibling_failures[-5:]:
                prompt += f"   - Attempt {f['attempt']}: {f['reason']}\n"
            if self.failure_dir:
                prompt += (
                    f"   Failure summaries at: {self.failure_dir}/\n"
                    f"   YOU MUST READ EVERY FILE in {self.failure_dir}/"
                    f" for a quick overview.\n"
                    f"   FOR FULL LOGS, read the sibling sandboxes:\n"
                )
                for f in self.sibling_failures[-5:]:
                    prompt += (
                        f"     - child_{f['attempt']}/test_output.log"
                        f" (compiler errors)\n"
                        f"     - child_{f['attempt']}/lifecycle.log"
                        f" (optimization output)\n"
                        f"     - child_{f['attempt']}/benchmark.log"
                        f" (benchmark output)\n"
                    )
                prompt += (
                    f"   Extract the root cause of each failure from the"
                    f" FULL logs and incorporate those lessons into your"
                    f" Strategies to AVOID section.\n"
                )

        prompt += (
            f"\n\nOUTPUT: Write a concise analysis to '{self.analysis_path}' with:\n"
            f"## Performance Trends\n"
            f"- Which optimizations produced the best gains historically\n"
            f"- Where performance plateaued or regressed\n\n"
            f"## Code Hotspots\n"
            f"- Which functions/modules are the bottleneck candidates\n"
            f"- Current code structure insights\n\n"
            f"## Recommended Strategies (TRY THESE)\n"
            f"- 3-5 specific optimization angles with concrete rationale\n"
            f"- Mention specific files, functions, algorithms\n\n"
            f"## Strategies to AVOID\n"
            f"- Approaches that failed in past generations\n"
            f"- Anti-patterns that caused test failures or regressions\n\n"
            f"IMPORTANT CONTEXT:\n"
            f"- The benchmark tests 3 batch sizes via idct_2d_batch() API.\n"
            f"- idct_2d_batch receives ALL blocks at once — the engine can dispatch"
            f" to CPU or GPU based on block count.\n"
            f"- Fitness = 0.5*high + 0.3*mid + 0.2*low (lower is better).\n"
            f"- GPU acceleration (OpenCL) wins on large batches.\n"
            f"- CPU path wins on small batches. CPU parallelism (rayon) also viable.\n"
            f"- idct_2d_batch() should auto-select CPU vs GPU based on count.\n\n"
            f"IMMUTABLE FILES — DO NOT RECOMMEND MODIFYING:\n"
            f"- src/bin/bench.rs: contains mathematical reference IDCT validation."
            f" Changing it would invalidate fitness and crash the benchmark.\n"
            f"- src/lib.rs FFI signatures: changing idct_2d_batch signature"
            f" breaks the benchmark binary.\n\n"
            f"CRITICAL: ACTUALLY READ THE CODE before recommending.\n"
            f"Check Cargo.toml feature flags match what code expects.\n"
            f"Check GPU kernel source compiles with the target OpenCL runtime.\n"
            f"Check that Send+Sync bounds are satisfied for GPU dispatch types.\n"
            f"Look for mismatches between features in Cargo.toml and"
            f" #[cfg(feature=...)] guards in source code.\n\n"
            f"Output ONLY the raw markdown content for the analysis file."
        )

        cmd = [
            "unbuffer", "opencode",
            "--model", "opencode-go/deepseek-v4-flash",
            "run", "--share", "--thinking", prompt,
        ]
        log_path = os.path.join(self.dir, "analysis.log")
        with open(log_path, "w") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != 0 or not os.path.exists(self.analysis_path):
            print(f"[Attempt {self.attempt}] Analysis step failed — continuing without it",
                  flush=True)
            return
        print(f"[Attempt {self.attempt}] Analysis complete -> {self.analysis_path}",
              flush=True)

    def _breed(self):
        prompt = (
            f"Read the parent template at '{self.parent_agent_path}'.\n"
            f"Read the strategy analysis at '{self.analysis_path}'.\n"
            f"Read the benchmark history at '{BENCHMARK_HISTORY}' if it exists.\n\n"
            f"CRITICAL: Follow the RECOMMENDED STRATEGIES from the analysis."
            f" Avoid the strategies listed as AVOID."
            f" The analysis studied performance trends and past failures"
            f" to guide your mutation.\n\n"
            f"Output a UNIQUE mutated version of this OpenCode agent markdown file."
            f" Write it to '{self.mutated_agent}'.\n"
            f"Use a temperature between 0.3-0.8."
            f" Tweak the frontmatter parameters (temperature, maxSteps 20-60).\n"
            f"Rephrase optimization strategies differently from the parent"
            f" — try a DIFFERENT algorithmic angle.\n\n"
            f"FREEDOM OF LANGUAGE:\n"
            f"You can implement the core engine in Python, C, or Rust.\n"
            f"GPU acceleration via OpenCL/Vulkan/CUDA is encouraged.\n\n"
            f"ARCHITECTURAL CONSTRAINTS:\n"
            f"1. Keep the Rust engine inside 'src/jpeg_engine/'.\n"
            f"2. The library must expose C-compatible functions:"
            f" 'void idct_2d(double* block)', 'void dct_2d(double* block)'.\n"
            f"3. Update execution instructions so child workflows know how to build.\n"
            f"4. You can modify 'flake.nix' to add compilers, tools, or libraries.\n"
            f"5. GPU kernels should use the GpuKernel trait in src/gpu.rs.\n\n"
            f"CRITICAL — YAML FRONTMATTER RULES:\n"
            f"- Keep the exact line 'permission: allow' verbatim on its own line."
            f" Do NOT expand it into granular rules with sub-keys.\n"
            f"- Do NOT add anything under 'permission'."
            f" It must stay as the single line 'permission: allow'.\n"
            f"- Do NOT change 'bash: true', 'write: true', 'edit: true'"
            f" — the agent needs these to function.\n"
            f"- Preserve the --- YAML delimiter syntax exactly.\n\n"
            f"Output ONLY the raw markdown content."
        )
        cmd = [
            "unbuffer", "opencode",
            "--model", "opencode-go/deepseek-v4-flash",
            "run", "--share", "--thinking", prompt,
        ]
        with open(os.path.join(self.dir, "mutation.log"), "w") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            return False
        if not os.path.exists(self.mutated_agent):
            return False
        return True

    def _optimize(self):
        prompt = (
            f"You are Child {self.attempt} of Generation {self.gen}"
            f" in an evolutionary swarm.\n"
            f"Your task:\n"
            f"1) Read the Rust JPEG engine in 'src/jpeg_engine/src/'"
            f" (relative to your workspace at '{self.dir}')"
            f" to understand the codebase.\n"
            f"2) Read '{BENCHMARK_HISTORY}' to see prior generation results.\n"
            f"3) Optimize the JPEG engine for maximum speed."
            f" Apply your unique mutation strategy.\n"
            f"   Modify ONLY files inside your sandbox at '{self.dir}'.\n"
            f"   Do NOT touch '{BASE_CODE}' — that's the parent seed.\n"
            f"   You may modify Cargo.toml to add dependencies.\n"
            f"   GPU ACCELERATION: The engine has a GpuKernel trait in"
            f" src/jpeg_engine/src/gpu.rs. Implement GPU kernels"
            f" via OpenCL, Vulkan, or CUDA. Use the 'gpu' Cargo feature.\n"
            f"   ENVIRONMENT: 'cargo' and 'rustc' are on PATH."
            f" You are already inside the correct shell.\n"
            f"4) Run 'cargo test --release --features gpu' from src/jpeg_engine/"
            f" to confirm all tests pass.\n"
            f"5) Run 'cargo run --release --features gpu --bin bench -- 100 fitness.score'"
            f" to VALIDATE YOUR CHANGES WORK."
            f" You MUST run this yourself to check performance improves."
            f" If the benchmark shows regression, debug and fix it before finishing.\n"
            f"6) Note: the orchestrator runs the final benchmark separately."
            f" Your self-benchmark is for your own validation only."
        )
        if self.sibling_failures:
            prompt += f"\n\n## Previous Sibling Failures (DO NOT REPEAT)\n"
            if self.failure_dir:
                prompt += (
                    f"Failure summaries at: {self.failure_dir}/\n"
                    f"READ EVERY FILE in {self.failure_dir}/"
                    f" for a quick overview before making changes.\n"
                    f"FOR FULL LOGS, read sibling sandboxes:\n"
                )
                for f in self.sibling_failures[-5:]:
                    prompt += (
                        f"  - child_{f['attempt']}/test_output.log"
                        f" (compiler errors)\n"
                        f"  - child_{f['attempt']}/lifecycle.log"
                        f" (optimization output)\n"
                    )
                prompt += (
                    f"Understand the exact errors that killed each sibling.\n"
                )
            for f in self.sibling_failures[-5:]:
                prompt += f"- Attempt {f['attempt']}: {f['reason']}"
                if self.failure_dir:
                    prompt += f" (see {self.failure_dir}/attempt_{f['attempt']}.txt)"
                prompt += "\n"
                for key in ("test_tail", "bench_tail", "lifecycle_tail"):
                    if key in f:
                        lines = f[key].strip().split("\n")[-3:]
                        prompt += f"  {chr(10).join('  ' + l for l in lines)}\n"
            prompt += "\nAnalyze these failures and choose a DIFFERENT approach."
        cmd = [
            "unbuffer", "opencode",
            "--model", "opencode-go/deepseek-v4-flash",
            "--agent", "dct-evolver",
            "--dir", self.dir,
            "run", "--share", "--thinking", prompt,
        ]
        with open(os.path.join(self.dir, "lifecycle.log"), "w") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        return result.returncode == 0

    def _benchmark(self):
        engine_dir = os.path.join(self.dir, "src", "jpeg_engine")

        print(f"  [Attempt {self.attempt}] Running cargo test...")
        test_result = subprocess.run(
            ["cargo", "test", "--release", "--features", "gpu"],
            cwd=engine_dir, capture_output=True, text=True,
        )
        test_log = os.path.join(self.dir, "test_output.log")
        death_log = os.path.join(self.dir, "death_test.log")
        with open(test_log, "w") as f:
            f.write(test_result.stdout or "")
            if test_result.stderr:
                f.write("\n" + test_result.stderr)

        if test_result.returncode != 0:
            print(f"  [Attempt {self.attempt}] DIED — cargo test failed")
            with open(os.path.join(self.dir, "fitness.score"), "w") as f:
                f.write("999.9")
            shutil.copy2(test_log, death_log)
            return False

        print(f"  [Attempt {self.attempt}] Running cargo bench...")
        bench_path = os.path.join(self.dir, "fitness.score")
        bench_result = subprocess.run(
            ["cargo", "run", "--release", "--features", "gpu", "--bin", "bench", "--",
             "5000", bench_path],
            cwd=engine_dir, capture_output=True, text=True,
        )

        bench_log = os.path.join(self.dir, "benchmark.log")
        with open(bench_log, "w") as f:
            f.write(bench_result.stdout or "")
            if bench_result.stderr:
                f.write("\n" + bench_result.stderr)

        score_file = os.path.join(self.dir, "fitness.score")
        try:
            with open(score_file) as f:
                score = float(f.read().strip())
        except (FileNotFoundError, ValueError):
            score = 999.0

        if bench_result.returncode != 0 or score >= 999.0:
            print(f"  [Attempt {self.attempt}] DIED — Benchmark failed")
            with open(score_file, "w") as f:
                f.write("999.9")
            return False

        print(f"  [Attempt {self.attempt}] SURVIVED — Score: {score:.6f}ms/iter")
        self.score = score
        return True


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
        if "Fitness" in line:
            try:
                parts = line.split()
                return float(parts[2])
            except (IndexError, ValueError):
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
        f.write(f"Attempt {attempt} — {failure['reason']}\n")
        f.write("=" * 60 + "\n")
        for key in ("test_tail", "bench_tail", "lifecycle_tail"):
            if key in failure:
                f.write(f"\n--- {key} ---\n")
                f.write(failure[key])
                f.write("\n")


def _tail_lines(text, n):
    lines = text.strip().split("\n")
    return "\n".join(lines[-n:])


def main():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "logs"), exist_ok=True)

    if not os.path.exists(GEN_FILE):
        with open(GEN_FILE, "w") as f:
            f.write("1")

    with open(GEN_FILE) as f:
        gen = int(f.read().strip())

    prev_best = None
    if gen > 1 and os.path.exists(BENCHMARK_HISTORY):
        with open(BENCHMARK_HISTORY) as f:
            for line in f:
                if line.strip().startswith(f"| Gen {gen - 1}"):
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 4:
                        try:
                            prev_best = float(parts[3].rstrip("ms"))
                        except ValueError:
                            pass
                    break

    if prev_best is not None:
        print(f"[swarm] Previous generation {gen - 1} best: {prev_best:.6f}ms/iter")

    print("[swarm] Running baseline benchmark on parent code...")
    baseline = _run_baseline()
    if baseline is not None:
        print(f"[swarm] Baseline: {baseline:.6f}ms/iter")
        if prev_best is None:
            prev_best = baseline
        elif baseline < prev_best:
            print(f"[swarm] Basline {baseline:.6f} < previous {prev_best:.6f} — tightening floor")
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
    print(f"  EVOLUTIONARY SWARM — Generation {gen}")
    print("=" * 80)
    print()

    best_score = 999.9
    winner_dir = None
    winner_attempt = 0
    sibling_failures = []
    failure_dir = os.path.join(gen_dir, "failures")
    os.makedirs(failure_dir, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        print()
        print("=" * 70)
        print(f"  Generation {gen} — Attempt {attempt} of {MAX_RETRIES}")
        print("=" * 70)

        child = ChildProcess(attempt, gen, gen_dir, parent_agent)
        child.sibling_failures = sibling_failures.copy()
        child.failure_dir = failure_dir

        print(f"\n--- Attempt {attempt} ---", flush=True)
        print(f"    dir: {child.dir}", flush=True)

        if child.run_lifecycle():
            if prev_best is not None and child.score >= prev_best:
                failure = {"attempt": child.attempt,
                           "reason": f"regression ({child.score:.6f}ms >= prev {prev_best:.6f}ms)"}
                sibling_failures.append(failure)
                _save_failure(failure_dir, attempt, failure)
                print()
                print(f"REGRESSION on attempt {attempt}: {child.score:.6f}ms >= previous gen {prev_best:.6f}ms")
            elif child.score < best_score:
                best_score = child.score
                winner_dir = child.dir
                winner_attempt = attempt
                print()
                print(f">>> NEW BEST on attempt {attempt}: {best_score:.6f}ms/iter")
            else:
                print()
                print(f"    Survived on attempt {attempt}: {child.score:.6f}ms/iter (best: {best_score:.6f})")
        else:
            failure = _collect_failure(child)
            sibling_failures.append(failure)
            _save_failure(failure_dir, attempt, failure)

            print()
            print(f"EXTINCTION on attempt {attempt}: {failure['reason']}")
            print("Breeding fresh child...")

    if winner_dir is None:
        print()
        print(f"FINAL EXTINCTION: All {MAX_RETRIES} attempts failed for Generation {gen}.")
        print("Re-running generation...")
        sys.exit(1)

    print()
    print(f">>> WINNER: Attempt {winner_attempt} — {best_score:.6f}ms/iter")

    if prev_best is not None and best_score >= prev_best:
        print()
        print(f"[swarm] REGRESSION: winner {best_score:.6f}ms >= previous {prev_best:.6f}ms")
        print(f"[swarm] ABORTING generation {gen} — no improvement. Re-run for fresh mutations.")
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

    with open(BENCHMARK_HISTORY, "a") as f:
        ts = datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()
        f.write(
            f"| Gen {gen} | Attempt {winner_attempt}"
            f" | {best_score:.6f}ms | {ts} |\n"
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
                f" at {best_score:.6f}ms/iter",
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
                        f"Evolution Gen {gen} Winner — {best_score:.6f}ms/iter",
                        "--body",
                        f"Attempt {winner_attempt} won Generation {gen}"
                        f" with {best_score:.6f}ms/iter.",
                        "--base", "main",
                    ],
                    check=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

    with open(GEN_FILE, "w") as f:
        f.write(str(gen + 1))

    print()
    print(
        f"Generation {gen} consolidated."
        f" Ready for Generation {gen + 1}:"
    )
    print(f"  nix develop --command python3 evolver.py")


if __name__ == "__main__":
    main()
