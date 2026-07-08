import os
import shutil
import subprocess
import time

from config import (
    ROOT_DIR,
    BASE_CODE,
    BENCHMARK_HISTORY,
    ARCHIVE_DIR,
    IMPROVEMENT_DIR,
    PRIMARY_MODEL,
    FALLBACK_MODEL,
)

_PERF_EVENTS = "instructions,cycles,cache-misses,branch-misses,task-clock"
_STALL_TIMEOUT = 300


def _perf_available():
    return shutil.which("perf") is not None


def _run_opencode(args, log_path):
    """Run opencode with primary model, fallback on failure."""
    for model in (PRIMARY_MODEL, FALLBACK_MODEL):
        cmd = ["unbuffer", "opencode", "--model", model] + args
        with open(log_path, "w") as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

        last_size = 0
        stall_start = None
        while proc.poll() is None:
            time.sleep(3)
            try:
                cur_size = os.path.getsize(log_path)
            except OSError:
                cur_size = 0
            if cur_size > last_size:
                last_size = cur_size
                stall_start = None
            elif stall_start is None:
                stall_start = time.monotonic()
            elif time.monotonic() - stall_start > _STALL_TIMEOUT:
                proc.kill()
                proc.wait()
                print(
                    f"  [timeout] agent stalled >{_STALL_TIMEOUT}s, terminating...",
                    flush=True,
                )
                break

        if proc.returncode == 0:
            return True
        print(
            f"  [fallback] model={model} failed (rc={proc.returncode}), retrying...",
            flush=True,
        )
    return False


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
            f"5) Improvement suggestions: '{IMPROVEMENT_DIR}' — external"
            f" optimization ideas to consider. Read EVERY .md file in this"
            f" directory.\n"
            f"6) Baseline profiling: '{ROOT_DIR}/logs/baseline_perf.log' —"
            f" hardware performance counters (instructions, cycles,"
            f" cache-misses, branch-misses, task-clock) from the parent"
            f" code's benchmark. This reveals CPU bottlenecks, cache behavior,"
            f" and branch-prediction efficiency."
            f" Compare sibling perf_stat.log files against this baseline.\n"
        )
        if self.sibling_failures:
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
                        f"     - child_{f['attempt']}/perf_stat.log"
                        f" (hardware counters)\n"
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
            f"- The benchmark tests 6 batch sizes (10-250K blocks) spanning\n"
            f"  from single blocks to a typical 5 MB JPEG decode.\n"
            f"- Fitness = weighted average: 250K(50%) + 25K(20%) + 5K(10%)\n"
            f"  + 1K(10%) + 250(7%) + 10(3%). Lower is better.\n"
            f"- GPU wins overwhelmingly on 250K blocks (50% of fitness).\n"
            f"  GPU overhead (~100-500us) hurts on <500 blocks.\n"
            f"- idct_2d_batch receives ALL blocks at once - dispatch CPU or\n"
            f"  GPU based on block count using GPU_THRESHOLD.\n"
            f"- GPU acceleration (OpenCL) wins on large batches.\n"
            f"- CPU path wins on small batches. CPU parallelism (rayon) also viable.\n"
            f"- idct_2d_batch() should auto-select CPU vs GPU based on count.\n\n"
            f"IMMUTABLE FILES - DO NOT RECOMMEND MODIFYING:\n"
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

        log_path = os.path.join(self.dir, "analysis.log")
        ok = _run_opencode(["run", "--share", "--thinking", prompt], log_path)
        if not ok or not os.path.exists(self.analysis_path):
            print(f"[Attempt {self.attempt}] Analysis step failed - continuing without it",
                  flush=True)
            return
        print(f"[Attempt {self.attempt}] Analysis complete -> {self.analysis_path}",
              flush=True)

    def _breed(self):
        prompt = (
            f"Read the parent template at '{self.parent_agent_path}'.\n"
            f"Read the strategy analysis at '{self.analysis_path}'.\n"
            f"Read the benchmark history at '{BENCHMARK_HISTORY}' if it exists.\n"
            f"Read all files in '{IMPROVEMENT_DIR}' for external optimization ideas.\n\n"
            f"CRITICAL: Follow the RECOMMENDED STRATEGIES from the analysis."
            f" Avoid the strategies listed as AVOID."
            f" The analysis studied performance trends and past failures"
            f" to guide your mutation.\n\n"
            f"Output a UNIQUE mutated version of this OpenCode agent markdown file."
            f" Write it to '{self.mutated_agent}'.\n"
            f"Use a temperature between 0.3-0.8."
            f" Tweak the frontmatter parameters (temperature, maxSteps 40-120).\n"
            f"Rephrase optimization strategies differently from the parent"
            f" - try a DIFFERENT algorithmic angle.\n\n"
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
            f"CRITICAL - YAML FRONTMATTER RULES:\n"
            f"- Keep the exact line 'permission: allow' verbatim on its own line."
            f" Do NOT expand it into granular rules with sub-keys.\n"
            f"- Do NOT add anything under 'permission'."
            f" It must stay as the single line 'permission: allow'.\n"
            f"- Do NOT change 'bash: true', 'write: true', 'edit: true'"
            f" - the agent needs these to function.\n"
            f"- Preserve the --- YAML delimiter syntax exactly.\n\n"
            f"Output ONLY the raw markdown content."
        )
        ok = _run_opencode(
            ["run", "--share", "--thinking", prompt],
            os.path.join(self.dir, "mutation.log"),
        )
        if not ok:
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
            f"   Do NOT touch '{BASE_CODE}' - that's the parent seed.\n"
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
                        f"  - child_{f['attempt']}/perf_stat.log"
                        f" (hardware counters)\n"
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
        ok = _run_opencode(
            ["--agent", "dct-evolver", "--dir", self.dir,
             "run", "--share", "--thinking", prompt],
            os.path.join(self.dir, "lifecycle.log"),
        )
        return ok

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
            print(f"  [Attempt {self.attempt}] DIED - cargo test failed")
            with open(os.path.join(self.dir, "fitness.score"), "w") as f:
                f.write("999.9")
            shutil.copy2(test_log, death_log)
            return False

        print(f"  [Attempt {self.attempt}] Running cargo bench...")
        bench_path = os.path.join(self.dir, "fitness.score")

        bench_cmd = ["cargo", "run", "--release", "--features", "gpu",
                      "--bin", "bench", "--", "5000", bench_path]
        use_perf = _perf_available()

        if use_perf:
            perf_log = os.path.join(self.dir, "perf_stat.log")
            perf_cmd = ["perf", "stat", "-e", _PERF_EVENTS, "-o", perf_log,
                        "--"] + bench_cmd
            bench_result = subprocess.run(
                perf_cmd,
                cwd=engine_dir, capture_output=True, text=True,
            )
            if bench_result.returncode != 0:
                use_perf = False

        if not use_perf:
            bench_result = subprocess.run(
                bench_cmd,
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
            print(f"  [Attempt {self.attempt}] DIED - Benchmark failed")
            with open(score_file, "w") as f:
                f.write("999.9")
            return False

        print(f"  [Attempt {self.attempt}] SURVIVED - Score: {score:.3f}ns/block")
        self.score = score
        return True
