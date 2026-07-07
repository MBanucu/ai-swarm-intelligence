#!/usr/bin/env python3
import os
import sys
import time
import shutil
import subprocess
import statistics

try:
    import psutil
except ImportError:
    psutil = None

SPAWN_THRESHOLD = 60.0
WORKER_CORES = [0, 1, 2]
BENCH_CORE = 3
MAX_RETRIES = 10

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GEN_FILE = os.path.join(ROOT_DIR, "logs", "current_gen.txt")
BASE_CODE = os.path.join(ROOT_DIR, "base_code")
ARCHIVE_DIR = os.path.join(ROOT_DIR, "logs", "archived_agents")
BASE_TEMPLATE = os.path.join(ROOT_DIR, ".opencode", "agents", "base_template.md")
BENCHMARK_HISTORY = os.path.join(ROOT_DIR, "logs", "benchmark_history.md")


class ChildProcess:
    def __init__(self, attempt, gen, gen_dir, worker_core, parent_agent_path):
        self.attempt = attempt
        self.gen = gen
        self.dir = os.path.join(gen_dir, f"child_{attempt}")
        self.core = worker_core
        self.parent_agent_path = parent_agent_path
        self.score = 999.9
        self.mutated_agent = os.path.join(
            self.dir, ".opencode", "agents", "dct-evolver.md"
        )

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
        print(f"[Attempt {self.attempt}] Creating mutation agent (core {self.core})...", flush=True)
        print(f"    logs -> {self.dir}/mutation.log", flush=True)
        if not self._breed():
            print(f"  [Attempt {self.attempt}] Mutation failed", flush=True)
            return False
        print(f"[Attempt {self.attempt}] Spawning child lifecycle (core {self.core})...", flush=True)
        print(f"    logs -> {self.dir}/lifecycle.log", flush=True)
        if not self._optimize():
            print(f"  [Attempt {self.attempt}] Lifecycle failed", flush=True)
            return False
        print(f"[Attempt {self.attempt}] Running fitness benchmark (core {BENCH_CORE})...", flush=True)
        return self._benchmark()

    def _breed(self):
        prompt = (
            f"Read the template at '{self.parent_agent_path}'"
            f" and the benchmark history at '{BENCHMARK_HISTORY}' if it exists.\n"
            f"Output a UNIQUE mutated version of this OpenCode agent markdown file."
            f" Write it to '{self.mutated_agent}'.\n"
            f"Use a temperature between 0.3-0.8."
            f" Tweak the frontmatter parameters (temperature, maxSteps 20-60).\n"
            f"Rephrase optimization strategies differently from the parent"
            f" — try a DIFFERENT algorithmic angle.\n\n"
            f"FREEDOM OF LANGUAGE:\n"
            f"You are explicitly allowed to move away from pure Python."
            f" You can choose to implement the core IDCT math in Python, C, or Rust.\n\n"
            f"ARCHITECTURAL CONSTRAINTS:\n"
            f"1. If you use Python, keep the structure inside 'src/dct_engine.py'.\n"
            f"2. If you use C or Rust, you must write a script or Makefile that"
            f" builds a shared object file at 'src/libdct_engine.so'.\n"
            f"3. The library must expose a C-compatible function:"
            f" 'void idct_2d(double* block);'.\n"
            f"4. Update the execution instructions in your core rules so your"
            f" sibling workflows know how to build your code.\n"
            f"5. DYNAMIC DEPENDENCIES: You can modify 'flake.nix' inside the"
            f" sandbox to add new compilers, tools, or libraries to"
            f" 'buildInputs'. If you add a dependency, instruct the child"
            f" agent to execute its build/test tasks via"
            f" 'nix develop --command <command>' inside their bash tool.\n\n"
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
            "taskset", "-c", str(self.core),
            "unbuffer", "opencode",
            "--model", "opencode-go/deepseek-v4-flash",
            "run", "--thinking", prompt,
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
            f"1) Read '{os.path.join(self.dir, 'src', 'dct_engine.py')}',"
            f" '{os.path.join(self.dir, 'flake.nix')}',"
            f" and '{os.path.join(self.dir, 'tests', 'test_dct_engine.py')}'"
            f" to understand the codebase and environment.\n"
            f"2) Read '{BENCHMARK_HISTORY}' to see prior generation results.\n"
            f"3) Optimize the DCT engine for maximum speed."
            f" Apply your unique mutation strategy.\n"
            f"   You may modify 'flake.nix' to add any required Nix packages"
            f" to 'buildInputs'.\n"
            f"   CRITICAL: If you modify 'flake.nix' or need to use packages"
            f" defined in it, you MUST run those build or test commands"
            f" prefixed with nix develop, like this:\n"
            f"   'nix develop --command python3 -m unittest tests.test_dct_engine -v'\n"
            f"4) Run tests and ensure they all pass.\n"
            f"5) Run a quick benchmark of 1000 idct_2d iterations and write"
            f" the ms/iter value ONLY to"
            f" '{os.path.join(self.dir, 'fitness.score')}'"
            f" (plain number, e.g. 0.085)."
        )
        cmd = [
            "taskset", "-c", str(self.core),
            "unbuffer", "opencode",
            "--model", "opencode-go/deepseek-v4-flash",
            "--agent", "dct-evolver",
            "--dir", self.dir,
            "run", "--thinking", prompt,
        ]
        with open(os.path.join(self.dir, "lifecycle.log"), "w") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        return result.returncode == 0

    def _benchmark(self):
        test_result = subprocess.run(
            ["python3", "-m", "unittest", "tests.test_dct_engine", "-v"],
            cwd=self.dir, capture_output=True, text=True,
        )
        test_log = os.path.join(self.dir, "test_output.log")
        with open(test_log, "w") as f:
            f.write(test_result.stdout or "")
            if test_result.stderr:
                f.write("\n" + test_result.stderr)

        if test_result.returncode != 0:
            print(f"  [Attempt {self.attempt}] DIED — Tests failed")
            with open(os.path.join(self.dir, "fitness.score"), "w") as f:
                f.write("999.9")
            shutil.copy2(test_log, os.path.join(self.dir, "death_test.log"))
            return False

        bench_script = (
            f"import time, statistics, sys, os, ctypes\n"
            f"CHILD_DIR = r'{self.dir}'\n"
            f"SO_PATH = os.path.join(CHILD_DIR, 'src', 'libdct_engine.so')\n"
            f"if os.path.exists(SO_PATH):\n"
            f"    lib = ctypes.CDLL(SO_PATH)\n"
            f"    lib.idct_2d.argtypes = [ctypes.POINTER(ctypes.c_double)]\n"
            f"    def run_idct(b):\n"
            f"        flat_arr = (ctypes.c_double * 64)(*[v for r in b for v in r])\n"
            f"        lib.idct_2d(flat_arr)\n"
            f"else:\n"
            f"    sys.path.insert(0, CHILD_DIR)\n"
            f"    from src.dct_engine import idct_2d\n"
            f"    def run_idct(b):\n"
            f"        idct_2d(b)\n"
            f"block = [[float(i*j%256-128) for j in range(8)] for i in range(8)]\n"
            f"for _ in range(200):\n"
            f"    run_idct(block)\n"
            f"rounds, iters = 10, 5000\n"
            f"samples = []\n"
            f"for _ in range(rounds):\n"
            f"    start = time.perf_counter()\n"
            f"    for _ in range(iters):\n"
            f"        run_idct(block)\n"
            f"    samples.append((time.perf_counter()-start)/iters*1000)\n"
            f"score = statistics.median(samples)\n"
            f"with open(os.path.join(CHILD_DIR, 'fitness.score'), 'w') as f:\n"
            f"    f.write(f'{{score:.6f}}')\n"
        )
        subprocess.run(
            ["taskset", "-c", str(BENCH_CORE), "python3", "-c", bench_script],
            capture_output=True, text=True,
        )

        score_file = os.path.join(self.dir, "fitness.score")
        try:
            with open(score_file) as f:
                score = float(f.read().strip())
        except (FileNotFoundError, ValueError):
            score = 999.0
        print(f"  [Attempt {self.attempt}] SURVIVED — Score: {score:.6f}ms/iter")
        self.score = score
        return True

def wait_for_cpu(headroom):
    if psutil is None:
        time.sleep(2)
        return
    psutil.cpu_percent(interval=None)
    while True:
        load = psutil.cpu_percent(interval=2.0)
        if load < headroom:
            return
        print(f"  CPU at {load:.0f}%, waiting for headroom < {headroom:.0f}%...")


def main():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "logs"), exist_ok=True)

    if not os.path.exists(GEN_FILE):
        with open(GEN_FILE, "w") as f:
            f.write("1")

    with open(GEN_FILE) as f:
        gen = int(f.read().strip())

    gen_dir = os.path.join(ROOT_DIR, "generations", f"gen_{gen}")
    os.makedirs(gen_dir, exist_ok=True)

    parent_agent = (
        BASE_TEMPLATE
        if gen == 1
        else os.path.join(ARCHIVE_DIR, f"gen_{gen - 1}_winner.md")
    )

    print("=" * 80)
    print(f"  EVOLUTIONARY SWARM — Generation {gen}")
    print(f"  Worker cores: {WORKER_CORES} | Benchmark core: {BENCH_CORE}")
    print("=" * 80)
    print()

    best_score = 999.9
    winner_dir = None
    winner_attempt = 0

    for attempt in range(1, MAX_RETRIES + 1):
        print()
        print("=" * 70)
        print(f"  Generation {gen} — Attempt {attempt} of {MAX_RETRIES}")
        print("=" * 70)

        core = WORKER_CORES[(attempt - 1) % len(WORKER_CORES)]
        child = ChildProcess(attempt, gen, gen_dir, core, parent_agent)

        print(f"\n--- Attempt {attempt} on core {core} ---", flush=True)
        print(f"    dir: {child.dir}", flush=True)
        wait_for_cpu(SPAWN_THRESHOLD)

        if child.run_lifecycle():
            best_score = child.score
            winner_dir = child.dir
            winner_attempt = attempt
            print()
            print(f">>> SURVIVOR FOUND on attempt {attempt}: {best_score:.6f}ms/iter")
            break

        print()
        print(f"EXTINCTION on attempt {attempt}: No viable survivor.")
        print("Breeding fresh child...")

    if winner_dir is None:
        print()
        print(f"FINAL EXTINCTION: All {MAX_RETRIES} attempts failed for Generation {gen}.")
        print("Re-running generation...")
        sys.exit(1)

    print()
    print(f">>> WINNER: Attempt {winner_attempt} — {best_score:.6f}ms/iter")
    print()

    if os.path.exists(BASE_CODE):
        shutil.rmtree(BASE_CODE)
    shutil.copytree(winner_dir, BASE_CODE, symlinks=True)

    for junk in (
        "fitness.score", "lifecycle.log", "mutation.log",
        "test_output.log", "death_test.log", "fitness_history.json",
    ):
        jp = os.path.join(BASE_CODE, junk)
        if os.path.exists(jp):
            if os.path.isdir(jp):
                shutil.rmtree(jp)
            else:
                os.remove(jp)

    for root, dirs, files in os.walk(BASE_CODE):
        for d in dirs:
            if d == ".git":
                shutil.rmtree(os.path.join(root, d))
        for f in files:
            if f == ".gitignore":
                os.remove(os.path.join(root, f))

    shutil.copy2(
        os.path.join(winner_dir, ".opencode", "agents", "dct-evolver.md"),
        os.path.join(ARCHIVE_DIR, f"gen_{gen}_winner.md"),
    )

    with open(BENCHMARK_HISTORY, "a") as f:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
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
            subprocess.run(
                ["git", "-C", ROOT_DIR, "push", "origin", branch],
                check=False, stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                [
                    "gh", "pr", "create",
                    "--title",
                    f"Evolution Gen {gen} Winner — {best_score:.6f}ms/iter",
                    "--body",
                    f"Attempt {winner_attempt} won Generation {gen}"
                    f" with {best_score:.6f}ms/iter.",
                    "--base", "main",
                ],
                check=False, stderr=subprocess.DEVNULL,
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
