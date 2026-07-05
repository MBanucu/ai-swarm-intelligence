#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
GENERATION_FILE="$ROOT_DIR/logs/current_gen.txt"
BENCHMARK_LOG="$ROOT_DIR/logs/benchmark_history.md"
AGENT_DIR="$ROOT_DIR/agents"

[ ! -f "$GENERATION_FILE" ] && echo "1" > "$GENERATION_FILE"
GEN=$(cat "$GENERATION_FILE")

mkdir -p "$ROOT_DIR/cemetery" "$ROOT_DIR/logs"

echo "================================================================================"
echo "  EVOLUTIONARY SANDBOX — Generation $GEN"
echo "================================================================================"

WORK_DIR="$AGENT_DIR/agent_generation_${GEN}"
mkdir -p "$WORK_DIR"

START_PERF="N/A"
if [ -f "$BENCHMARK_LOG" ]; then
    START_PERF=$(tail -5 "$BENCHMARK_LOG" | grep -oP 'perf_iter:\s*\K[\d.]+' | head -1 || echo "N/A")
fi

echo "[*] Spawning agent in $WORK_DIR"
echo "[*] Baseline performance (prev gen): $START_PERF"

set +e
opencode \
    --model opencode-go/deepseek-v4-flash \
    run \
    "You are an evolutionary optimization agent inside generation $GEN of the DCT engine evolution. Your task: 1) Read src/dct_engine.py and tests/test_dct_engine.py to understand the current codebase. 2) Read logs/benchmark_history.md to see historical performance data (if any). 3) Optimize src/dct_engine.py for faster IDCT throughput without breaking any tests. 4) Run 'python3 -m unittest tests.test_dct_engine -v' to confirm all tests pass. 5) If your optimization is complete, write the new benchmark result using: echo '| Gen $GEN | \$(date -Iseconds) | perf_iter: <ms>ms |' >> logs/benchmark_history.md"
EVOLVER_EXIT=$?
set -e

echo ""
echo "--- Fitness Evaluation ---"

if [ $EVOLVER_EXIT -ne 0 ]; then
    echo "FAILURE: Agent process exited with code $EVOLVER_EXIT"
    EVOLVER_RESULT=1
else
    echo "Running test suite..."
    if python3 -m unittest tests.test_dct_engine -v 2>&1; then
        echo "SUCCESS: All tests pass. Generation $GEN survives!"
        EVOLVER_RESULT=0
    else
        echo "FAILURE: Tests do not pass."
        EVOLVER_RESULT=1
    fi
fi

if [ "$EVOLVER_RESULT" -eq 0 ]; then
    echo ""
    echo ">>> REPRODUCTION: Generation $GEN has survived!"

    git -C "$ROOT_DIR" add -A

    if git -C "$ROOT_DIR" diff --cached --quiet; then
        echo "No changes to commit. Skipping reproduction."
    else
        BRANCH_NAME="evolution/gen-${GEN}"
        git -C "$ROOT_DIR" checkout -b "$BRANCH_NAME" 2>/dev/null || git -C "$ROOT_DIR" checkout "$BRANCH_NAME"
        git -C "$ROOT_DIR" commit -m "evolution(gen-$GEN): survived fitness test"

        REMOTE=$(git -C "$ROOT_DIR" remote get-url origin 2>/dev/null || echo "")
        if [ -n "$REMOTE" ]; then
            git -C "$ROOT_DIR" push origin "$BRANCH_NAME" 2>/dev/null || echo "Warning: Could not push to remote"
            gh pr create \
                --title "Evolution: Gen $GEN Optimization" \
                --body "Automated PR for successful generation $GEN mutation." \
                --base main \
                2>/dev/null || echo "Warning: Could not create PR"
        else
            echo "No git remote configured — skipping GitHub sync."
        fi
    fi

    echo "Generation $GEN complete. Incrementing generation counter."
    echo $((GEN + 1)) > "$GENERATION_FILE"
    NEXT_GEN=$((GEN + 1))
    echo ""
    echo "Run again to spawn Generation $NEXT_GEN:"
    echo "  bash evolver.sh"
else
    echo ""
    echo ">>> DEATH: Generation $GEN has failed the fitness test."

    FAIL_DIR="$ROOT_DIR/cemetery/gen_${GEN}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$FAIL_DIR"
    cp -r "$ROOT_DIR/src" "$FAIL_DIR/src" 2>/dev/null || true
    git -C "$ROOT_DIR" log -1 --format="%H %s" > "$FAIL_DIR/death_log.txt" 2>/dev/null || true

    echo "Archived failed generation to $FAIL_DIR"

    echo $((GEN + 1)) > "$GENERATION_FILE"
    NEXT_GEN=$((GEN + 1))
    echo ""
    echo "Generation counter incremented to $NEXT_GEN."
    echo "Run again to spawn Generation $NEXT_GEN from the last healthy state:"
    echo "  bash evolver.sh"
fi
