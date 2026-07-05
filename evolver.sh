#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
POPULATION_SIZE=4
GENERATION_FILE="$ROOT_DIR/logs/current_gen.txt"

[ ! -f "$GENERATION_FILE" ] && echo "1" > "$GENERATION_FILE"
GEN=$(cat "$GENERATION_FILE")

GEN_DIR="$ROOT_DIR/generations/gen_$GEN"
ARCHIVE_DIR="$ROOT_DIR/logs/archived_agents"
BASE_CODE="$ROOT_DIR/base_code"
BASE_TEMPLATE="$ROOT_DIR/.opencode/agents/base_template.md"

mkdir -p "$GEN_DIR" "$ARCHIVE_DIR"

if [ "$GEN" -eq 1 ]; then
    PARENT_AGENT="$BASE_TEMPLATE"
else
    PREV_GEN=$((GEN - 1))
    PARENT_AGENT="$ARCHIVE_DIR/gen_${PREV_GEN}_winner.md"
fi

echo "================================================================================"
echo "  EVOLUTIONARY SWARM — Generation $GEN (Population: $POPULATION_SIZE)"
echo "================================================================================"
echo ""

# ============================================================================
# STEP 1: BREED & SPAWN THE POPULATION (Parallel)
# ============================================================================
for i in $(seq 1 $POPULATION_SIZE); do
    (
        CHILD_DIR="$GEN_DIR/child_$i"
        mkdir -p "$CHILD_DIR/.opencode/agents"

        cp -r "$BASE_CODE"/* "$CHILD_DIR/"
        cp "$ROOT_DIR/.gitignore" "$CHILD_DIR/" 2>/dev/null || true

        MUTATED_AGENT="$CHILD_DIR/.opencode/agents/dct-evolver.md"

        echo "[Child $i] Breeding unique genome..."
        opencode \
            --model opencode-go/deepseek-v4-flash \
            run \
            --thinking \
            "Read the template at '$PARENT_AGENT' and the benchmark history at '$ROOT_DIR/logs/benchmark_history.md' if it exists.
             Output a UNIQUE mutated version of this OpenCode agent markdown file. Write it to '$MUTATED_AGENT'.
             Use a temperature between 0.3-0.8. Tweak the frontmatter parameters (temperature, maxSteps 20-60).
             Rephrase optimization strategies differently from the parent — try a DIFFERENT algorithmic angle than siblings.
             YOU MUST preserve the exact YAML syntax between --- delimiters. NEVER set permission to anything other than 'allow'. NEVER disable bash.
             Output ONLY the raw markdown content." \
            > "$CHILD_DIR/mutation.log" 2>&1

        echo "[Child $i] Starting lifecycle..."
        opencode \
            --model opencode-go/deepseek-v4-flash \
            --agent dct-evolver \
            --dir "$CHILD_DIR" \
            run \
            --thinking \
            "You are Child $i of Generation $GEN in an evolutionary swarm.
             Your task:
             1) Read '$CHILD_DIR/src/dct_engine.py' and '$CHILD_DIR/tests/test_dct_engine.py' to understand the codebase.
             2) Read '$ROOT_DIR/logs/benchmark_history.md' to see prior generation results.
             3) Optimize the DCT engine for maximum speed. Apply your unique mutation strategy.
             4) Run 'cd $CHILD_DIR && python3 -m unittest tests.test_dct_engine -v' to confirm all tests pass.
             5) Benchmark 10000 iterations of idct_2d and write ONLY the per-iteration ms value to '$CHILD_DIR/fitness.score' (plain number, e.g., 0.085)." \
            > "$CHILD_DIR/lifecycle.log" 2>&1

        echo "[Child $i] Running final fitness evaluation..."
        if (cd "$CHILD_DIR" && python3 -m unittest tests.test_dct_engine -v > test_output.log 2>&1); then
            python3 -c "
import time
import sys
sys.path.insert(0, '$CHILD_DIR')
from src.dct_engine import idct_2d
block = [[float(i * j % 256 - 128) for j in range(8)] for i in range(8)]
N = 10000
for _ in range(100):
    idct_2d(block)
start = time.perf_counter()
for _ in range(N):
    idct_2d(block)
elapsed = time.perf_counter() - start
score = (elapsed / N) * 1000
with open('$CHILD_DIR/fitness.score', 'w') as f:
    f.write(f'{score:.6f}')
" 2>/dev/null
            SCORE=$(cat "$CHILD_DIR/fitness.score" 2>/dev/null || echo "999")
            echo "  [Child $i] SURVIVED — Score: ${SCORE}ms/iter"
        else
            echo "999.9" > "$CHILD_DIR/fitness.score"
            SCORE=$(cat "$CHILD_DIR/fitness.score")
            echo "  [Child $i] DIED — Tests failed"
            cp "$CHILD_DIR/test_output.log" "$CHILD_DIR/death_test.log" 2>/dev/null || true
        fi
    ) &
done

echo "[*] Waiting for population lifecycles to conclude..."
wait

# ============================================================================
# STEP 2: NATURAL SELECTION
# ============================================================================
echo ""
echo "--- Natural Selection ---"

BEST_SCORE="999.9"
WINNER_INDEX=0

for i in $(seq 1 $POPULATION_SIZE); do
    SCORE_FILE="$GEN_DIR/child_$i/fitness.score"
    if [ -f "$SCORE_FILE" ]; then
        SCORE=$(cat "$SCORE_FILE")
        echo "  Child $i: $SCORE ms/iter"
        if (( $(echo "$SCORE < $BEST_SCORE" | bc -l) )); then
            BEST_SCORE=$SCORE
            WINNER_INDEX=$i
        fi
    else
        echo "  Child $i: NO SCORE (dead)"
    fi
done

if [ "$WINNER_INDEX" -eq 0 ] || [ "$BEST_SCORE" == "999.9" ]; then
    echo ""
    echo "EXTINCTION: No viable survivor in Generation $GEN."
    echo "Re-running generation..."
    exit 1
fi

WINNER_DIR="$GEN_DIR/child_$WINNER_INDEX"

echo ""
echo ">>> WINNER: Child $WINNER_INDEX — ${BEST_SCORE}ms/iter"
echo ""

# ============================================================================
# STEP 3: CONSOLIDATION — Promote winner to baseline
# ============================================================================
rm -rf "$BASE_CODE"/*
cp -r "$WINNER_DIR"/* "$BASE_CODE/" 2>/dev/null || true
rm -f "$BASE_CODE/fitness.score" "$BASE_CODE/bench.log" "$BASE_CODE/lifecycle.log" "$BASE_CODE/mutation.log" "$BASE_CODE/test_output.log" "$BASE_CODE/death_test.log" "$BASE_CODE/.gitignore" 2>/dev/null || true

cp "$WINNER_DIR/.opencode/agents/dct-evolver.md" "$ARCHIVE_DIR/gen_${GEN}_winner.md"

echo "| Gen $GEN | Child $WINNER_INDEX | ${BEST_SCORE}ms | $(date -Iseconds) |" >> "$ROOT_DIR/logs/benchmark_history.md"

# ============================================================================
# STEP 4: GIT & GITHUB SYNC
# ============================================================================
git -C "$ROOT_DIR" add -A

if ! git -C "$ROOT_DIR" diff --cached --quiet; then
    BRANCH_NAME="evolution/gen-${GEN}-winner"
    git -C "$ROOT_DIR" commit -m "evolution(gen-$GEN): winner child-$WINNER_INDEX at ${BEST_SCORE}ms/iter"
    git -C "$ROOT_DIR" branch -f "$BRANCH_NAME"
    git -C "$ROOT_DIR" checkout "$BRANCH_NAME"

    REMOTE=$(git -C "$ROOT_DIR" remote get-url origin 2>/dev/null || echo "")
    if [ -n "$REMOTE" ]; then
        git -C "$ROOT_DIR" push origin "$BRANCH_NAME" 2>/dev/null || echo "Warning: Could not push to remote"
        gh pr create \
            --title "Evolution Gen $GEN Winner — ${BEST_SCORE}ms/iter" \
            --body "Child $WINNER_INDEX won Generation $GEN with ${BEST_SCORE}ms/iter." \
            --base main \
            2>/dev/null || echo "Warning: Could not create PR"
    fi
fi

echo $((GEN + 1)) > "$GENERATION_FILE"
echo ""
echo "Generation $GEN consolidated. Ready for Generation $((GEN + 1)):"
echo "  bash evolver.sh"
