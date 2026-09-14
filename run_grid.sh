#!/usr/bin/env bash
#
# run_grid.sh — Run the full training grid sequentially:
#   {models} x {trainers} x {reward combos}
#
# Each run -> train_asset.py with a consistent output name:
#   runs/<model>_<trainer>_<reward>   (final adapter: ...-Final)
# Per-run stdout/stderr -> logs/<model>_<trainer>_<reward>.log
# One failing run does NOT abort the grid; a summary is printed at the end.
#
# Usage examples:
#   # local end-to-end test (1B only, CPU, smoke):
#   MODELS="llama-1b" DEVICE=cpu SMOKE=1 PYTHON="F:/PycharmProjects/GRPOSS/.venv/Scripts/python.exe" ./run_grid.sh
#
#   # full grid on the big machine (GPU):
#   ./run_grid.sh
#
#   # just print what would run, without running:
#   DRY_RUN=1 ./run_grid.sh
#
set -u
cd "$(dirname "$0")"   # run from the script's directory (repo root)

# ---------------- Config (env-overridable) ----------------
SCRIPT="train_asset.py"
OUT_ROOT="${OUT_ROOT:-runs}"
LOG_DIR="${LOG_DIR:-logs}"
DEVICE="${DEVICE:-cuda}"          # cuda (production 4-bit) | cpu (no-quant test)
SMOKE="${SMOKE:-0}"               # 1 -> add -s (tiny subset, few steps)
DRY_RUN="${DRY_RUN:-0}"           # 1 -> print commands only

# Which axes to sweep (space-separated; override via env).
MODELS=(${MODELS:-llama-1b llama-3b})
TRAINERS=(${TRAINERS:-grpo rloo})

# Reward combos as "shortname|reward_func_arg". Edit/trim as needed.
REWARDS=(
  "sari|sari_reward_func"
  "mb|meaningBert_reward_func"
  "sari-mb|sari_reward_func,meaningBert_reward_func"
)

# Python interpreter: honor $PYTHON, else search this dir and parents for a .venv, else 'python'.
# (The venv may live at the project root, several levels above the scripts dir.)
if [ -z "${PYTHON:-}" ]; then
  d="$(pwd)"
  for _ in 1 2 3 4 5; do
    if   [ -x "$d/.venv/bin/python" ];         then PYTHON="$d/.venv/bin/python"; break        # Linux venv
    elif [ -x "$d/.venv/Scripts/python.exe" ]; then PYTHON="$d/.venv/Scripts/python.exe"; break # Windows venv
    fi
    d="$(dirname "$d")"
  done
  PYTHON="${PYTHON:-python}"
fi
# ----------------------------------------------------------

# model key -> short label used in folder/log names
declare -A MSHORT=( [llama-1b]=llama1b [llama-3b]=llama3b )

mkdir -p "$OUT_ROOT" "$LOG_DIR"
SMOKE_FLAG=""; [ "$SMOKE" = "1" ] && SMOKE_FLAG="-s"

total=0; ok=0; fail=0
declare -a SUMMARY=()
start_all=$(date +%s)

echo "=============================================================="
echo " GRID: models=[${MODELS[*]}] trainers=[${TRAINERS[*]}] rewards=[$(for r in "${REWARDS[@]}"; do printf '%s ' "${r%%|*}"; done)]"
echo " device=$DEVICE  smoke=$SMOKE  python=$PYTHON  out=$OUT_ROOT/  dry_run=$DRY_RUN"
echo "=============================================================="

for m in "${MODELS[@]}"; do
  for t in "${TRAINERS[@]}"; do
    for rc in "${REWARDS[@]}"; do
      rname="${rc%%|*}"; rarg="${rc##*|}"
      mshort="${MSHORT[$m]:-$m}"
      run="${mshort}_${t}_${rname}"
      out="$OUT_ROOT/$run"
      log="$LOG_DIR/${run}.log"
      total=$((total+1))

      cmd=("$PYTHON" "$SCRIPT" -o "$out" -r "$rarg" -m "$m" -t "$t" -d "$DEVICE")
      [ -n "$SMOKE_FLAG" ] && cmd+=("$SMOKE_FLAG")

      echo
      echo "[$(date '+%H:%M:%S')] RUN $total/$(( ${#MODELS[@]} * ${#TRAINERS[@]} * ${#REWARDS[@]} )): $run"
      echo "    ${cmd[*]}"
      echo "    log -> $log"

      if [ "$DRY_RUN" = "1" ]; then
        SUMMARY+=("DRY   $run"); continue
      fi

      "${cmd[@]}" > "$log" 2>&1
      code=$?
      if [ $code -eq 0 ]; then
        ok=$((ok+1));  SUMMARY+=("OK    $run")
        echo "[$(date '+%H:%M:%S')] OK    $run"
      else
        fail=$((fail+1)); SUMMARY+=("FAIL  $run (exit=$code, see $log)")
        echo "[$(date '+%H:%M:%S')] FAIL  $run (exit=$code) -> tail:"
        tail -n 5 "$log" | sed 's/^/        /'
      fi
    done
  done
done

end_all=$(date +%s)
echo
echo "===================== GRID SUMMARY ====================="
printf '  %s\n' "${SUMMARY[@]}"
echo "-------------------------------------------------------"
echo "  total=$total  ok=$ok  fail=$fail  elapsed=$(( end_all - start_all ))s"
echo "  outputs in: $OUT_ROOT/   logs in: $LOG_DIR/"
echo "======================================================="

[ $fail -eq 0 ]
