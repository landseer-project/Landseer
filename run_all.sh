#!/bin/bash
set -euo pipefail
shopt -s nullglob

PIPELINE_DIR="/share/landseer/workspace-ayushi/Landseer/configs/configs_by_in_tool"
ATTACK_DIR="configs/attack"
LOG_DIR="run_logs"
MAIN_SCRIPT="src/landseer_pipeline/main.py"

mkdir -p "$LOG_DIR"

pipeline_configs=($(find "$PIPELINE_DIR" -maxdepth 1 -name "*.yaml" -type f | sort))
attack_configs=($(find "$ATTACK_DIR" -maxdepth 1 -name "*.yaml" -type f | sort))

total_runs=0
success_count=0
fail_count=0
failed_runs=()

for pipeline_config in "${pipeline_configs[@]}"; do
  for attack_config in "${attack_configs[@]}"; do
    config_name=$(basename "$pipeline_config" .yaml)
    attack_name=$(basename "$attack_config" .yaml)
    log_base="${config_name}__${attack_name}"
    log_file="$LOG_DIR/${log_base}.log"
    success_flag="$LOG_DIR/${log_base}.success"
    fail_flag="$LOG_DIR/${log_base}.fail"

    # Skip if already marked as success
    if [[ -f "$success_flag" ]]; then
      echo "✔ Skipping (already successful): $log_base"
      continue
    fi

    echo "▶ Running: $log_base"
    echo "  Log: $log_file"

    # Run and capture success/failure
    if python3 -m landseer_pipeline.main -c "$pipeline_config" -a "$attack_config" > "$log_file" 2>&1; then
      touch "$success_flag"
      echo "✔ SUCCESS: $log_base"
      ((success_count++))
    else
      touch "$fail_flag"
      echo "✖ FAILED: $log_base"
      failed_runs+=("$log_base")
      ((fail_count++))
    fi

    ((total_runs++))
  done
done

# Summary
echo ""
echo "================= SUMMARY ================="
echo "Total Runs     : $total_runs"
echo "Successful     : $success_count"
echo "Failed         : $fail_count"

if ((fail_count > 0)); then
  echo ""
  echo "❌ Failed Runs:"
  for run in "${failed_runs[@]}"; do
    echo " - $run"
  done
fi

echo ""
echo "💡 Check $LOG_DIR/*.log for details."
