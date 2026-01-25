#!/bin/bash

set -aux

# =============================================================================
# Run Pipeline Multiple Times for Deviation Analysis (ALL COMBINATIONS)
# =============================================================================
# Purpose: Run the same pipeline configuration multiple times with no cache
#          to determine if there is deviation in results across runs.
#          This helps decide if multiple runs are needed for paper reporting.
#
# Usage:
#   ./run_deviation_test.sh              # Run all combinations 10 times
#   ./run_deviation_test.sh 5            # Run all combinations 5 times
#   ./run_deviation_test.sh 3 comb_010   # Run specific combo 3 times
# =============================================================================
echo "Running deviation test for DNN Watermarking pipeline"

set -e

echo "Running deviation test for DNN Watermarking pipeline"
# Configuration
NUM_RUNS="${1:-10}"
COMBO_ID="${2:-}"  # Empty means run ALL combinations
CONFIG="/share/landseer/workspace-ayushi/Landseer/configs/pipeline/dnn_watermarking.yaml"
ATTACK_CONFIG="configs/attack/fairness.yaml"

# Use poetry run to ensure correct environment
PYTHON_BIN="poetry run python"

# Create output directory for this deviation test
TIMESTAMP=$(date +%Y%m%d%H%M%S)
DEVIATION_DIR="deviation_test_${TIMESTAMP}"
mkdir -p "$DEVIATION_DIR"

echo "=========================================="
echo "Deviation Test for Landseer Pipeline"
echo "=========================================="
echo "Configuration:"
echo "  - Pipeline config: $CONFIG"
echo "  - Attack config:   $ATTACK_CONFIG"
if [ -z "$COMBO_ID" ]; then
    echo "  - Combinations:    ALL (no filter)"
else
    echo "  - Combo ID:        $COMBO_ID"
fi
echo "  - Number of runs:  $NUM_RUNS"
echo "  - Output dir:      $DEVIATION_DIR"
echo "  - Python:          $PYTHON_BIN"
echo "=========================================="
echo ""

# Track timing
START_TIME=$(date +%s)

for i in $(seq 1 $NUM_RUNS); do
    RUN_START=$(date +%s)
    
    echo "=========================================="
    echo "Run $i of $NUM_RUNS ($(date))"
    echo "=========================================="
    
    # Create run-specific results directory
    RUN_RESULTS_DIR="${DEVIATION_DIR}/run_${i}"
    mkdir -p "$RUN_RESULTS_DIR"
    
    # Clean cache before each run to ensure fresh computation
    echo "Cleaning cache/artifact_store..."
    rm -rf cache/artifact_store/* 2>/dev/null || true
    
    # Build command - optionally include combo-id
    CMD="$PYTHON_BIN -m landseer_pipeline.main \
        -c $CONFIG \
        -a $ATTACK_CONFIG \
        --no-cache \
        --results-dir $RUN_RESULTS_DIR"
    
    if [ -n "$COMBO_ID" ]; then
        CMD="$CMD --combo-id $COMBO_ID"
    fi
    
    # Run the pipeline
    echo "Starting pipeline run $i..."
    eval "$CMD" 2>&1 | tee "${DEVIATION_DIR}/run_${i}.log"
    
    EXIT_CODE=${PIPESTATUS[0]}
    RUN_END=$(date +%s)
    RUN_DURATION=$((RUN_END - RUN_START))
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Run $i completed successfully in ${RUN_DURATION}s"
    else
        echo "✗ Run $i failed with exit code $EXIT_CODE (after ${RUN_DURATION}s)"
    fi
    
    echo ""
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo "=========================================="
echo "All $NUM_RUNS runs completed in ${TOTAL_DURATION}s"
echo "=========================================="
echo ""
echo "Results saved in: $DEVIATION_DIR"
echo ""
echo "To analyze results:"
echo "  cd helper_scripts && python analyze_deviation.py ../$DEVIATION_DIR"
echo ""
echo "Output files will include:"
echo "  - deviation_summary.csv     (aggregate stats per combo+metric)"
echo "  - deviation_raw_data.csv    (individual datapoints for Tableau)"
echo "  - deviation_report.txt      (human-readable analysis)"
