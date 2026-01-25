#!/bin/bash

# Run comb_020 multiple times to check for variability in results
# Usage: ./run_comb_020_multiple.sh

COMBO_ID="comb_010"
NUM_RUNS=10
CONFIG="configs/pipeline/no_attack/trades.yaml"
ATTACK_CONFIG="configs/attack/test_config_1.yaml"
PYTHON_BIN="/home/ayushi/.venv/bin/python"

echo "=========================================="
echo "Running ${COMBO_ID} ${NUM_RUNS} times"
echo "=========================================="
echo ""

for i in $(seq 1 $NUM_RUNS); do
    echo "=========================================="
    echo "Run $i of $NUM_RUNS"
    echo "=========================================="
    
    # Clean cache before each run
    echo "Cleaning cache..."
    rm -rf cache/artifact_store/*
    
    # Run the pipeline
    echo "Starting pipeline..."
    $PYTHON_BIN -m landseer_pipeline.main \
        -c $CONFIG \
        -a $ATTACK_CONFIG \
        --no-cache \
        --combo-id $COMBO_ID \
        2>&1 | tee "run_${COMBO_ID}_${i}.log"
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Run $i completed successfully"
    else
        echo "✗ Run $i failed with exit code $EXIT_CODE"
    fi
    
    echo ""
done

echo "=========================================="
echo "All runs completed"
echo "=========================================="
echo "Logs saved as: run_${COMBO_ID}_1.log through run_${COMBO_ID}_${NUM_RUNS}.log"
