#!/usr/bin/env python3
"""
ML Defense Pipeline - Entry Point
"""
import argparse
import sys
from pipeline import DefensePipeline

def main():
    parser = argparse.ArgumentParser(description="Modular ML Defense Pipeline")
    parser.add_argument("--config", "-c", type=str, help="Path to configuration JSON for preconfigured mode")
    args = parser.parse_args()
    
    try:
        pipeline = DefensePipeline(args.config)
        pipeline.run()
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()