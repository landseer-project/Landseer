#!/usr/bin/env python3
"""
ML Defense Pipeline - Entry Point
"""
import argparse
import sys
from main_pipeline import DefensePipeline


def main():
    parser = argparse.ArgumentParser(description="ML Defense Pipeline")
    parser.add_argument("--config", "-c", type=str,
                        help="Path to configuration JSON for preconfigured mode")
    args = parser.parse_args()

    pipeline = DefensePipeline(args.config)
    pipeline.run()

if __name__ == "__main__":
    main()
