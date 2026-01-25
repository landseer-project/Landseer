#!/usr/bin/env python3
"""
Deviation Analysis for Landseer Pipeline Results

This script analyzes multiple pipeline runs to determine:
1. Whether there is deviation in results across runs
2. Statistical measures (mean, median, std, min, max, coefficient of variation)
3. Recommendations for paper reporting
4. Tableau-friendly CSV exports for visualization

Usage:
    python analyze_deviation.py <deviation_test_dir>
    
Example:
    python analyze_deviation.py ../deviation_test_20260109120000
    
Output Files:
    - deviation_summary.csv     (aggregate stats per combination + metric)
    - deviation_raw_data.csv    (individual datapoints for Tableau)
    - deviation_report.txt      (human-readable analysis)
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import statistics
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class MetricStats:
    """Statistics for a single metric across runs."""
    combination: str
    metric_name: str
    values: List[float] = field(default_factory=list)
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    cv: float = 0.0  # Coefficient of variation (std/mean * 100)
    range_pct: float = 0.0  # (max-min)/mean * 100
    n_samples: int = 0


# Metrics to analyze
METRICS_OF_INTEREST = [
    "acc_train_clean",
    "acc_test_clean", 
    "pgd_acc",
    "carlini_acc",
    "ood_auc",
    "fingerprinting",
    "asr",
    "privacy_epsilon",
    "dp_accuracy",
    "watermark_accuracy",
    "mia_auc",
    "eps_estimate",
    "total_duration"
]

# Pipeline stage columns for Tableau grouping
STAGE_COLUMNS = ["pre_training", "in_training", "post_training", "deployment"]


def find_results_csv(run_dir: Path) -> Optional[Path]:
    """Find the results_combinations.csv in a run directory."""
    for csv_file in run_dir.rglob("results_combinations.csv"):
        return csv_file
    return None


def parse_results_csv(csv_path: Path) -> List[Dict]:
    """Parse a results CSV file and return all rows with their data."""
    results = []
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Only include successful combinations with valid metrics
                if row.get('combination_status') == 'success':
                    results.append(dict(row))
    except Exception as e:
        print(f"Warning: Error parsing {csv_path}: {e}")
    
    return results


def collect_all_runs(deviation_dir: Path) -> Tuple[List[Dict], List[str]]:
    """
    Collect results from all runs in the deviation test directory.
    Returns: (list of all row dicts with run_id added, list of run_ids)
    """
    all_rows = []
    run_ids = []
    
    # Find all run directories
    run_dirs = sorted([d for d in deviation_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    
    for run_dir in run_dirs:
        run_id = run_dir.name  # e.g., "run_1", "run_2"
        run_ids.append(run_id)
        
        csv_path = find_results_csv(run_dir)
        if csv_path:
            rows = parse_results_csv(csv_path)
            for row in rows:
                row['run_id'] = run_id
                all_rows.append(row)
            print(f"  {run_id}: Found {len(rows)} successful combinations")
        else:
            print(f"  {run_id}: No results CSV found")
    
    return all_rows, run_ids


def compute_statistics(combination: str, metric_name: str, values: List[float]) -> Optional[MetricStats]:
    """Compute statistics for a list of metric values."""
    if not values or len(values) < 2:
        return None
    
    mean = statistics.mean(values)
    median = statistics.median(values)
    std = statistics.stdev(values)
    min_val = min(values)
    max_val = max(values)
    
    # Coefficient of variation (as percentage)
    cv = (std / mean * 100) if mean != 0 else 0
    
    # Range as percentage of mean
    range_pct = ((max_val - min_val) / mean * 100) if mean != 0 else 0
    
    return MetricStats(
        combination=combination,
        metric_name=metric_name,
        values=values,
        mean=mean,
        median=median,
        std=std,
        min_val=min_val,
        max_val=max_val,
        cv=cv,
        range_pct=range_pct,
        n_samples=len(values)
    )


def analyze_deviation(deviation_dir: Path) -> Tuple[Dict[str, Dict[str, MetricStats]], List[Dict]]:
    """
    Analyze deviation across all runs for each combination and metric.
    Returns: (stats_by_combo, all_raw_rows)
    """
    print(f"\nAnalyzing results in: {deviation_dir}")
    print("=" * 60)
    
    all_rows, run_ids = collect_all_runs(deviation_dir)
    
    if not all_rows:
        print("ERROR: No results found!")
        return {}, []
    
    print(f"\nFound {len(all_rows)} total datapoints across {len(run_ids)} runs")
    
    # Group values by (combination, metric)
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    
    for row in all_rows:
        combo = row.get('combination', 'unknown')
        for metric in METRICS_OF_INTEREST:
            try:
                val = float(row.get(metric, -1))
                if val != -1:  # -1 is the "not computed" marker
                    grouped[combo][metric].append(val)
            except (ValueError, TypeError):
                pass
    
    # Compute statistics for each (combination, metric) pair
    stats_by_combo: Dict[str, Dict[str, MetricStats]] = {}
    
    for combo, metrics in grouped.items():
        stats_by_combo[combo] = {}
        for metric_name, values in metrics.items():
            stat = compute_statistics(combo, metric_name, values)
            if stat:
                stats_by_combo[combo][metric_name] = stat
    
    return stats_by_combo, all_rows


def print_report(stats_by_combo: Dict[str, Dict[str, MetricStats]], output_file: Optional[Path] = None):
    """Print deviation analysis report."""
    lines = []
    
    lines.append("\n" + "=" * 100)
    lines.append("DEVIATION ANALYSIS REPORT")
    lines.append("=" * 100)
    
    if not stats_by_combo:
        lines.append("No metrics found to analyze.")
        output = "\n".join(lines)
        print(output)
        return
    
    # Collect all stats for summary
    all_stats: List[MetricStats] = []
    for combo_stats in stats_by_combo.values():
        all_stats.extend(combo_stats.values())
    
    # Summary table header
    lines.append(f"\n{'Combination':<12} {'Metric':<20} {'Mean':>10} {'Median':>10} {'Std':>10} {'CV(%)':>8} {'Min':>10} {'Max':>10} {'N':>5}")
    lines.append("-" * 105)
    
    # Sort by combination, then by CV (most variable first)
    for combo in sorted(stats_by_combo.keys()):
        combo_stats = stats_by_combo[combo]
        sorted_metrics = sorted(combo_stats.values(), key=lambda x: x.cv, reverse=True)
        
        for stat in sorted_metrics:
            lines.append(
                f"{stat.combination:<12} {stat.metric_name:<20} {stat.mean:>10.4f} {stat.median:>10.4f} "
                f"{stat.std:>10.4f} {stat.cv:>8.2f} {stat.min_val:>10.4f} {stat.max_val:>10.4f} {stat.n_samples:>5}"
            )
        lines.append("")  # Blank line between combos
    
    # Summary and recommendations
    lines.append("=" * 100)
    lines.append("OVERALL SUMMARY")
    lines.append("=" * 100)
    
    # Categorize metrics by variability
    high_var = [s for s in all_stats if s.cv > 5]
    med_var = [s for s in all_stats if 1 < s.cv <= 5]
    low_var = [s for s in all_stats if s.cv <= 1]
    
    lines.append(f"\nTotal combinations analyzed: {len(stats_by_combo)}")
    lines.append(f"Total metric datapoints: {len(all_stats)}")
    
    if high_var:
        lines.append(f"\n⚠️  HIGH VARIABILITY (CV > 5%): {len(high_var)} metrics")
        for s in sorted(high_var, key=lambda x: x.cv, reverse=True)[:10]:
            lines.append(f"   - {s.combination}/{s.metric_name}: CV={s.cv:.2f}%")
        lines.append("   RECOMMENDATION: Run multiple times and report mean ± std")
    
    if med_var:
        lines.append(f"\n📊 MODERATE VARIABILITY (1% < CV ≤ 5%): {len(med_var)} metrics")
        lines.append("   RECOMMENDATION: Consider running 3-5 times for robustness")
    
    if low_var:
        lines.append(f"\n✅ LOW VARIABILITY (CV ≤ 1%): {len(low_var)} metrics")
        lines.append("   RECOMMENDATION: Single run may be sufficient")
    
    # Overall recommendation
    lines.append("\n" + "-" * 100)
    if high_var:
        pct_high = len(high_var) / len(all_stats) * 100
        lines.append(f"📋 OVERALL: {pct_high:.1f}% of metrics have high variability.")
        lines.append("   Suggest: Run 5-10 times, report mean ± std for accuracy metrics")
    elif med_var:
        lines.append("📋 OVERALL: Some deviation detected. 3-5 runs recommended for key metrics.")
    else:
        lines.append("📋 OVERALL: Results are stable. Single run appears sufficient.")
    lines.append("-" * 100)
    
    output = "\n".join(lines)
    print(output)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(output)
        print(f"\nReport saved to: {output_file}")


def export_summary_csv(stats_by_combo: Dict[str, Dict[str, MetricStats]], output_csv: Path):
    """Export summary statistics to CSV (one row per combination+metric)."""
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'combination', 'metric', 'mean', 'median', 'std', 'cv_pct', 
            'min', 'max', 'range_pct', 'n_samples'
        ])
        
        for combo in sorted(stats_by_combo.keys()):
            for metric_name, stat in sorted(stats_by_combo[combo].items()):
                writer.writerow([
                    stat.combination,
                    stat.metric_name,
                    f"{stat.mean:.6f}",
                    f"{stat.median:.6f}",
                    f"{stat.std:.6f}",
                    f"{stat.cv:.4f}",
                    f"{stat.min_val:.6f}",
                    f"{stat.max_val:.6f}",
                    f"{stat.range_pct:.4f}",
                    stat.n_samples
                ])
    
    print(f"Summary statistics exported to: {output_csv}")


def export_raw_data_csv(all_rows: List[Dict], output_csv: Path):
    """
    Export raw datapoints to CSV (one row per run+combination).
    This is the Tableau-friendly format with all individual datapoints.
    """
    if not all_rows:
        print("No raw data to export")
        return
    
    # Define column order
    columns = [
        'run_id', 'combination', 
        'pre_training', 'in_training', 'post_training', 'deployment',
        'dataset_name', 'dataset_type', 'combination_status'
    ] + METRICS_OF_INTEREST
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        
        for row in sorted(all_rows, key=lambda x: (x.get('run_id', ''), x.get('combination', ''))):
            writer.writerow(row)
    
    print(f"Raw datapoints exported to: {output_csv} ({len(all_rows)} rows)")


def export_pivot_csv(stats_by_combo: Dict[str, Dict[str, MetricStats]], output_csv: Path):
    """
    Export pivot table format: combinations as rows, metrics as columns.
    Shows mean (median) ± std for each cell.
    """
    if not stats_by_combo:
        print("No data for pivot table")
        return
    
    # Get all metrics present
    all_metrics = set()
    for combo_stats in stats_by_combo.values():
        all_metrics.update(combo_stats.keys())
    all_metrics = sorted(all_metrics)
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header row
        header = ['combination'] + [f"{m}_mean" for m in all_metrics] + [f"{m}_median" for m in all_metrics] + [f"{m}_std" for m in all_metrics] + [f"{m}_cv" for m in all_metrics]
        writer.writerow(header)
        
        for combo in sorted(stats_by_combo.keys()):
            row = [combo]
            # Means
            for m in all_metrics:
                if m in stats_by_combo[combo]:
                    row.append(f"{stats_by_combo[combo][m].mean:.6f}")
                else:
                    row.append("")
            # Medians
            for m in all_metrics:
                if m in stats_by_combo[combo]:
                    row.append(f"{stats_by_combo[combo][m].median:.6f}")
                else:
                    row.append("")
            # Stds
            for m in all_metrics:
                if m in stats_by_combo[combo]:
                    row.append(f"{stats_by_combo[combo][m].std:.6f}")
                else:
                    row.append("")
            # CVs
            for m in all_metrics:
                if m in stats_by_combo[combo]:
                    row.append(f"{stats_by_combo[combo][m].cv:.4f}")
                else:
                    row.append("")
            writer.writerow(row)
    
    print(f"Pivot table exported to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze deviation across multiple pipeline runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "deviation_dir",
        type=Path,
        help="Directory containing multiple run results"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: same as deviation_dir)"
    )
    
    args = parser.parse_args()
    
    if not args.deviation_dir.exists():
        print(f"ERROR: Directory not found: {args.deviation_dir}")
        sys.exit(1)
    
    output_dir = args.output_dir or args.deviation_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze deviation
    stats_by_combo, all_rows = analyze_deviation(args.deviation_dir)
    
    if not stats_by_combo:
        print("No data to analyze")
        sys.exit(1)
    
    # Export all outputs
    print("\n" + "=" * 60)
    print("EXPORTING RESULTS")
    print("=" * 60)
    
    # 1. Summary statistics CSV (for quick viewing)
    export_summary_csv(stats_by_combo, output_dir / "deviation_summary.csv")
    
    # 2. Raw datapoints CSV (for Tableau - individual values)
    export_raw_data_csv(all_rows, output_dir / "deviation_raw_data.csv")
    
    # 3. Pivot table CSV (combinations as rows, metrics as columns)
    export_pivot_csv(stats_by_combo, output_dir / "deviation_pivot.csv")
    
    # 4. Human-readable report
    print_report(stats_by_combo, output_dir / "deviation_report.txt")
    
    print("\n" + "=" * 60)
    print("DONE! Files created:")
    print("=" * 60)
    print(f"  📊 {output_dir / 'deviation_summary.csv'}     - Aggregate stats (mean, median, std, CV)")
    print(f"  📈 {output_dir / 'deviation_raw_data.csv'}    - Individual datapoints (for Tableau)")
    print(f"  📋 {output_dir / 'deviation_pivot.csv'}       - Pivot table format")
    print(f"  📝 {output_dir / 'deviation_report.txt'}      - Human-readable analysis")
    print("")
    print("To view in Tableau:")
    print("  1. Open Tableau Desktop")
    print("  2. Connect to Text File → Select 'deviation_raw_data.csv'")
    print("  3. Create visualizations with run_id, combination, and metric columns")


if __name__ == "__main__":
    main()
