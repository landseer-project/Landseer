#!/usr/bin/env python3
import json
import itertools
import subprocess
import time
import os
import logging
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re
import argparse

# Setup logging
LOG_DIR = "combination_test_logs"
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
log_file = os.path.join(LOG_DIR, f"test-run-{timestamp}.log")

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Metrics patterns
# Update these patterns to match your actual output metrics
METRIC_PATTERNS = {
    'accuracy': r'Accuracy:\s*([\d.]+)',
    'precision': r'Precision:\s*([\d.]+)',
    'recall': r'Recall:\s*([\d.]+)',
    'f1_score': r'F1 score:\s*([\d.]+)',
    'latency': r'Latency:\s*([\d.]+)\s*ms',
    'throughput': r'Throughput:\s*([\d.]+)\s*req/s',
    'memory_usage': r'Memory usage:\s*([\d.]+)\s*MB',
    'error_rate': r'Error rate:\s*([\d.]+)%',
    # Add any other metrics you need to extract
}


def read_config(config_path):
    """Read the configuration file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def write_config(config, config_path="config.json"):
    """Write configuration to file"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Updated config: {json.dumps(config, indent=2)}")
    return True


def generate_power_set(tools):
    """Generate all possible subsets of tools (power set)"""
    return [list(subset) for subset in itertools.chain.from_iterable(
        itertools.combinations(tools, r) for r in range(len(tools) + 1)
    )]


def generate_all_combinations(tools_by_stage):
    """Generate all possible combinations across all stages"""
    pre_stage_subsets = generate_power_set(tools_by_stage.get('preStage', []))
    during_stage_subsets = generate_power_set(
        tools_by_stage.get('duringStage', []))
    post_stage_subsets = generate_power_set(
        tools_by_stage.get('postStage', []))

    combinations = []
    for pre_tools in pre_stage_subsets:
        for during_tools in during_stage_subsets:
            for post_tools in post_stage_subsets:
                combinations.append({
                    'preStage': pre_tools,
                    'duringStage': during_tools,
                    'postStage': post_tools
                })

    return combinations


def extract_metrics(output):
    """Extract metrics from test output using regex patterns"""
    metrics = {}
    for metric_name, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, output)
        if match:
            try:
                metrics[metric_name] = float(match.group(1))
            except ValueError:
                metrics[metric_name] = match.group(1)

    return metrics


def run_test(combination):
    """Run test with the current configuration"""
    pre_tools = ', '.join(
        combination['preStage']) if combination['preStage'] else 'none'
    during_tools = ', '.join(
        combination['duringStage']) if combination['duringStage'] else 'none'
    post_tools = ', '.join(
        combination['postStage']) if combination['postStage'] else 'none'

    logger.info(f"Running test with combination:")
    logger.info(f"  Pre-stage: {pre_tools}")
    logger.info(f"  During-stage: {during_tools}")
    logger.info(f"  Post-stage: {post_tools}")

    # Create a unique log file for this combination
    combo_id = f"pre-{'_'.join(combination['preStage']) if combination['preStage'] else 'none'}-" \
        f"during-{'_'.join(combination['duringStage']) if combination['duringStage'] else 'none'}-" \
        f"post-{'_'.join(combination['postStage']) if combination['postStage'] else 'none'}"

    combo_log_file = os.path.join(LOG_DIR, f"combo-{combo_id}-{timestamp}.log")

    # Replace with your actual test command
    try:
        with open(combo_log_file, 'w') as log_f:
            result = subprocess.run(
                ['npm', 'test'],
                capture_output=True,
                text=True,
                check=False
            )

            # Write test output to combination log file
            log_f.write(f"== TEST OUTPUT ==\n")
            log_f.write(result.stdout)
            log_f.write("\n\n")

            if result.stderr:
                log_f.write(f"== ERROR OUTPUT ==\n")
                log_f.write(result.stderr)

            # Extract metrics from the output
            metrics = extract_metrics(result.stdout)
            log_f.write("\n\n== EXTRACTED METRICS ==\n")
            for metric, value in metrics.items():
                log_f.write(f"{metric}: {value}\n")

        logger.info(f"Test completed with status code: {result.returncode}")
        logger.info(f"Detailed log saved to: {combo_log_file}")

        if metrics:
            logger.info("Extracted metrics:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value}")

        return {
            'success': result.returncode == 0,
            'output': result.stdout,
            'error': result.stderr if result.returncode != 0 else None,
            'metrics': metrics,
            'log_file': combo_log_file
        }
    except Exception as e:
        logger.error(f"Error running test: {str(e)}")
        return {
            'success': False,
            'output': None,
            'error': str(e),
            'metrics': {},
            'log_file': combo_log_file
        }


def analyze_results(results):
    """Analyze test results and print summary"""
    success_count = sum(1 for r in results if r['result']['success'])
    logger.info(
        f"\nSummary: {success_count}/{len(results)} combinations passed")

    # Group by stage tools
    by_pre_stage = defaultdict(
        lambda: {'total': 0, 'success': 0, 'metrics': defaultdict(list)})
    by_during_stage = defaultdict(
        lambda: {'total': 0, 'success': 0, 'metrics': defaultdict(list)})
    by_post_stage = defaultdict(
        lambda: {'total': 0, 'success': 0, 'metrics': defaultdict(list)})

    # Create a metrics lookup table for fast lookups
    metrics_lookup = {}

    for result in results:
        pre_key = ','.join(result['combination']['preStage']) or 'none'
        during_key = ','.join(result['combination']['duringStage']) or 'none'
        post_key = ','.join(result['combination']['postStage']) or 'none'

        combination_key = f"{pre_key}|{during_key}|{post_key}"
        metrics = result['result'].get('metrics', {})
        metrics_lookup[combination_key] = metrics

        # Update counters
        by_pre_stage[pre_key]['total'] += 1
        by_during_stage[during_key]['total'] += 1
        by_post_stage[post_key]['total'] += 1

        if result['result']['success']:
            by_pre_stage[pre_key]['success'] += 1
            by_during_stage[during_key]['success'] += 1
            by_post_stage[post_key]['success'] += 1

        # Collect metrics for each stage
        for metric, value in metrics.items():
            by_pre_stage[pre_key]['metrics'][metric].append(value)
            by_during_stage[during_key]['metrics'][metric].append(value)
            by_post_stage[post_key]['metrics'][metric].append(value)

    # Print analysis
    logger.info('\nAnalysis by tool configuration:')

    logger.info('\nPre-stage tools success rates and average metrics:')
    for tools, stats in by_pre_stage.items():
        success_rate = (stats['success'] / stats['total']
                        * 100) if stats['total'] > 0 else 0
        logger.info(
            f"  {tools}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")

        if stats['metrics']:
            logger.info(f"    Average metrics:")
            for metric, values in stats['metrics'].items():
                if values:
                    avg = sum(values) / len(values)
                    logger.info(f"      {metric}: {avg:.3f}")

    logger.info('\nDuring-stage tools success rates and average metrics:')
    for tools, stats in by_during_stage.items():
        success_rate = (stats['success'] / stats['total']
                        * 100) if stats['total'] > 0 else 0
        logger.info(
            f"  {tools}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")

        if stats['metrics']:
            logger.info(f"    Average metrics:")
            for metric, values in stats['metrics'].items():
                if values:
                    avg = sum(values) / len(values)
                    logger.info(f"      {metric}: {avg:.3f}")

    logger.info('\nPost-stage tools success rates and average metrics:')
    for tools, stats in by_post_stage.items():
        success_rate = (stats['success'] / stats['total']
                        * 100) if stats['total'] > 0 else 0
        logger.info(
            f"  {tools}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")

        if stats['metrics']:
            logger.info(f"    Average metrics:")
            for metric, values in stats['metrics'].items():
                if values:
                    avg = sum(values) / len(values)
                    logger.info(f"      {metric}: {avg:.3f}")

    # Create a report for each metric
    all_metrics = set()
    for result in results:
        all_metrics.update(result['result'].get('metrics', {}).keys())

    for metric in all_metrics:
        logger.info(f"\n=== Combinations ranked by {metric} ===")

        # Sort combinations by this metric
        metric_rankings = []
        for result in results:
            if metric in result['result'].get('metrics', {}):
                combo = result['combination']
                pre_key = ','.join(combo['preStage']) or 'none'
                during_key = ','.join(combo['duringStage']) or 'none'
                post_key = ','.join(combo['postStage']) or 'none'

                metric_rankings.append({
                    'pre': pre_key,
                    'during': during_key,
                    'post': post_key,
                    'value': result['result']['metrics'][metric],
                    'success': result['result']['success']
                })

        # Sort by metric value (descending)
        metric_rankings.sort(key=lambda x: x['value'], reverse=True)

        # Display top 10
        for i, ranking in enumerate(metric_rankings[:10]):
            status = "✓" if ranking['success'] else "✗"
            logger.info(
                f"{i+1}. {status} Pre: {ranking['pre']}, During: {ranking['during']}, Post: {ranking['post']} - {metric}: {ranking['value']}")

    # Find combinations with the best metrics overall
    logger.info("\n=== Best overall combinations ===")

    # Create a scoring system - normalize and sum all metrics
    # First, gather min/max for each metric to normalize
    metric_ranges = {}
    for metric in all_metrics:
        values = [r['result']['metrics'].get(
            metric, 0) for r in results if metric in r['result'].get('metrics', {})]
        if values:
            metric_ranges[metric] = {
                'min': min(values),
                'max': max(values),
                'range': max(values) - min(values) if max(values) > min(values) else 1
            }

    # Calculate scores
    combo_scores = []
    for result in results:
        score = 0
        metrics = result['result'].get('metrics', {})

        if metrics and result['result']['success']:
            for metric, value in metrics.items():
                if metric in metric_ranges:
                    # Normalize score between 0-1 (higher is better)
                    normalized = (
                        value - metric_ranges[metric]['min']) / metric_ranges[metric]['range']
                    score += normalized

            combo = result['combination']
            pre_key = ','.join(combo['preStage']) or 'none'
            during_key = ','.join(combo['duringStage']) or 'none'
            post_key = ','.join(combo['postStage']) or 'none'

            combo_scores.append({
                'pre': pre_key,
                'during': during_key,
                'post': post_key,
                'score': score,
                'metrics': metrics
            })

    # Sort by overall score
    combo_scores.sort(key=lambda x: x['score'], reverse=True)

    # Display top 5 overall
    for i, combo in enumerate(combo_scores[:5]):
        logger.info(
            f"{i+1}. Pre: {combo['pre']}, During: {combo['during']}, Post: {combo['post']} - Overall score: {combo['score']:.2f}")
        for metric, value in combo['metrics'].items():
            logger.info(f"   {metric}: {value}")


def export_results_to_csv(results, filename):
    """Export test results to CSV file for further analysis"""
    with open(filename, 'w', newline='') as csvfile:
        # Determine all possible metrics
        all_metrics = set()
        for result in results:
            all_metrics.update(result['result'].get('metrics', {}).keys())

        # Create CSV header
        fieldnames = ['pre_stage', 'during_stage', 'post_stage',
                      'success', 'execution_time', 'log_file']
        fieldnames.extend(sorted(all_metrics))

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Write each result
        for result in results:
            combo = result['combination']
            pre_key = ','.join(combo['preStage']) or 'none'
            during_key = ','.join(combo['duringStage']) or 'none'
            post_key = ','.join(combo['postStage']) or 'none'

            # Create row
            row = {
                'pre_stage': pre_key,
                'during_stage': during_key,
                'post_stage': post_key,
                'success': result['result']['success'],
                'execution_time': result.get('execution_time', ''),
                'log_file': result['result'].get('log_file', '')
            }

            # Add metrics
            metrics = result['result'].get('metrics', {})
            for metric in all_metrics:
                row[metric] = metrics.get(metric, '')

            writer.writerow(row)

    logger.info(f"Results exported to CSV: {filename}")


def test_all_combinations(config_path="config.json"):
    """Test all possible tool combinations"""
    # Read the original config
    original_config = read_config(config_path)

    # Extract all available tools by stage
    tools_by_stage = {
        'preStage': original_config.get('preStage', []),
        'duringStage': original_config.get('duringStage', []),
        'postStage': original_config.get('postStage', [])
    }

    logger.info(f"Available tools by stage:")
    logger.info(
        f"  Pre-stage: {', '.join(tools_by_stage['preStage']) if tools_by_stage['preStage'] else 'none'}")
    logger.info(
        f"  During-stage: {', '.join(tools_by_stage['duringStage']) if tools_by_stage['duringStage'] else 'none'}")
    logger.info(
        f"  Post-stage: {', '.join(tools_by_stage['postStage']) if tools_by_stage['postStage'] else 'none'}")

    # Generate all possible combinations
    combinations = generate_all_combinations(tools_by_stage)
    logger.info(f"Generated {len(combinations)} combinations to test")

    # Store the original config to restore later
    backup_config = original_config.copy()

    # Results collection
    results = []

    # Test each combination
    for i, combination in enumerate(combinations):
        logger.info(f"\nTesting combination {i + 1}/{len(combinations)}")

        # Update config with the current combination
        updated_config = original_config.copy()
        updated_config['preStage'] = combination['preStage']
        updated_config['duringStage'] = combination['duringStage']
        updated_config['postStage'] = combination['postStage']

        # Write the updated config
        if write_config(updated_config):
            start_time = time.time()
            try:
                # Run the test with this configuration
                test_result = run_test(combination)

                # Calculate execution time
                execution_time = time.time() - start_time

                # Store the result
                results.append({
                    'combination': combination,
                    'result': test_result,
                    'execution_time': execution_time
                })

                logger.info(f"Test completed in {execution_time:.2f} seconds")

            except Exception as e:
                logger.error(f"Error testing combination: {e}")
                results.append({
                    'combination': combination,
                    'result': {'success': False, 'error': str(e)},
                    'execution_time': time.time() - start_time
                })

    # Restore the original config
    write_config(backup_config)

    # Generate results directory
    results_dir = os.path.join(LOG_DIR, f"results-{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Write results to a file
    results_filename = os.path.join(
        results_dir, "combination-test-results.json")
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Export to CSV for easier analysis
    csv_filename = os.path.join(results_dir, "combination-test-results.csv")
    export_results_to_csv(results, csv_filename)

    # Create a metrics report
    metrics_filename = os.path.join(results_dir, "metrics-report.md")
    with open(metrics_filename, 'w') as f:
        f.write(f"# Tool Combination Test Results\n\n")
        f.write(f"Test run: {timestamp}\n\n")

        # Summary section
        success_count = sum(1 for r in results if r['result']['success'])
        f.write(f"## Summary\n\n")
        f.write(f"* Total combinations tested: {len(results)}\n")
        f.write(f"* Successful combinations: {success_count}\n")
        f.write(f"* Success rate: {success_count/len(results)*100:.1f}%\n\n")

        # Metrics overview
        all_metrics = set()
        for result in results:
            all_metrics.update(result['result'].get('metrics', {}).keys())

        if all_metrics:
            f.write(f"## Metrics Collected\n\n")
            for metric in sorted(all_metrics):
                f.write(f"* {metric}\n")
            f.write("\n")

        # Best combinations for each metric
        for metric in sorted(all_metrics):
            f.write(f"## Top 5 Combinations for {metric}\n\n")
            f.write(
                f"| Rank | Pre-Stage | During-Stage | Post-Stage | {metric} | Success |\n")
            f.write(
                f"|------|-----------|--------------|------------|----------|--------|\n")

            metric_combos = []
            for result in results:
                if metric in result['result'].get('metrics', {}):
                    combo = result['combination']
                    pre_key = ','.join(combo['preStage']) or 'none'
                    during_key = ','.join(combo['duringStage']) or 'none'
                    post_key = ','.join(combo['postStage']) or 'none'

                    metric_combos.append({
                        'pre': pre_key,
                        'during': during_key,
                        'post': post_key,
                        'value': result['result']['metrics'][metric],
                        'success': result['result']['success']
                    })

            # Sort and write top 5
            metric_combos.sort(key=lambda x: x['value'], reverse=True)
            for i, combo in enumerate(metric_combos[:5]):
                success_mark = "✓" if combo['success'] else "✗"
                f.write(
                    f"| {i+1} | {combo['pre']} | {combo['during']} | {combo['post']} | {combo['value']} | {success_mark} |\n")

            f.write("\n")

    logger.info(f"\nAll combination tests completed.")
    logger.info(f"Results saved to:")
    logger.info(f"  JSON: {results_filename}")
    logger.info(f"  CSV: {csv_filename}")
    logger.info(f"  Metrics report: {metrics_filename}")

    # Analyze and display summary
    analyze_results(results)

    return results


def test_specific_combination(pre_stage, during_stage, post_stage):
    """Test a specific combination of tools"""
    original_config = read_config()
    backup_config = original_config.copy()

    # Update config with the specified combination
    updated_config = original_config.copy()
    updated_config['preStage'] = pre_stage
    updated_config['duringStage'] = during_stage
    updated_config['postStage'] = post_stage

    # Write the updated config
    if write_config(updated_config):
        combination = {
            'preStage': pre_stage,
            'duringStage': during_stage,
            'postStage': post_stage
        }

        try:
            # Run the test with this configuration
            start_time = time.time()
            test_result = run_test(combination)
            execution_time = time.time() - start_time

            logger.info(f"Test completed in {execution_time:.2f} seconds")
            logger.info(
                f"Test result: {json.dumps(test_result, indent=2, default=str)}")

            # Create a small results file just for this test
            specific_result = {
                'combination': combination,
                'result': test_result,
                'execution_time': execution_time
            }

            # Generate results directory
            results_dir = os.path.join(LOG_DIR, f"specific-{timestamp}")
            os.makedirs(results_dir, exist_ok=True)

            # Write result to a file
            results_filename = os.path.join(
                results_dir, "specific-test-result.json")
            with open(results_filename, 'w') as f:
                json.dump(specific_result, f, indent=2, default=str)

            logger.info(f"Result saved to: {results_filename}")

        except Exception as e:
            logger.error(f"Error testing combination: {e}")

    write_config(backup_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Defense Pipeline")
    parser.add_argument("--config", "-c", type=str,
                        help="Path to configuration JSON for preconfigured mode")
    args = parser.parse_args()

    logger.info("=== ML Defense Tool Combination Tester ===")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")

    if args.config:
        if os.path.exists(args.config):
            logger.info(f"Using configuration file: {args.config}")
            test_all_combinations(args.config)
        results = test_all_combinations()

        results_dir = os.path.join(LOG_DIR, f"results-{timestamp}")

        # dashboard_file = create_dashboard(results_dir)
        # Open {dashboard_file} to view interactive results.")
        logger.info(f"Test run complete.")

    logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
