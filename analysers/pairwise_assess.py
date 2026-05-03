import pandas as pd
import ast
from typing import Dict, List, Tuple, Optional

# Tool metrics configuration
TOOL_METRICS = {
    'pre-xgbod': {'metric': 'ood_auc', 'higher_better': True},
    'in-trades': {'metric': 'pgd_acc', 'higher_better': True},
    'post_neuronprune': {'metric': 'asr', 'higher_better': False},
    'post-magnet': {'metric': 'carlini_acc', 'higher_better': True},
    'deploy_dataset_inference': {'metric': 'fingerprinting', 'higher_better': True},
    'deploy_dp': {'metric': 'acc_test_clean', 'higher_better': True},
    'in-teaching': {'metric': 'acc_test_clean', 'higher_better': True},
    'in-dp': {'metric': 'mia_auc', 'higher_better': False},
    'watermarking': {'metric': 'watermark_accuracy', 'higher_better': True},
    'watermarkbn': {'metric': 'wmacc_badnets', 'higher_better': True},
    'deploy_explainshap': {'metric': 'drop10_score', 'higher_better': True},
}

# Define what constitutes "no operation" for each stage
NOOP_TOOLS = {
    'pre_training': ['noop'],
    'in_training': ['in_noop'],
    'post_training': ['post_noop'],
    'deployment': ['deploy_noop']
}

def parse_tool_list(tool_str):
    """Parse tool string representation into a list of tools"""
    if isinstance(tool_str, str):
        try:
            # Handle both single quotes and double quotes
            if tool_str.startswith('[') and tool_str.endswith(']'):
                return ast.literal_eval(tool_str.replace("'", '"'))
            return [tool_str]
        except:
            return [tool_str]
    elif isinstance(tool_str, list):
        return tool_str
    return []

def is_noop_tool(tool, stage):
    """Check if a tool is considered noop for a given stage"""
    return tool in NOOP_TOOLS.get(stage, [])

def get_tool_baseline(df, tool_name, stage_columns):
    """Find the baseline combination for a single tool (only that tool + noops)"""
    for _, row in df.iterrows():
        all_tools = []
        
        # Check all stages for tools
        for stage_col in stage_columns:
            stage_tools = parse_tool_list(row[stage_col])
            all_tools.extend(stage_tools)
        
        # Filter out noop tools
        active_tools = []
        for stage_col in stage_columns:
            stage_tools = parse_tool_list(row[stage_col])
            for tool in stage_tools:
                if not is_noop_tool(tool, stage_col):
                    active_tools.append(tool)
        
        # Should have exactly one active tool (the baseline tool)
        if len(active_tools) == 1 and active_tools[0] == tool_name:
            return row
    
    return None

def get_combination_with_exact_tools(df, tool1, tool2, stage_columns):
    """Find combination where exactly these two tools are present (plus noops)"""
    combinations = []
    
    for _, row in df.iterrows():
        all_tools = []
        
        # Check all stages for tools
        for stage_col in stage_columns:
            stage_tools = parse_tool_list(row[stage_col])
            all_tools.extend(stage_tools)
        
        # Filter out noop tools to get active tools only
        active_tools = []
        for stage_col in stage_columns:
            stage_tools = parse_tool_list(row[stage_col])
            for tool in stage_tools:
                if not is_noop_tool(tool, stage_col):
                    active_tools.append(tool)
        
        # Should have exactly two active tools (the pair we're looking for)
        if (len(active_tools) == 2 and 
            tool1 in active_tools and 
            tool2 in active_tools):
            
            # Get order information if tools are in the same stage
            order_info = {}
            for stage_col in stage_columns:
                stage_tools = parse_tool_list(row[stage_col])
                # Filter out noop tools from this stage
                stage_active_tools = [t for t in stage_tools if not is_noop_tool(t, stage_col)]
                if len(stage_active_tools) == 2 and tool1 in stage_active_tools and tool2 in stage_active_tools:
                    order_info['same_stage'] = stage_col
                    order_info['order'] = stage_active_tools
            
            combinations.append({
                'row': row,
                'order_info': order_info,
                'combination_id': row['combination']
            })
    
    return combinations

# def calculate_difference(baseline_value, combo_value, higher_better):
#     """Calculate the difference based on whether higher is better"""
#     if pd.isna(baseline_value) or pd.isna(combo_value):
#         return None
    
#     if higher_better:
#         return combo_value - baseline_value
#     else:
#         return baseline_value - combo_value

def calculate_difference(baseline_value, combo_value, higher_better):
    """Calculate relative difference and treat -1 as missing data."""

    # Treat -1 as missing because it means metric does not exist
    if baseline_value == -1 or combo_value == -1:
        return None

    if pd.isna(baseline_value) or pd.isna(combo_value):
        return None

    # Avoid division by zero
    if baseline_value == 0:
        return None

    # Relative change
    if higher_better:
        return (combo_value - baseline_value) / abs(baseline_value)
    else:
        return (baseline_value - combo_value) / abs(baseline_value)
    


def assess_change_and_direction(difference, t1, t2, higher_better):
    """Assess the severity of change and determine direction"""
    if difference is None:
        return "N/A - Missing data", "N/A"
    
    abs_diff = abs(difference)
    
    if abs_diff < t1:
        return "negligible", "no change"
    elif t1 <= abs_diff < t2:
        severity = "moderate"
    else:
        severity = "severe"
    
    # Determine direction for non-negligible changes
    if higher_better:
        direction = "better" if difference > 0 else "worse"
    else:
        direction = "better" if difference > 0 else "worse"
    
    return severity, direction

def assess_overall_combination(change1, change2, direction1, direction2):
    """Assess overall combination quality"""
    if change1 == "N/A - Missing data" or change2 == "N/A - Missing data":
        return "unknown"
    
    # Check if either tool performs badly (moderate or severe worse)
    if (change1 in ["moderate", "severe"] and direction1 == "worse") or \
       (change2 in ["moderate", "severe"] and direction2 == "worse"):
        return "bad"
    
    # If both changes are negligible (no change), mark as good
    if change1 == "negligible" and change2 == "negligible":
        return "good"
    
    # If both are good improvements
    if (change1 in ["moderate", "severe"] and direction1 == "better" and 
        change2 in ["moderate", "severe"] and direction2 == "better"):
        return "good"
    
    # Mixed case: one negligible (no change), one good improvement
    if (change1 == "negligible" and change2 in ["moderate", "severe"] and direction2 == "better") or \
       (change2 == "negligible" and change1 in ["moderate", "severe"] and direction1 == "better"):
        return "good"
    
    return "mixed"

def analyze_tool_pair_interaction(df, tool1, tool2, t1=0.05, t2=0.15):
    """Analyze interaction between two tools (only exact pairs)"""
    stage_columns = ['pre_training', 'in_training', 'post_training', 'deployment']
    
    # Get baselines
    baseline1 = get_tool_baseline(df, tool1, stage_columns)
    baseline2 = get_tool_baseline(df, tool2, stage_columns)
    
    # Get combinations where exactly these two tools are present
    combinations = get_combination_with_exact_tools(df, tool1, tool2, stage_columns)
    
    if baseline1 is None or baseline2 is None or not combinations:
        return None
    
    results = []
    
    for combo_info in combinations:
        combo = combo_info['row']
        
        # Get metric info
        metric1 = TOOL_METRICS[tool1]['metric']
        higher_better1 = TOOL_METRICS[tool1]['higher_better']
        
        metric2 = TOOL_METRICS[tool2]['metric']
        higher_better2 = TOOL_METRICS[tool2]['higher_better']
        
        # Extract values
        baseline1_value = baseline1[metric1]
        baseline2_value = baseline2[metric2]
        combo1_value = combo[metric1]
        combo2_value = combo[metric2]
        
        # Calculate differences
        diff1 = calculate_difference(baseline1_value, combo1_value, higher_better1)
        diff2 = calculate_difference(baseline2_value, combo2_value, higher_better2)
        
        # Assess changes and directions
        change1, direction1 = assess_change_and_direction(diff1, t1, t2, higher_better1)
        change2, direction2 = assess_change_and_direction(diff2, t1, t2, higher_better2)
        
        # Assess overall combination
        overall_quality = assess_overall_combination(change1, change2, direction1, direction2)
        
        # Add order information if tools are in the same stage
        order_info = ""
        if combo_info['order_info']:
            stage = combo_info['order_info']['same_stage']
            order = combo_info['order_info']['order']
            order_info = f"{stage}:{'>'.join(order)}"
        
        result = {
            'tool_pair': f"{tool1}_{tool2}",
            'tool1': tool1,
            'tool2': tool2,
            'tool1_metric': metric1,
            'tool2_metric': metric2,
            'baseline1_value': baseline1_value,
            'baseline2_value': baseline2_value,
            'combo1_value': combo1_value,
            'combo2_value': combo2_value,
            'relative_difference1': diff1,
            'relative_difference2': diff2,
            'change_severity1': change1,
            'change_severity2': change2,
            'direction1': direction1,
            'direction2': direction2,
            'overall_combination': overall_quality,
            'combination_id': combo['combination'],
            'total_duration': combo['total_duration'],
            'tool_order': order_info
        }
        
        results.append(result)
    
    return results

def analyze_all_pairs(df, t1=0.05, t2=0.15):
    """Analyze all possible pairs of tools (only exact pairs)"""
    results = []
    tools = list(TOOL_METRICS.keys())
    
    for i in range(len(tools)):
        for j in range(i + 1, len(tools)):
            tool1 = tools[i]
            tool2 = tools[j]
            
            pair_results = analyze_tool_pair_interaction(df, tool1, tool2, t1, t2)
            if pair_results:
                results.extend(pair_results)
    
    return pd.DataFrame(results)

def main():
    # Load the data
    # df = pd.read_csv('TRADES_wo_dp_new.csv')
    df = pd.read_csv('new_DP_results_combinations.csv')
    # df = pd.read_csv('new_trades_w_depdp.csv')
    # df = pd.read_csv('teaching.csv')
    # df = pd.read_csv('in-dp.csv')
    # df = pd.read_csv('watermarking.csv')
    
    # Analyze all pairs
    print("Analyzing all tool pairs (exact pairs only)...")
    all_results = analyze_all_pairs(df, t1=0.05, t2=0.15)
    
    if not all_results.empty:
        # Save to CSV
        output_file = 'tool_pair_analysis.csv'
        all_results.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        # Print summary
        print(f"\nAnalysis complete! Found {len(all_results)} exact tool pair combinations.")
        print(f"Overall combination quality distribution:")
        print(all_results['overall_combination'].value_counts())
        
        # Show pairs with different orderings
        same_stage_pairs = all_results[all_results['tool_order'] != ""]
        if not same_stage_pairs.empty:
            print(f"\nTool pairs applied in the same stage (order matters):")
            for tool_pair in same_stage_pairs['tool_pair'].unique():
                pair_data = same_stage_pairs[same_stage_pairs['tool_pair'] == tool_pair]
                if len(pair_data) > 1:
                    print(f"\n{tool_pair}:")
                    for _, row in pair_data.iterrows():
                        print(f"  Order: {row['tool_order']}, Overall: {row['overall_combination']}")
        
        # Show sample of results
        print(f"\nSample of results:")
        sample_cols = ['tool_pair', 'combination_id', 'overall_combination', 'tool_order', 'change_severity1', 'direction1', 'change_severity2', 'direction2']
        print(all_results[sample_cols].head(10))
    
    else:
        print("No exact tool pairs found for analysis.")

if __name__ == "__main__":
    main()

