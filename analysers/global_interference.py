import collections
import pandas as pd
import ast
import argparse
from collections import defaultdict, OrderedDict
import os
import json
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import codecs

# --- Arguments ---
def parse_arguments():
    parser = argparse.ArgumentParser(description='Global Interference Analysis with Visualization')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    parser.add_argument('--output', '-o', required=True, help='Output folder path')
    parser.add_argument('--t1', type=float, default=0.02, help='Threshold for moderate change')
    parser.add_argument('--t2', type=float, default=0.05, help='Threshold for severe change')
    return parser.parse_args()

# --- Tool definitions ---
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
    'no_tools': {'metric': 'acc_test_clean', 'higher_better': True}
}

TOOL_ABBREVIATIONS = {
    'pre-xgbod': 'X', 'in-trades': 'T', 'post_neuronprune': 'N',
    'post-magnet': 'M', 'deploy_dp': 'D', 'deploy_dataset_inference': 'I',
    'in-dp': 'O', 'in-teaching': 'G', 'watermarking': 'W', 'watermarkbn': 'B', 'deploy_explainshap': 'S'
}

NOOP_TOKENS = {
    'noop',
    'in_noop',
    'post_noop',
    'deploy_noop',
    'pre_noop'
}

def get_tool_abbreviation(tool_name):
    return TOOL_ABBREVIATIONS.get(tool_name, tool_name[:2].upper())

def tools_to_abbreviation(tool_list):
    return ''.join([get_tool_abbreviation(tool) for tool in tool_list])

# def parse_tool_list(tool_str):
#     if pd.isna(tool_str) or tool_str in ['noop', 'in_noop', "['noop']", "['in_noop']"]:
#         return []
#     if isinstance(tool_str, str):
#         if tool_str.startswith('[') and tool_str.endswith(']'):
#             try:
#                 parsed = ast.literal_eval(tool_str)
#                 if isinstance(parsed, list):
#                     return [t for t in parsed if t not in ['noop','in_noop']]
#                 elif parsed not in ['noop','in_noop']:
#                     return [parsed]
#             except:
#                 cleaned = tool_str.strip('[]"\'').replace(' ','').split(',')
#                 return [t for t in cleaned if t not in ['noop','in_noop']]
#         elif tool_str not in ['noop','in_noop']:
#             return [tool_str]
#     return []

def parse_tool_list(tool_str):
    if pd.isna(tool_str):
        return []

    if isinstance(tool_str, str):
        if tool_str.startswith('[') and tool_str.endswith(']'):
            try:
                parsed = ast.literal_eval(tool_str)
                if isinstance(parsed, list):
                    return [t for t in parsed if t not in NOOP_TOKENS]
                elif parsed not in NOOP_TOKENS:
                    return [parsed]
            except:
                cleaned = tool_str.strip('[]"\'').replace(' ','').split(',')
                return [t for t in cleaned if t not in NOOP_TOKENS]
        elif tool_str not in NOOP_TOKENS:
            return [tool_str]

    return []

def create_ordered_toolset_identifier(row):
    stages = ['pre_training','in_training','post_training','deployment']
    parts = []
    for s in stages:
        tools = parse_tool_list(row[s])
        if tools:
            parts.append(f"{s.split('_')[0]}:{'+'.join(tools)}")
    return "|".join(parts) if parts else "no_tools"

def get_ordered_tools(row):
    stages = ['pre_training','in_training','post_training','deployment']
    ordered = []
    for s in stages:
        ordered.extend(parse_tool_list(row[s]))
    return ordered

# --- Metric helpers ---
def get_metric_value(metrics, metric_name):
    if metric_name in metrics and not pd.isna(metrics[metric_name]) and metrics[metric_name]!=-1:
        return float(metrics[metric_name])
    return np.nan

# def calculate_difference(prime_val, focus_val, higher_better=True):
#     if np.isnan(prime_val) or np.isnan(focus_val):
#         return np.nan, 'na'
#     diff = focus_val - prime_val if higher_better else prime_val - focus_val
#     direction = 'positive' if diff > 0 else ('negative' if diff < 0 else 'none')
#     return diff, direction

def calculate_difference(prime_val, focus_val, higher_better=True):
    """
    Desirability-aligned relative change.
    Positive = focus combo improved compared to prime combo.
    Negative = focus combo got worse compared to prime combo.
    """

    if np.isnan(prime_val) or np.isnan(focus_val):
        return np.nan, 'na'

    # Avoid unstable relative change when baseline is zero or extremely small
    if abs(prime_val) < 1e-6:
        return np.nan, 'na'

    if higher_better:
        diff = (focus_val - prime_val) / abs(prime_val)
    else:
        diff = (prime_val - focus_val) / abs(prime_val)

    direction = 'positive' if diff > 0 else ('negative' if diff < 0 else 'none')
    return diff, direction


def classify_change(diff, t1, t2):
    if pd.isna(diff):
        return 'na'
    abs_diff = abs(diff)
    if abs_diff < t1:
        return 'negligible'
    elif t1 <= abs_diff < t2:
        return 'moderate'
    else:
        return 'severe'

# --- Interference analysis ---
def analyze_global_interference(prime_combo, focus_combo, focus_tool, t1, t2):
    prime_metrics = prime_combo['metrics']
    focus_metrics = focus_combo['metrics']
    prime_tools = prime_combo['ordered_tools']
    focus_tools = focus_combo['ordered_tools']

    valid = any(prime_tools[:i]+[focus_tool]+prime_tools[i:] == focus_tools for i in range(len(prime_tools)+1))
    if not valid:
        print(f"Warning: Invalid prime-focus relationship for {focus_tool}")
        return None

    results = {
        'prime_identifier': prime_combo['identifier'],
        'focus_identifier': focus_combo['identifier'],
        'focus_tool': focus_tool,
        'prime_tools': prime_tools,
        'focus_tools': focus_tools,
        'metric_changes': {},
        'interference_detected': False,
        'interference_severity': 'none',
        'interference_direction': 'none',
        'focus_tool_performance': 'metric_equal',
        'acc_test_clean_diff': None,
        'acc_test_clean_direction': 'none',
        'acc_test_clean_severity': 'negligible',
        'focus_tool_metric_diff': None,
        'focus_tool_metric_direction': 'none',
        'focus_tool_metric_severity': 'negligible'
    }

    # --- acc_test_clean diff ---
    prime_acc = get_metric_value(prime_metrics, 'acc_test_clean')
    focus_acc = get_metric_value(focus_metrics, 'acc_test_clean')
    diff, direction = calculate_difference(prime_acc, focus_acc, higher_better=True)
    results['acc_test_clean_diff'] = diff
    results['acc_test_clean_direction'] = direction
    results['acc_test_clean_severity'] = classify_change(diff, t1, t2)

    # --- focus tool metric ---
    if focus_tool in TOOL_METRICS:
        metric_name = TOOL_METRICS[focus_tool]['metric']
        higher_better = TOOL_METRICS[focus_tool]['higher_better']
        prime_val = get_metric_value(prime_metrics, metric_name)
        focus_val = get_metric_value(focus_metrics, metric_name)
        diff, direction = calculate_difference(prime_val, focus_val, higher_better)
        severity = classify_change(diff, t1, t2)
        results['focus_tool_metric_diff'] = diff
        results['focus_tool_metric_direction'] = direction
        results['focus_tool_metric_severity'] = severity
        results['focus_tool_performance'] = 'metric_equal' if severity=='negligible' else f"metric_{severity}_{direction}"

    # --- existing prime tool metrics ---
    for tool in prime_tools:
        if tool not in TOOL_METRICS:
            continue
        metric_name = TOOL_METRICS[tool]['metric']
        higher_better = TOOL_METRICS[tool]['higher_better']
        prime_val = get_metric_value(prime_metrics, metric_name)
        focus_val = get_metric_value(focus_metrics, metric_name)
        diff, direction = calculate_difference(prime_val, focus_val, higher_better)
        severity = classify_change(diff, t1, t2)
        results['metric_changes'][tool] = {
            'metric': metric_name,
            'difference': diff,
            'direction': direction,
            'change_type': severity,
            'prime_value': prime_val,
            'focus_value': focus_val
        }

    # --- interference detection ---
    # --- interference detection (prime tools only) ---
    significant = [v for k, v in results['metric_changes'].items() 
                if k in prime_tools and v['change_type'] in ('moderate','severe')]
    if significant:
        results['interference_detected'] = True
        dirs = [v['direction'] for v in significant]
        sev = [v['change_type'] for v in significant]
        results['interference_severity'] = 'mixed' if len(set(sev))>1 else sev[0]
        results['interference_direction'] = 'mixed' if len(set(dirs))>1 else dirs[0]
    else:
        results['interference_detected'] = False
        results['interference_severity'] = 'none'
        results['interference_direction'] = 'none'

    return results


# Update the create_comprehensive_trees function to use the new layout
def create_comprehensive_trees(interference_results, all_combos, output_path):
    """Create comprehensive trees for each focus tool showing horizontal and vertical relationships"""
    os.makedirs(os.path.join(output_path, 'comprehensive_trees'), exist_ok=True)
    
    for focus_tool, analyses in interference_results.items():
        if not analyses:
            continue
            
        print(f"Creating comprehensive tree for focus tool: {focus_tool}")
        
        # Create a graph for this focus tool
        G = nx.DiGraph()
        node_info = {}
        
        # Add all nodes from analyses
        for analysis in analyses.values():
            if analysis:
                prime_id = analysis['prime_identifier']
                focus_id = analysis['focus_identifier']
                
                # Add nodes with abbreviations
                prime_abbr = tools_to_abbreviation(analysis['prime_tools'])
                focus_abbr = tools_to_abbreviation(analysis['focus_tools'])
                
                G.add_node(prime_id, abbr=prime_abbr, tools=analysis['prime_tools'], 
                          tool_count=len(analysis['prime_tools']), node_type='prime')
                G.add_node(focus_id, abbr=focus_abbr, tools=analysis['focus_tools'], 
                          tool_count=len(analysis['focus_tools']), node_type='focus', 
                          focus_tool=focus_tool)
                
                # Add horizontal edge (prime -> focus)
                edge_attrs = {
                    'relationship': 'horizontal',
                    'interference': analysis['interference_detected'],
                    'severity': analysis['interference_severity'],
                    'direction': analysis['interference_direction']
                }
                G.add_edge(prime_id, focus_id, **edge_attrs)
                
                # Store node info
                node_info[prime_id] = {'abbr': prime_abbr, 'tools': analysis['prime_tools']}
                node_info[focus_id] = {'abbr': focus_abbr, 'tools': analysis['focus_tools']}
        
        # Add vertical relationships
        all_nodes = list(G.nodes())
        for i, node1_id in enumerate(all_nodes):
            node1_tools = G.nodes[node1_id]['tools']
            node1_count = len(node1_tools)
            
            for j, node2_id in enumerate(all_nodes):
                if i == j:
                    continue
                    
                node2_tools = G.nodes[node2_id]['tools']
                node2_count = len(node2_tools)
                
                if node1_count == node2_count + 1:
                    for k in range(len(node1_tools)):
                        expected_node2 = node1_tools[:k] + node1_tools[k+1:]
                        if expected_node2 == node2_tools:
                            G.add_edge(node1_id, node2_id, relationship='vertical')
                            break
        
        # Create both types of visualizations
        visualize_comprehensive_tree(G, focus_tool, output_path, node_info)
        visualize_horizontal_layout_tree(G, focus_tool, output_path, node_info)
        create_interactive_horizontal_tree(G, focus_tool, output_path, node_info)



def visualize_comprehensive_tree(G, focus_tool, output_path, node_info):
    """Visualize the comprehensive tree with horizontal and vertical relationships"""
    if len(G.nodes()) == 0:
        return
        
    plt.figure(figsize=(25, 15))
    
    # Create hierarchical layout based on tool count
    layers = defaultdict(list)
    for node in G.nodes():
        tool_count = G.nodes[node]['tool_count']
        layers[tool_count].append(node)
    
    # Create positions
    pos = {}
    max_tool_count = max(layers.keys()) if layers else 0
    layer_spacing = 3.0
    
    for tool_count, nodes in layers.items():
        y_pos = (max_tool_count - tool_count) * layer_spacing
        x_positions = np.linspace(0, 20, len(nodes))
        
        for i, node in enumerate(nodes):
            pos[node] = (x_positions[i], y_pos)
    
    # Draw nodes with different styles
    prime_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'prime']
    focus_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'focus']
    
    # Draw prime nodes (blue circles)
    nx.draw_networkx_nodes(G, pos, nodelist=prime_nodes, 
                          node_color='lightblue', node_size=1200, alpha=0.8)
    
    # Draw focus nodes (green squares)
    nx.draw_networkx_nodes(G, pos, nodelist=focus_nodes, 
                          node_color='lightgreen', node_size=1200, alpha=0.8, node_shape='s')
    
    # Draw edges with different styles for horizontal and vertical relationships
    horizontal_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relationship') == 'horizontal']
    vertical_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relationship') == 'vertical']
    
    # Draw horizontal edges with colors based on interference
    for severity, color in [('none', 'gray'), ('moderate', 'orange'), 
                          ('severe', 'red'), ('mixed', 'purple')]:
        edges = [(u, v) for u, v in horizontal_edges 
                if G[u][v].get('severity') == severity]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color, 
                              width=3, arrows=True, arrowsize=20, alpha=0.8,
                              connectionstyle="arc3,rad=0.1")  # Curved edges
    
    # Draw vertical edges (dashed lines)
    nx.draw_networkx_edges(G, pos, edgelist=vertical_edges, edge_color='blue',
                          width=2, arrows=True, arrowsize=15, alpha=0.6,
                          style='dashed')
    
    # Create labels with abbreviations
    labels = {}
    for node in G.nodes():
        abbr = G.nodes[node]['abbr']
        tool_count = G.nodes[node]['tool_count']
        labels[node] = f"{abbr}\n({tool_count})"
    
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                  markersize=15, label='Prime Nodes'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgreen', 
                  markersize=15, label='Focus Nodes'),
        plt.Line2D([0], [0], color='gray', lw=3, label='No Interference (Horizontal)'),
        plt.Line2D([0], [0], color='orange', lw=3, label='Moderate Interference (Horizontal)'),
        plt.Line2D([0], [0], color='red', lw=3, label='Severe Interference (Horizontal)'),
        plt.Line2D([0], [0], color='purple', lw=3, label='Mixed Interference (Horizontal)'),
        plt.Line2D([0], [0], color='blue', lw=2, linestyle='dashed', label='Vertical Relationship')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1))
    
    plt.title(f'Comprehensive Tree for Focus Tool: {focus_tool} ({get_tool_abbreviation(focus_tool)})\n'
             f'Horizontal: Prime → Focus with Interference | Vertical: Tool Breakdown Relationships',
             fontsize=14, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the visualization
    safe_tool_name = focus_tool.replace(' ', '_').replace('-', '_')
    plt.savefig(os.path.join(output_path, 'comprehensive_trees', f'{safe_tool_name}_comprehensive_tree.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    


def visualize_horizontal_layout_tree(G, focus_tool, output_path, node_info):
    """Visualize tree with prime nodes next to their focus combos with dynamic spacing"""
    if len(G.nodes()) == 0:
        return
        
    plt.figure(figsize=(50, 40))  # Very large figure to ensure no overlap
    
    # Get all relationships
    horizontal_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relationship') == 'horizontal']
    vertical_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relationship') == 'vertical']
    
    # Group horizontal pairs by tool count of focus node
    pairs_by_tool_count = defaultdict(list)
    for prime, focus in horizontal_edges:
        tool_count = G.nodes[focus]['tool_count']
        pairs_by_tool_count[tool_count].append((prime, focus))
    
    # Create positions with dynamic spacing based on label length
    pos = {}
    max_tool_count = max(pairs_by_tool_count.keys(), default=0)
    layer_spacing = 10.0  # More vertical space
    
    # Calculate required horizontal space for each row
    row_widths = {}
    for tool_count, pairs in pairs_by_tool_count.items():
        # Estimate width needed based on number of pairs and their label lengths
        max_label_length = max(
            len(G.nodes[prime]['abbr']) + len(G.nodes[focus]['abbr']) 
            for prime, focus in pairs
        )
        row_widths[tool_count] = len(pairs) * (max_label_length * 0.8 + 6.0)
    
    max_row_width = max(row_widths.values(), default=50)
    
    # Position each row of prime-focus pairs with dynamic spacing
    for tool_count, pairs in pairs_by_tool_count.items():
        y_pos = (max_tool_count - tool_count) * layer_spacing
        
        if len(pairs) == 1:
            # Center single pairs with generous spacing
            prime, focus = pairs[0]
            pos[prime] = (max_row_width/2 - 4.0, y_pos)  # Prime left
            pos[focus] = (max_row_width/2 + 4.0, y_pos)  # Focus right
        else:
            # Distribute multiple pairs with spacing based on content
            x_positions = np.linspace(5, max_row_width - 5, len(pairs))
            
            for i, (prime, focus) in enumerate(pairs):
                prime_abbr_len = len(G.nodes[prime]['abbr'])
                focus_abbr_len = len(G.nodes[focus]['abbr'])
                
                # Dynamic spacing based on label lengths
                spacing = 3.0 + (prime_abbr_len + focus_abbr_len) * 0.3
                center_x = x_positions[i]
                
                pos[prime] = (center_x - spacing/2, y_pos)  # Prime left
                pos[focus] = (center_x + spacing/2, y_pos)  # Focus right
    
    # Draw nodes with guaranteed no overlap
    prime_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'prime']
    focus_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'focus']
    
    # Draw prime nodes (blue circles)
    for node in prime_nodes:
        if node in pos:
            x, y = pos[node]
            circle = plt.Circle((x, y), radius=0.6, facecolor='lightblue', edgecolor='darkblue', 
                               linewidth=3, alpha=0.95, zorder=3)
            plt.gca().add_patch(circle)
    
    # Draw focus nodes (green squares)
    for node in focus_nodes:
        if node in pos:
            x, y = pos[node]
            square = plt.Rectangle((x - 0.6, y - 0.6), 1.2, 1.2, facecolor='lightgreen', 
                                  edgecolor='darkgreen', linewidth=3, alpha=0.95, zorder=2)
            plt.gca().add_patch(square)
    
    # Draw HORIZONTAL edges with better visibility
    for severity, color in [('none', 'gray'), ('moderate', 'orange'), 
                          ('severe', 'red'), ('mixed', 'purple')]:
        edges = [(u, v) for u, v in horizontal_edges 
                if G[u][v].get('severity') == severity]
        
        for u, v in edges:
            if u in pos and v in pos:
                # Draw straight arrow with enough space
                plt.annotate("", xy=pos[v], xytext=pos[u],
                            arrowprops=dict(arrowstyle="->", color=color, 
                                          lw=5, alpha=0.9, shrinkA=25, shrinkB=25))
    
    # Draw VERTICAL edges (focus -> focus)
    focus_vertical_edges = []
    for u, v in vertical_edges:
        if (G.nodes[u]['node_type'] == 'focus' and 
            G.nodes[v]['node_type'] == 'focus' and 
            u in pos and v in pos):
            focus_vertical_edges.append((u, v))
    
    for u, v in focus_vertical_edges:
        plt.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                'b--', alpha=0.7, linewidth=3, zorder=1)
        
        if G.nodes[u]['tool_count'] > G.nodes[v]['tool_count']:
            plt.annotate("", xy=pos[v], xytext=pos[u],
                        arrowprops=dict(arrowstyle="->", color='blue', 
                                      lw=3, alpha=0.7, linestyle='dashed'))
        else:
            plt.annotate("", xy=pos[u], xytext=pos[v],
                        arrowprops=dict(arrowstyle="->", color='blue', 
                                      lw=3, alpha=0.7, linestyle='dashed'))
    
    # Create and draw labels with proper spacing
    for node, (x, y) in pos.items():
        abbr = G.nodes[node]['abbr']
        tool_count = G.nodes[node]['tool_count']
        node_type = G.nodes[node]['node_type']
        
        if node_type == 'focus':
            focus_tool_abbr = get_tool_abbreviation(G.nodes[node]['focus_tool'])
            label_text = f"{abbr}\n({tool_count}t)"
        else:
            label_text = f"{abbr}\n({tool_count}t)"
        
        plt.text(x, y, label_text, fontsize=11, fontweight='bold',
                ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'),
                zorder=4)
    
    # Add tool count labels
    for tool_count in pairs_by_tool_count.keys():
        if pairs_by_tool_count[tool_count]:
            y_pos = (max_tool_count - tool_count) * layer_spacing
            plt.text(-10, y_pos, f"{tool_count} tools", fontsize=16, fontweight='bold', 
                    ha='right', va='center', bbox=dict(facecolor='white', alpha=0.9))
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                  markersize=20, label='Prime Nodes', markeredgecolor='darkblue', markeredgewidth=2),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgreen', 
                  markersize=20, label='Focus Nodes', markeredgecolor='darkgreen', markeredgewidth=2),
        plt.Line2D([0], [0], color='gray', lw=5, label='No Interference (Prime→Focus)'),
        plt.Line2D([0], [0], color='orange', lw=5, label='Moderate Interference (Prime→Focus)'),
        plt.Line2D([0], [0], color='red', lw=5, label='Severe Interference (Prime→Focus)'),
        plt.Line2D([0], [0], color='purple', lw=5, label='Mixed Interference (Prime→Focus)'),
        plt.Line2D([0], [0], color='blue', lw=4, linestyle='dashed', label='Vertical Breakdown (Focus→Focus)')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1), fontsize=16)
    
    plt.title(f'Complete Visualization for Focus Tool: {focus_tool} ({get_tool_abbreviation(focus_tool)})\n'
             f'Horizontal: Prime → Focus interference relationships\n'
             f'Vertical: Focus → Focus breakdown relationships',
             fontsize=22, fontweight='bold', pad=60)
    plt.axis('off')
    plt.tight_layout()
    
    safe_tool_name = focus_tool.replace(' ', '_').replace('-', '_')
    plt.savefig(os.path.join(output_path, 'comprehensive_trees', f'{safe_tool_name}_complete_layout.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

def create_interactive_horizontal_tree(G, focus_tool, output_path, node_info):
    """Create interactive HTML version with proper relationships"""
    horizontal_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relationship') == 'horizontal']
    vertical_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('relationship') == 'vertical']
    
    # Group horizontal pairs
    pairs_by_tool_count = defaultdict(list)
    for prime, focus in horizontal_edges:
        tool_count = G.nodes[focus]['tool_count']
        pairs_by_tool_count[tool_count].append((prime, focus))
    
    # Create positions
    pos = {}
    max_tool_count = max(pairs_by_tool_count.keys(), default=0)
    layer_spacing = 5.0
    horizontal_spacing = 3.0
    
    for tool_count, pairs in pairs_by_tool_count.items():
        y_pos = (max_tool_count - tool_count) * layer_spacing
        x_positions = np.linspace(0, 30, len(pairs))
        
        for i, (prime, focus) in enumerate(pairs):
            center_x = x_positions[i] * horizontal_spacing
            pos[prime] = (center_x - 1.2, y_pos)
            pos[focus] = (center_x + 1.2, y_pos)
    
    # Create edge traces
    horizontal_traces = []
    vertical_traces = []
    
    # Horizontal edges (prime -> focus)
    for severity, color in [('none', 'gray'), ('moderate', 'orange'), 
                          ('severe', 'red'), ('mixed', 'purple')]:
        edges = [(u, v) for u, v in horizontal_edges 
                if G[u][v].get('severity') == severity]
        
        if edges:
            x_edges, y_edges = [], []
            for u, v in edges:
                if u in pos and v in pos:
                    # Curved line for horizontal relationships
                    x_edges.extend([pos[u][0], (pos[u][0] + pos[v][0])/2, pos[v][0], None])
                    y_edges.extend([pos[u][1], pos[u][1] + 0.3, pos[v][1], None])
            
            horizontal_traces.append(go.Scatter(
                x=x_edges, y=y_edges,
                line=dict(width=5, color=color),
                hoverinfo='none',
                mode='lines',
                name=f'{severity.capitalize()} Interference (Prime→Focus)'
            ))
    
    # Vertical edges (focus -> focus)
    focus_vertical_edges = [(u, v) for u, v in vertical_edges 
                           if G.nodes[u]['node_type'] == 'focus' and G.nodes[v]['node_type'] == 'focus']
    
    if focus_vertical_edges:
        x_edges, y_edges = [], []
        for u, v in focus_vertical_edges:
            if u in pos and v in pos:
                x_edges.extend([pos[u][0], pos[v][0], None])
                y_edges.extend([pos[u][1], pos[v][1], None])
        
        vertical_traces.append(go.Scatter(
            x=x_edges, y=y_edges,
            line=dict(width=3, color='blue', dash='dash'),
            hoverinfo='none',
            mode='lines',
            name='Vertical Breakdown (Focus→Focus)'
        ))
    
    # Create node traces
    prime_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'prime' and n in pos]
    focus_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'focus' and n in pos]
    
    # Prime nodes
    prime_x, prime_y, prime_text = [], [], []
    for node in prime_nodes:
        x, y = pos[node]
        prime_x.append(x)
        prime_y.append(y)
        prime_text.append(f"Prime: {node_info[node]['abbr']}<br>Tools: {G.nodes[node]['tool_count']}<br>ID: {node}")
    
    prime_trace = go.Scatter(
        x=prime_x, y=prime_y,
        mode='markers+text',
        text=[node_info[node]['abbr'] for node in prime_nodes],
        textposition="middle center",
        hovertext=prime_text,
        hoverinfo='text',
        marker=dict(color='lightblue', size=35, line=dict(width=3, color='darkblue')),
        name='Prime Nodes'
    )
    
    # Focus nodes
    focus_x, focus_y, focus_text = [], [], []
    for node in focus_nodes:
        x, y = pos[node]
        focus_x.append(x)
        focus_y.append(y)
        focus_tool_abbr = get_tool_abbreviation(G.nodes[node]['focus_tool'])
        focus_text.append(f"Focus: {node_info[node]['abbr']}<br>Tools: {G.nodes[node]['tool_count']}<br>Focus Tool: {G.nodes[node]['focus_tool']} ({focus_tool_abbr})<br>ID: {node}")
    
    focus_trace = go.Scatter(
        x=focus_x, y=focus_y,
        mode='markers+text',
        text=[f"{node_info[node]['abbr']}" for node in focus_nodes],
        textposition="middle center",
        hovertext=focus_text,
        hoverinfo='text',
        marker=dict(color='lightgreen', size=35, line=dict(width=3, color='darkgreen'), symbol='square'),
        name='Focus Nodes'
    )
    
    # Combine all traces
    all_traces = horizontal_traces + vertical_traces + [prime_trace, focus_trace]
    
    # Create figure
    fig = go.Figure(data=all_traces,
                   layout=go.Layout(
                       title=f'Complete Relationships for {focus_tool}',
                       showlegend=True,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=60),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       width=1400,
                       height=800,
                       legend=dict(x=1.05, y=1)
                   ))
    
    safe_tool_name = focus_tool.replace(' ', '_').replace('-', '_')
    fig.write_html(os.path.join(output_path, 'comprehensive_trees', f'{safe_tool_name}_complete_interactive.html'))


def verify_focus_prime_relationship(prime_tools, focus_tools, focus_tool):
    """Verify that focus_tools is prime_tools with focus_tool inserted in correct stage order"""
    # Check if focus_tools is exactly prime_tools with focus_tool inserted somewhere
    for i in range(len(prime_tools) + 1):
        expected = prime_tools[:i] + [focus_tool] + prime_tools[i:]
        if expected == focus_tools:
            return True
    
    return False



def create_interference_graph(interference_results):
    """Create a graph visualization of interference relationships"""
    G = nx.DiGraph()
    
    # Add nodes and edges
    for focus_tool, analyses in interference_results.items():
        for analysis in analyses.values():
            if analysis:  # Skip None analyses
                # Add prime node
                prime_id = analysis['prime_identifier']
                prime_tool_count = len(analysis['prime_tools'])
                G.add_node(prime_id, tool_count=prime_tool_count, node_type='prime')
                
                # Add focus node
                focus_id = analysis['focus_identifier']
                focus_tool_count = len(analysis['focus_tools'])
                G.add_node(focus_id, tool_count=focus_tool_count, node_type='focus', 
                          focus_tool=analysis['focus_tool'])
                
                # Add edge with interference info
                edge_attrs = {
                    'interference': analysis['interference_detected'],
                    'severity': analysis['interference_severity'],
                    'direction': analysis['interference_direction'],
                    'focus_performance': analysis['focus_tool_performance']
                }
                G.add_edge(prime_id, focus_id, **edge_attrs)
    
    return G

def analyze_interference_propagation(interference_results, all_combos):
    """Analyze interference propagation through the tool combination hierarchy"""
    propagation_results = {}
    focus_tool_stats = {}
    
    # First pass: check for absolute interference per focus tool
    for focus_tool, analyses in interference_results.items():
        if not analyses:
            continue
            
        # Count total nodes and nodes with interference for this focus tool
        total_nodes = len(analyses)
        nodes_with_interference = sum(1 for a in analyses.values() 
                                    if a and a['interference_detected'])
        
        absolute_interference = (total_nodes > 0 and 
                               nodes_with_interference == total_nodes)
        
        focus_tool_stats[focus_tool] = {
            'total_nodes': total_nodes,
            'nodes_with_interference': nodes_with_interference,
            'absolute_interference': absolute_interference,
            'interference_percentage': (nodes_with_interference / total_nodes * 100 
                                      if total_nodes > 0 else 0)
        }
    
    # Second pass: analyze individual node propagation
    for focus_tool, analyses in interference_results.items():
        if not analyses:
            continue
            
        print(f"Analyzing interference propagation for: {focus_tool}")
        propagation_results[focus_tool] = {
            'absolute_interference': focus_tool_stats[focus_tool]['absolute_interference'],
            'nodes': {}
        }
        
        # Build the graph for this focus tool
        G = nx.DiGraph()
        
        for analysis in analyses.values():
            if analysis:
                prime_id = analysis['prime_identifier']
                focus_id = analysis['focus_identifier']
                
                G.add_node(prime_id, 
                          tools=analysis['prime_tools'],
                          tool_count=len(analysis['prime_tools']),
                          node_type='prime',
                          interference=False)
                
                G.add_node(focus_id,
                          tools=analysis['focus_tools'],
                          tool_count=len(analysis['focus_tools']),
                          node_type='focus',
                          focus_tool=focus_tool,
                          interference=analysis['interference_detected'],
                          severity=analysis['interference_severity'],
                          direction=analysis['interference_direction'])
                
                G.add_edge(prime_id, focus_id, relationship='horizontal')
        
        # Add vertical relationships
        all_nodes = list(G.nodes())
        for i, node1_id in enumerate(all_nodes):
            if node1_id not in G.nodes():
                continue
                
            node1_tools = G.nodes[node1_id]['tools']
            node1_count = len(node1_tools)
            
            for j, node2_id in enumerate(all_nodes):
                if i == j or node2_id not in G.nodes():
                    continue
                    
                node2_tools = G.nodes[node2_id]['tools']
                node2_count = len(node2_tools)
                
                if (node1_count == node2_count + 1 and 
                    G.nodes[node1_id]['node_type'] == 'focus' and 
                    G.nodes[node2_id]['node_type'] == 'focus'):
                    
                    for k in range(len(node1_tools)):
                        expected_node2 = node1_tools[:k] + node1_tools[k+1:]
                        if expected_node2 == node2_tools:
                            G.add_edge(node2_id, node1_id, relationship='vertical')
                            break
        
        # Analyze each focus node
        focus_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'focus']
        for focus_id in focus_nodes:
            try:
                propagation_path = trace_interference_propagation(G, focus_id)
                propagation_results[focus_tool]['nodes'][focus_id] = propagation_path
            except Exception as e:
                print(f"Error analyzing node {focus_id}: {e}")
                continue
    
    # Add focus tool stats to results
    propagation_results['_stats'] = focus_tool_stats
    
    return propagation_results



def trace_interference_propagation(G, start_node):
    """
    Trace interference propagation from a node back to its roots.
    Labels interference_pattern as: continuous | sporadic | tapering | isolated | none.
    Supports multiple root cause and inherited-from ancestors.
    """
    propagation_path = {
        'start_node': start_node,
        'start_tools': G.nodes[start_node].get('tools', []),
        'interference_present': G.nodes[start_node].get('interference', False),
        'severity': G.nodes[start_node].get('severity', 'none'),
        'direction': G.nodes[start_node].get('direction', 'none'),
        'propagation_path': [],
        'root_cause_type': 'none',
        'root_cause_node': start_node,
        'root_cause_tools': G.nodes[start_node].get('tools', []),
        'inherited_from': None,
        'interference_pattern': 'none',
        # plural-aware
        'root_cause_nodes': [],
        'root_cause_tools_list': [],
        'inherited_from_nodes': [],
        'inherited_from_tools_list': []
    }

    if not propagation_path['interference_present']:
        return propagation_path

    # --- collect vertical ancestors (focus-only) ---
    from collections import deque, defaultdict
    layers = defaultdict(list)
    visited = set([start_node])
    q = deque([(start_node, 0)])
    while q:
        node, dist = q.popleft()
        layers[dist].append(node)
        for pred in G.predecessors(node):
            if G.has_edge(pred, node) and G[pred][node].get('relationship') == 'vertical':
                if pred not in visited:
                    visited.add(pred)
                    q.append((pred, dist + 1))

    ancestors = []
    for d in sorted(layers.keys()):
        if d == 0:
            continue
        for n in layers[d]:
            if G.nodes[n].get('node_type') == 'focus':
                ancestors.append((n, d))

    for n, d in ancestors:
        has_int = G.nodes[n].get('interference', False)
        propagation_path['propagation_path'].append({
            'node': n,
            'tools': G.nodes[n].get('tools', []),
            'tool_count': G.nodes[n].get('tool_count', 0),
            'distance': d,
            'has_interference': has_int,
            'severity': G.nodes[n].get('severity', 'none') if has_int else None,
            'direction': G.nodes[n].get('direction', 'none') if has_int else None
        })

    # --- no ancestors → isolated ---
    if not ancestors:
        propagation_path['root_cause_type'] = 'self_caused'
        propagation_path['interference_pattern'] = 'isolated'
        propagation_path['root_cause_nodes'] = [start_node]
        propagation_path['root_cause_tools_list'] = [propagation_path['start_tools']]
        return propagation_path

    # --- interfering ancestors ---
    ancestors_int = [(n, d) for (n, d) in ancestors if G.nodes[n].get('interference', False)]
    if not ancestors_int:
        propagation_path['root_cause_type'] = 'self_caused'
        propagation_path['interference_pattern'] = 'isolated'
        propagation_path['root_cause_nodes'] = [start_node]
        propagation_path['root_cause_tools_list'] = [propagation_path['start_tools']]
        return propagation_path

    # nearest interfering
    min_dist = min(d for _, d in ancestors_int)
    nearest = [n for (n, d) in ancestors_int if d == min_dist]
    propagation_path['inherited_from_nodes'] = nearest[:]
    propagation_path['inherited_from_tools_list'] = [G.nodes[n].get('tools', []) for n in nearest]
    propagation_path['inherited_from'] = nearest[0] if nearest else None

    # earliest interfering (no interfering predecessor upstream)
    interferers = set(n for (n, _) in ancestors_int)

    def has_int_pred(n):
        stack = [n]
        seen = set()
        while stack:
            cur = stack.pop()
            for pred in G.predecessors(cur):
                if not (G.has_edge(pred, cur) and G[pred][cur].get('relationship') == 'vertical'):
                    continue
                if pred in interferers:
                    return True
                if pred not in seen:
                    seen.add(pred)
                    stack.append(pred)
        return False

    earliest = [n for (n, _) in ancestors_int if not has_int_pred(n)]
    if earliest:
        propagation_path['root_cause_type'] = 'inherited'
        propagation_path['root_cause_nodes'] = earliest[:]
        propagation_path['root_cause_tools_list'] = [G.nodes[n].get('tools', []) for n in earliest]
        propagation_path['root_cause_node'] = earliest[0]
        propagation_path['root_cause_tools'] = G.nodes[earliest[0]].get('tools', [])
    else:
        propagation_path['root_cause_type'] = 'self_caused'
        propagation_path['root_cause_nodes'] = [start_node]
        propagation_path['root_cause_tools_list'] = [propagation_path['start_tools']]

    # --- classify interference pattern ---
    def is_all_int_path(src, dst):
        stack = [src]
        visited = set([src])
        while stack:
            cur = stack.pop()
            if cur == dst:
                return True
            for succ in G.successors(cur):
                if not (G[cur][succ].get('relationship') == 'vertical'):
                    continue
                if succ in visited:
                    continue
                if not G.nodes[succ].get('interference', False):
                    continue
                visited.add(succ)
                stack.append(succ)
        return False

    def has_sporadic_gap(start):
        # detect True → False → True along any upward path
        q = deque([(start, 0)])  # state=0: all True, state=1: saw False
        visited = set([(start, 0)])
        while q:
            node, state = q.popleft()
            for pred in G.predecessors(node):
                if not (G.has_edge(pred, node) and G[pred][node].get('relationship') == 'vertical'):
                    continue
                pred_int = G.nodes[pred].get('interference', False)
                if state == 0:
                    if pred_int:
                        next_state = 0
                    else:
                        next_state = 1
                else:  # state==1
                    if pred_int:
                        return True  # interference reappeared after a gap
                    next_state = 1
                key = (pred, next_state)
                if key not in visited:
                    visited.add(key)
                    q.append((pred, next_state))
        return False

    any_cont = any(is_all_int_path(src, start_node) for src in propagation_path['root_cause_nodes'])
    if any_cont:
        propagation_path['interference_pattern'] = 'continuous'
    else:
        if has_sporadic_gap(start_node):
            propagation_path['interference_pattern'] = 'sporadic'
        else:
            propagation_path['interference_pattern'] = 'tapering'

    return propagation_path


def get_vertical_ancestors(G, node):
    """Get all ancestors through vertical relationships"""
    ancestors = set()
    visited = set()
    queue = collections.deque([node])
    
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        
        # Get all predecessors
        for predecessor in list(G.predecessors(current)):
            # Check if edge exists and has vertical relationship
            if G.has_edge(predecessor, current):
                edge_data = G[predecessor][current]
                if edge_data.get('relationship') == 'vertical':
                    ancestors.add(predecessor)
                    queue.append(predecessor)
    
    return list(ancestors)

def generate_interference_report(propagation_results, output_path):
    """Generate a comprehensive interference propagation report"""
    os.makedirs(output_path, exist_ok=True)
    
    report_data = {}
    
    for focus_tool, tool_data in propagation_results.items():
        if focus_tool == '_stats':
            continue  # Skip the stats entry
            
        report_data[focus_tool] = {
            'absolute_interference': tool_data.get('absolute_interference', False),
            'nodes': {}
        }
        
        # Safely process each node analysis
        nodes = tool_data.get('nodes', {})
        for node_id, analysis in nodes.items():
            # Skip if analysis is not a dictionary (e.g., boolean values)
            if not isinstance(analysis, dict):
                print(f"Warning: Skipping node {node_id} - analysis is not a dictionary: {analysis}")
                continue
                
            if 'error' in analysis:
                # Skip nodes with errors
                continue
            
            # Safely access dictionary values with defaults
            tools_abbr = tools_to_abbreviation(analysis.get('start_tools', []))
            
            report_data[focus_tool]['nodes'][node_id] = {
                'tools': tools_abbr,
                'tool_count': len(analysis.get('start_tools', [])),
                'interference_present': analysis.get('interference_present', False),
                'severity': analysis.get('severity', 'none'),
                'direction': analysis.get('direction', 'none'),
                'absolute_interference': analysis.get('absolute_interference', False),
                'root_cause_level': analysis.get('root_cause_level', 'unknown'),
                'root_cause_type': analysis.get('root_cause_type', 'unknown'),
                'root_cause_node': analysis.get('root_cause_node', ''),
                'interference_source': analysis.get('interference_source', 'unknown'),
                'propagation_path': [
                    {
                        'tools': tools_to_abbreviation(item.get('tools', [])),
                        'tool_count': item.get('tool_count', 0),
                        'has_interference': item.get('has_interference', False),
                        'severity': item.get('severity', 'none'),
                        'direction': item.get('direction', 'none')
                    }
                    for item in analysis.get('propagation_path', [])
                    if isinstance(item, dict)  # Ensure each item is a dictionary
                ]
            }
    
    # Add statistics to the report
    report_data['_stats'] = propagation_results.get('_stats', {})
    
    # Save detailed report
    report_path = os.path.join(output_path, 'interference_propagation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # Generate summary statistics with safe access
    summary = {}
    for focus_tool, tool_data in propagation_results.items():
        if focus_tool == '_stats':
            continue
            
        summary[focus_tool] = {
            'total_nodes': 0,
            'nodes_with_interference': 0,
            'absolute_interference': tool_data.get('absolute_interference', False),
            'root_cause_distribution': defaultdict(int),
            'severity_distribution': defaultdict(int)
        }
        
        nodes = tool_data.get('nodes', {})
        for node_id, analysis in nodes.items():
            if not isinstance(analysis, dict):
                continue
                
            summary[focus_tool]['total_nodes'] += 1
            
            if analysis.get('interference_present', False):
                summary[focus_tool]['nodes_with_interference'] += 1
            
            root_cause_type = analysis.get('root_cause_type', 'unknown')
            summary[focus_tool]['root_cause_distribution'][root_cause_type] += 1
            
            severity = analysis.get('severity', 'none')
            if severity != 'none':
                summary[focus_tool]['severity_distribution'][severity] += 1
    
    # Add focus tool stats from the _stats section
    stats_data = propagation_results.get('_stats', {})
    for focus_tool, stats in stats_data.items():
        if focus_tool in summary:
            summary[focus_tool]['total_nodes_stats'] = stats.get('total_nodes', 0)
            summary[focus_tool]['nodes_with_interference_stats'] = stats.get('nodes_with_interference', 0)
            summary[focus_tool]['interference_percentage'] = stats.get('interference_percentage', 0)
    
    summary_path = os.path.join(output_path, 'interference_propagation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print key insights with safe access
    print("\n=== INTERFERENCE PROPAGATION INSIGHTS ===")
    for focus_tool, stats in summary.items():
        if focus_tool == '_stats':
            continue
            
        print(f"\nFocus Tool: {focus_tool}")
        print(f"Absolute Interference: {stats.get('absolute_interference', False)}")
        print(f"Nodes with interference: {stats.get('nodes_with_interference', 0)}/{stats.get('total_nodes', 0)}")
        print(f"Interference percentage: {stats.get('interference_percentage', 0):.1f}%")
        
        print("Root cause distribution:")
        for cause_type, count in stats.get('root_cause_distribution', {}).items():
            print(f"  {cause_type}: {count}")
        
        print("Severity distribution:")
        for severity, count in stats.get('severity_distribution', {}).items():
            print(f"  {severity}: {count}")
    
    return report_data


def generate_interference_csv_report(propagation_results, output_path):
    """Generate a CSV report of interference propagation results with proper Unicode handling.
    Now includes an 'interference_pattern' column: continuous | sporadic | tapering | isolated | none.
    Backward-compatible, and serializes plural fields by joining with '|'.
    """
    import csv
    import codecs
    import os

    csv_path = os.path.join(output_path, 'interference_propagation_report.csv')

    # Use UTF-8 encoding with BOM to handle Unicode characters properly
    with codecs.open(csv_path, 'w', encoding='utf-8-sig') as csvfile:
        fieldnames = [
            'focus_tool',
            'focus_tool_absolute_interference',
            'node_id',
            'tools_abbreviation',
            'tool_count',
            'has_interference',
            'interference_severity',
            'interference_direction',
            'interference_pattern',          # NEW COLUMN
            'root_cause_type',
            'root_cause_node',              # joined with '|' if multiple
            'root_cause_tools_abbr',        # joined with '|' if multiple
            'inherited_from_node',          # joined with '|' if multiple
            'inherited_from_tools_abbr',    # joined with '|' if multiple
            'propagation_path_length',
            'propagation_path_summary'
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Write focus tool stats first
        stats = propagation_results.get('_stats', {})
        for focus_tool, tool_stats in stats.items():
            writer.writerow({
                'focus_tool': focus_tool,
                'focus_tool_absolute_interference': tool_stats.get('absolute_interference', False),
                'node_id': 'FOCUS_TOOL_STATS',
                'tools_abbreviation': f"Total: {tool_stats.get('total_nodes', 0)}",
                'tool_count': tool_stats.get('total_nodes', 0),
                'has_interference': f"WithInt: {tool_stats.get('nodes_with_interference', 0)}",
                'interference_severity': f"{tool_stats.get('interference_percentage', 0):.1f}%",
                'interference_direction': '',
                'interference_pattern': '',
                'root_cause_type': '',
                'root_cause_node': '',
                'root_cause_tools_abbr': '',
                'inherited_from_node': '',
                'inherited_from_tools_abbr': '',
                'propagation_path_length': '',
                'propagation_path_summary': ''
            })

        # Helper to join a list of tool-lists into abbrs
        def join_tools_abbr(tool_lists):
            return '|'.join(tools_to_abbreviation(t or []) for t in tool_lists)

        # Write individual node data
        for focus_tool, tool_data in propagation_results.items():
            if focus_tool == '_stats':
                continue

            absolute_tool_interference = tool_data.get('absolute_interference', False)

            for node_id, analysis in tool_data.get('nodes', {}).items():
                if not isinstance(analysis, dict):
                    continue

                tools_abbr = tools_to_abbreviation(analysis.get('start_tools', []))

                # ----- Plural-aware extraction with backward compatibility -----
                rc_nodes = analysis.get('root_cause_nodes')
                if not rc_nodes:
                    single_rc = analysis.get('root_cause_node')
                    rc_nodes = [single_rc] if single_rc else []
                rc_tools_list = analysis.get('root_cause_tools_list')
                if not rc_tools_list:
                    single_rc_tools = analysis.get('root_cause_tools')
                    rc_tools_list = [single_rc_tools] if single_rc_tools else []

                inh_nodes = analysis.get('inherited_from_nodes')
                if not inh_nodes:
                    single_inh = analysis.get('inherited_from')
                    inh_nodes = [single_inh] if single_inh else []
                inh_tools_list = analysis.get('inherited_from_tools_list')
                if not inh_tools_list:
                    single_inh_tools = analysis.get('inherited_from_tools')
                    inh_tools_list = [single_inh_tools] if single_inh_tools else []

                rc_nodes_str = '|'.join(rc_nodes) if rc_nodes else ''
                rc_tools_abbr_str = join_tools_abbr(rc_tools_list) if rc_tools_list else ''
                inh_nodes_str = '|'.join(inh_nodes) if inh_nodes else ''
                inh_tools_abbr_str = join_tools_abbr(inh_tools_list) if inh_tools_list else ''

                # ----- Propagation path summary (mark INT; mark earliest root causes with *) -----
                rc_node_set = set(rc_nodes or [])
                propagation_summary = []
                path_items = analysis.get('propagation_path', [])

                for path_item in path_items:
                    if not isinstance(path_item, dict):
                        continue
                    path_tools = tools_to_abbreviation(path_item.get('tools', []))
                    tool_count = path_item.get('tool_count', 0)
                    has_int = path_item.get('has_interference', False)
                    node_in_item = path_item.get('node')  # may be None in older outputs

                    summary_item = f"{path_tools}({tool_count}t)"
                    if has_int:
                        summary_item += "_INT"
                        if node_in_item and node_in_item in rc_node_set:
                            summary_item += "*"
                    propagation_summary.append(summary_item)

                # Add current node
                current_summary = f"{tools_abbr}({len(analysis.get('start_tools', []))}t)"
                if analysis.get('interference_present', False):
                    current_summary += "_INT"
                    if analysis.get('root_cause_type') == 'self_caused':
                        current_summary += "*"
                propagation_summary.append(current_summary)

                path_summary = ' ← '.join(propagation_summary)  # Unicode left arrow

                writer.writerow({
                    'focus_tool': focus_tool,
                    'focus_tool_absolute_interference': absolute_tool_interference,
                    'node_id': node_id,
                    'tools_abbreviation': tools_abbr,
                    'tool_count': len(analysis.get('start_tools', [])),
                    'has_interference': analysis.get('interference_present', False),
                    'interference_severity': analysis.get('severity', 'none'),
                    'interference_direction': analysis.get('direction', 'none'),
                    'interference_pattern': analysis.get('interference_pattern', 'none'),  # NEW
                    'root_cause_type': analysis.get('root_cause_type', 'unknown'),
                    'root_cause_node': rc_nodes_str,
                    'root_cause_tools_abbr': rc_tools_abbr_str,
                    'inherited_from_node': inh_nodes_str,
                    'inherited_from_tools_abbr': inh_tools_abbr_str,
                    'propagation_path_length': len(path_items),
                    'propagation_path_summary': path_summary
                })

    print(f"CSV report saved to: {csv_path}")
    return csv_path

def save_global_interference_csv(interference_results, output_path):
    """Save global interference JSON content into a CSV for easy reading"""
    import csv
    os.makedirs(output_path, exist_ok=True)
    csv_path = os.path.join(output_path, 'global_interference_readable.csv')
    
    fieldnames = [
        'focus_tool', 'focus_combo_id', 'prime_identifier', 'focus_identifier',
        'prime_tools_abbr', 'focus_tools_abbr',
        'interference_detected', 'interference_severity', 'interference_direction',
        'acc_test_clean_rel_diff', 'acc_test_clean_direction', 'acc_test_clean_severity',
        'focus_tool_metric_rel_diff', 'focus_tool_metric_direction', 'focus_tool_metric_severity'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for focus_tool, analyses in interference_results.items():
            for combo_id, analysis in analyses.items():
                writer.writerow({
                    'focus_tool': focus_tool,
                    'focus_combo_id': combo_id,
                    'prime_identifier': analysis['prime_identifier'],
                    'focus_identifier': analysis['focus_identifier'],
                    'prime_tools_abbr': tools_to_abbreviation(analysis['prime_tools']),
                    'focus_tools_abbr': tools_to_abbreviation(analysis['focus_tools']),
                    'interference_detected': analysis.get('interference_detected', False),
                    'interference_severity': analysis.get('interference_severity', 'none'),
                    'interference_direction': analysis.get('interference_direction', 'none'),
                    'acc_test_clean_rel_diff': analysis.get('acc_test_clean_rel_diff', None),
                    'acc_test_clean_direction': analysis.get('acc_test_clean_direction', 'none'),
                    'acc_test_clean_severity': analysis.get('acc_test_clean_severity', 'negligible'),
                    'focus_tool_metric_rel_diff': analysis.get('focus_tool_metric_rel_diff', None),
                    'focus_tool_metric_direction': analysis.get('focus_tool_metric_direction', 'none'),
                    'focus_tool_metric_severity': analysis.get('focus_tool_metric_severity', 'negligible')
                })
    
    print(f"Global interference CSV saved to: {csv_path}")
    return csv_path



def main():
    args = parse_arguments()
    os.makedirs(args.output, exist_ok=True)
    df = pd.read_csv(args.input)

    # --- Create all_combos ---
    all_combos = {}
    for _, row in df.iterrows():
        combo_id = row['combination']
        metrics = {c: row[c] for c in df.columns if c not in ['combination','combination_status','dataset_name','dataset_type']}
        ordered_tools = get_ordered_tools(row)
        all_combos[combo_id] = {
            'identifier': create_ordered_toolset_identifier(row),
            'ordered_tools': ordered_tools,
            'metrics': metrics,
            'tool_count': len(ordered_tools),
            'original_data': row.to_dict()
        }

    # Mapping for prime combos
    tool_list_to_combo = {}
    for combo_id, data in all_combos.items():
        ttuple = tuple(data['ordered_tools'])
        tool_list_to_combo.setdefault(ttuple, []).append(combo_id)

    # --- Global interference ---
    interference_results = {}
    focus_tools = [f for f in TOOL_METRICS.keys() if f != 'no_tools']
    for focus_tool in focus_tools:
        interference_results[focus_tool] = {}
        print(f"Analyzing focus tool: {focus_tool}")
        focus_combos = {cid:data for cid,data in all_combos.items() if focus_tool in data['ordered_tools']}
        for focus_combo_id, focus_combo_data in focus_combos.items():
            focus_list = focus_combo_data['ordered_tools']
            try:
                idx = focus_list.index(focus_tool)
                prime_tools = focus_list[:idx] + focus_list[idx+1:]
            except ValueError:
                continue
            prime_tuple = tuple(prime_tools)
            if prime_tuple in tool_list_to_combo:
                prime_combo_id = tool_list_to_combo[prime_tuple][0]
                prime_combo = all_combos[prime_combo_id]
                analysis = analyze_global_interference(prime_combo, focus_combo_data, focus_tool, args.t1, args.t2)
                if analysis:
                    interference_results[focus_tool][focus_combo_id] = analysis
            else:
                print(f"No prime found for focus combo {focus_combo_id} with tools {focus_list}")

    # --- Save results and generate trees/CSV/JSON---

    # Save results
    results_path = os.path.join(args.output, 'global_interference_analysis.json')
    with open(results_path, 'w') as f:
        json.dump(interference_results, f, indent=2, default=str)
    
    # Create summary statistics
    summary = {}
    for focus_tool, analyses in interference_results.items():
        summary[focus_tool] = {
            'total_analyses': len(analyses),
            'interference_detected': sum(1 for a in analyses.values() if a['interference_detected']),
            'severity_counts': defaultdict(int),
            'direction_counts': defaultdict(int),
            'performance_counts': defaultdict(int)
        }
        
        for analysis in analyses.values():
            summary[focus_tool]['severity_counts'][analysis['interference_severity']] += 1
            summary[focus_tool]['direction_counts'][analysis['interference_direction']] += 1
            summary[focus_tool]['performance_counts'][analysis['focus_tool_performance']] += 1
    
    summary_path = os.path.join(args.output, 'interference_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create comprehensive trees with horizontal and vertical relationships
    create_comprehensive_trees(interference_results, all_combos, args.output)

    propagation_results = analyze_interference_propagation(interference_results, all_combos)
    
    # Generate JSON report
    generate_interference_report(propagation_results, args.output)
    
    # Generate CSV reports
    csv_report = generate_interference_csv_report(propagation_results, args.output)
    
    print("=== GLOBAL INTERFERENCE ANALYSIS COMPLETE ===")
    print(f"Results saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Interference propagation report saved to: {os.path.join(args.output, 'interference_propagation_report.json')}")
    print(f"CSV report saved to: {csv_report}")
    print(f"Comprehensive trees saved to: {os.path.join(args.output, 'comprehensive_trees')}")

    # Save a readable CSV version of global interference results
    save_global_interference_csv(interference_results, args.output)

        
    print(f"\n=== SUMMARY ===")
    for tool, stats in summary.items():
        print(f"{tool}:")
        print(f"  Total analyses: {stats['total_analyses']}")
        print(f"  Interference detected: {stats['interference_detected']}")
        print(f"  Severity: {dict(stats['severity_counts'])}")
        print(f"  Direction: {dict(stats['direction_counts'])}")
        print(f"  Performance: {dict(stats['performance_counts'])}")


if __name__ == "__main__":
    main()



