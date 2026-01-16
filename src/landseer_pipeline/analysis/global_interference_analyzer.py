#!/usr/bin/env python3
"""
Global Interference Analysis for Landseer Pipeline.

This module analyzes tool interactions and interference patterns in ML defense pipelines.
It's designed to integrate with the Landseer pipeline system and automatically run after
pipeline execution completes.

The analysis:
- Parses tools from pipeline results CSV
- Computes interference metrics for each tool combination
- Generates detailed interference reports and summaries
- Provides insights into tool composability and conflicts
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Default metrics configuration for Landseer pipeline
DEFAULT_METRICS = {
    "acc_test_clean": {
        "aliases": ["acc_test_clean", "acc.clean", "acc_clean", "clean_acc", "acc_test"],
        "higher_is_better": True,
    },
    "asr": {
        "aliases": ["asr", "attack_success_rate", "attack_sr"],
        "higher_is_better": False,  # lower is better
    },
    "ood_auc": {
        "aliases": ["ood_auc", "auroc_ood", "ood.auroc", "auroc_ood_maxconf"],
        "higher_is_better": True,
    },
    "pgd_acc": {
        "aliases": ["acc_robust", "acc_roboust", "robust_acc", "pgd_acc", "acc.pgd", "acc_roboust"],
        "higher_is_better": True,
    },
    "carlini_robustness": {
        "aliases": ["carlini_robustness", "carlini_acc"],
        "higher_is_better": True,
    },
    "fingerprinting": {
        "aliases": ["fingerprinting", "dataset_fingerprinting", "fp_score", "fingerprint_score"],
        "higher_is_better": True,
    },
    "privacy_epsilon": {
        "aliases": ["privacy_epsilon", "dp_epsilon", "epsilon", "eps"],
        "higher_is_better": False,  # LOWER epsilon = better privacy
    },
    "mia_auc": {
        "aliases": ["mia_auc", "membership_inference_auc", "mi_auc"],
        "higher_is_better": False,  # lower is better
    },
    "watermark_accuracy": {
        "aliases": ["watermark_accuracy", "watermark_acc", "wm_acc"],
        "higher_is_better": True,
    },
}

# Tool to main metric mapping for Landseer tools
DEFAULT_TOOL_MAIN = {
    "pre-xgbod": "ood_auc",
    "in-trades": "pgd_acc", 
    "fine pruning": "asr",
    "neuronprune": "asr",  # Added for Landseer
    "post_magnet": "carlini_robustness",
    "dataset_inference": "fingerprinting",
    "deploy_dp": "privacy_epsilon",
    "in-teaching": "acc_test_clean",
    "in-dp": "mia_auc",
    "watermarking": "watermark_accuracy",
    "in_noop": "acc_test_clean",  # Default for no-op tools
    "noop": "acc_test_clean",
}

# Heuristic patterns for tool main metric inference
HEURISTIC_TOOL_MAIN = [
    (re.compile(r"xgbod", re.I), "ood_auc"),
    (re.compile(r"trades", re.I), "pgd_acc"),
    (re.compile(r"magnet", re.I), "carlini_robustness"),
    (re.compile(r"prun", re.I), "asr"),  # neuronprune, fine-pruning
    (re.compile(r"\\bdeploy[_-]?dp\\b", re.I), "privacy_epsilon"),
    (re.compile(r"\\bdataset[_-]?inference\\b", re.I), "fingerprinting"),
    (re.compile(r"\\bin[-_]?teaching\\b", re.I), "acc_test_clean"),
    (re.compile(r"\\bin[-_]?dp\\b", re.I), "mia_auc"),
    (re.compile(r"\\bwatermark(ing)?\\b", re.I), "watermark_accuracy"),
    (re.compile(r"noop", re.I), "acc_test_clean"),
]

# Token splitter for parsing tool lists
TOKEN_SPLIT_RE = re.compile(r"[+|;,/\\[\\]'\"]+")


class GlobalInterferenceAnalyzer:
    """
    Analyzes global interference patterns in ML defense pipeline results.
    
    This class processes pipeline results CSV files to identify how different
    defense tools interact with each other, computing interference metrics
    and generating comprehensive reports.
    """
    
    def __init__(self, 
                 metrics_config: Optional[Dict] = None,
                 tool_config: Optional[Dict] = None,
                 t1: float = 0.02,
                 t2: float = 0.05):
        """
        Initialize the analyzer.
        
        Args:
            metrics_config: Configuration for metrics (aliases, higher_is_better)
            tool_config: Mapping from tool names to their main metrics
            t1: Threshold for negligible vs moderate interference (default 2%)
            t2: Threshold for moderate vs severe interference (default 5%)
        """
        self.metrics_config = metrics_config or DEFAULT_METRICS
        self.tool_config = tool_config or DEFAULT_TOOL_MAIN
        self.t1 = t1
        self.t2 = t2
        
        # Build alias mapping for metrics
        self.alias_map = self._build_alias_map()
        
        # Set of metric tokens to avoid treating metrics as tools
        self.metric_tokens = set(self.alias_map.keys()) | {m.lower() for m in self.metrics_config.keys()}
        
    def _build_alias_map(self) -> Dict[str, str]:
        """Build mapping from metric aliases to canonical names."""
        alias_map = {}
        for canon, spec in self.metrics_config.items():
            for alias in spec.get("aliases", []):
                alias_map[alias.lower()] = canon
            alias_map[canon.lower()] = canon
        return alias_map
        
    def _normalize_token(self, token: str, normalize_map: Dict[str, str] = None) -> str:
        """Normalize tool token for consistent comparison."""
        if normalize_map is None:
            normalize_map = {}
            
        # Basic cleaning
        t = token.strip().lower()
        t = re.sub(r'[\\[\\]\"\\`]+', '', t)  # Remove wrapper chars
        t = re.sub(r"\\s+", "_", t)  # Normalize spaces to underscores
        
        # Handle stage-specific noops
        if t.endswith("_noop"):
            t = "noop"
            
        # Apply user normalizations
        if t in normalize_map:
            t = normalize_map[t]
            
        # Standard normalizations for Landseer
        fixes = {
            "finepruning": "fine_pruning",
            "fine_pruning": "neuronprune",  # Standardize to neuronprune
            "fine-prune": "neuronprune",
            "post-magnet": "post_magnet",
            "intrades": "in-trades",
            "in_teaching": "in-teaching",
            "inteaching": "in-teaching", 
            "in_dp": "in-dp",
            "indp": "in-dp",
            "deploy-dp": "deploy_dp",
            "dataset-inference": "dataset_inference",
            "watermark": "watermarking",
        }
        
        if t in fixes:
            t = fixes[t]
            
        return t
        
    def _parse_tools_from_row(self, row: pd.Series, normalize_map: Dict[str, str] = None) -> Set[str]:
        """
        Parse tools from a pipeline results row.
        
        This handles the Landseer pipeline format where tools are stored in 
        stage-specific columns (pre_training, in_training, post_training, deployment).
        """
        if normalize_map is None:
            normalize_map = {}
            
        tools = set()
        
        # Landseer stage columns
        stage_columns = ["pre_training", "in_training", "post_training", "deployment"]
        
        for col in stage_columns:
            if col not in row.index:
                continue
                
            cell = row[col]
            if pd.isna(cell) or str(cell).strip().lower() in ["nan", "", "none"]:
                continue
                
            # Handle list format: ['tool1', 'tool2'] or string format
            cell_str = str(cell).strip()
            
            # Parse list format
            if cell_str.startswith('[') and cell_str.endswith(']'):
                # Remove brackets and quotes, split by comma
                inner = cell_str[1:-1]
                tokens = [t.strip().strip("'\"") for t in inner.split(',')]
            else:
                # Split by common delimiters
                tokens = TOKEN_SPLIT_RE.split(cell_str)
                
            for token in tokens:
                if not token:
                    continue
                    
                norm_token = self._normalize_token(token, normalize_map)
                
                # Skip metrics and noops
                if norm_token in self.metric_tokens:
                    continue
                if norm_token in {"noop", "pre-noop", "in-noop", "post-noop", "deployment-noop"}:
                    continue
                    
                tools.add(norm_token)
                
        return tools
        
    def _tools_key(self, tools: Set[str]) -> str:
        """Create a consistent key for a set of tools."""
        return "+".join(sorted(tools))
        
    def _resolve_metric_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Map canonical metric names to actual DataFrame columns."""
        lower_cols = {c.lower(): c for c in df.columns}
        resolved = {}
        
        for alias_lower, canon in self.alias_map.items():
            if alias_lower in lower_cols and canon not in resolved:
                resolved[canon] = lower_cols[alias_lower]
                
        return resolved
        
    def _find_combo_id_col(self, df_cols: List[str]) -> Optional[str]:
        """Find the combination ID column."""
        for col in df_cols:
            if "comb" in col.lower():
                return col
        return None
        
    def _get_tool_main_metric(self, tool: str) -> Optional[str]:
        """Get the main metric for a tool."""
        if tool in self.tool_config:
            return self.tool_config[tool]
            
        # Try heuristic patterns
        for pattern, metric in HEURISTIC_TOOL_MAIN:
            if pattern.search(tool):
                return metric
                
        return None
        
    def _desirability_change(self, to_val: float, from_val: float, higher_is_better: bool) -> float:
        """Calculate desirability-aligned change (positive = better)."""
        return (to_val - from_val) if higher_is_better else (from_val - to_val)
        
    def _basic_label(self, change: float) -> str:
        """Get basic better/worse/equal label."""
        if change > 1e-12:
            return "better"
        elif change < -1e-12:
            return "worse"
        else:
            return "equal"
            
    def _threshold_label(self, raw_delta: float, baseline_from: float, 
                        sign_label: str) -> str:
        """
        Classify change using relative thresholds.
        
        Args:
            raw_delta: Raw change (to - from)
            baseline_from: Baseline value for relative thresholding
            sign_label: Basic better/worse/equal label
            
        Returns:
            Threshold-based label: equal, pos_moderate, neg_moderate, pos_severe, neg_severe
        """
        denom = max(abs(baseline_from), 1e-12)
        pct = abs(raw_delta) / denom
        
        if sign_label == "equal" or pct < self.t1:
            return "equal"
        elif pct < self.t2:
            return "pos_moderate" if sign_label == "better" else "neg_moderate"
        else:
            return "pos_severe" if sign_label == "better" else "neg_severe"
            
    def _get_metrics(self, row: pd.Series, metric_cols: Dict[str, str]) -> Dict[str, float]:
        """Extract metric values from a row, handling -1 as NaN."""
        vals = {}
        for canon, col in metric_cols.items():
            v = float(row[col])
            vals[canon] = float("nan") if v == -1 else v
        return vals
        
    def _output_metric_name(self, metric: str) -> str:
        """Convert metric name for output (acc_test_clean -> acc_test)."""
        return "acc_test" if metric == "acc_test_clean" else metric
        
    def analyze(self, 
                input_csv: str, 
                output_dir: str,
                focus_tools: Optional[List[str]] = None,
                extra_normalize: Optional[Dict[str, str]] = None) -> Dict:
        """
        Perform global interference analysis on pipeline results.
        
        Args:
            input_csv: Path to results CSV file
            output_dir: Directory for output files
            focus_tools: Specific tools to analyze (default: all)
            extra_normalize: Additional token normalizations
            
        Returns:
            Dictionary with analysis results and file paths
        """
        logger.info(f"Starting global interference analysis on {input_csv}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and prepare data
        df = pd.read_csv(input_csv)
        logger.info(f"Loaded {len(df)} combinations from CSV")
        
        # Setup normalization
        normalize_map = extra_normalize or {}
        
        # Resolve metrics
        metric_cols = self._resolve_metric_columns(df)
        resolved_metrics = list(metric_cols.keys())
        
        if not resolved_metrics:
            logger.warning("No known metric columns found in CSV")
            
        logger.info(f"Resolved metrics: {resolved_metrics}")
        
        # Find combination ID column
        combo_id_col = self._find_combo_id_col(list(df.columns))
        df["_combo_id"] = df[combo_id_col] if combo_id_col else df.index.astype(str)
        
        # Parse tools from each row
        toolsets = []
        for _, row in df.iterrows():
            tools = self._parse_tools_from_row(row, normalize_map)
            toolsets.append(tools)
            
        df["_tools_set"] = toolsets
        df["_tools_key"] = df["_tools_set"].apply(self._tools_key)
        
        # Build key mappings
        key_to_indices = {}
        key_to_combo_id = {}
        for idx, key in enumerate(df["_tools_key"].tolist()):
            key_to_indices.setdefault(key, []).append(idx)
            if key not in key_to_combo_id:
                key_to_combo_id[key] = str(df["_combo_id"].iloc[idx])
                
        # Determine observed tools
        observed_tools = set().union(*df["_tools_set"].tolist())
        logger.info(f"Observed tools: {sorted(observed_tools)}")
        
        # Select focus tools
        if focus_tools:
            focus_tools = [t for t in focus_tools if t in observed_tools]
        else:
            focus_tools = sorted(observed_tools)
            
        logger.info(f"Analyzing tools: {focus_tools}")
        
        # Perform analysis
        results = self._analyze_combinations(
            df, focus_tools, metric_cols, resolved_metrics,
            key_to_indices, key_to_combo_id
        )
        
        # Generate outputs
        output_files = self._generate_outputs(results, output_dir, resolved_metrics)
        
        logger.info(f"Analysis complete. Generated {len(output_files)} output files.")
        
        return {
            "success": True,
            "combinations_analyzed": len(results),
            "tools_analyzed": len(focus_tools),
            "output_files": output_files,
            "summary": self._generate_summary(results)
        }
        
    def _analyze_combinations(self, df: pd.DataFrame, focus_tools: List[str],
                            metric_cols: Dict[str, str], resolved_metrics: List[str],
                            key_to_indices: Dict[str, List[int]], 
                            key_to_combo_id: Dict[str, str]) -> List[Dict]:
        """Perform the core interference analysis."""
        results = []
        
        for tool in focus_tools:
            for ridx, row in df.iterrows():
                combo_tools = row["_tools_set"]
                if tool not in combo_tools:
                    continue
                    
                combo_metrics = self._get_metrics(row, metric_cols)
                combo_id = str(row["_combo_id"])
                
                # Find prime (combo without focus tool)
                prime_tools = set(combo_tools)
                prime_tools.discard(tool)
                prime_key = self._tools_key(prime_tools)
                
                p_indices = key_to_indices.get(prime_key, [])
                if not p_indices:
                    # No prime found
                    results.append({
                        "focus_tool": tool,
                        "combo_index": combo_id, 
                        "combo_tools": "+".join(sorted(combo_tools)),
                        "prime_missing": True,
                        "prime_index": None,
                        "prime_tools": "+".join(sorted(prime_tools)),
                    })
                    continue
                    
                # Analyze with prime
                pidx = p_indices[0]
                prime_row = df.iloc[pidx]
                prime_metrics = self._get_metrics(prime_row, metric_cols)
                prime_id = key_to_combo_id.get(prime_key, str(prime_row["_combo_id"]))
                
                # Compute changes and labels for each metric
                changes = {}
                labels = {}
                
                for metric in resolved_metrics:
                    spec = self.metrics_config[metric]
                    higher_is_better = bool(spec.get("higher_is_better", True))
                    
                    to_val = combo_metrics.get(metric, float("nan"))
                    from_val = prime_metrics.get(metric, float("nan"))
                    
                    if pd.isna(to_val) or pd.isna(from_val):
                        change = 0.0
                        label = "equal"
                    else:
                        change = self._desirability_change(to_val, from_val, higher_is_better)
                        sign_label = self._basic_label(change)
                        raw_delta = to_val - from_val
                        label = self._threshold_label(raw_delta, from_val, sign_label)
                        
                    changes[metric] = change
                    labels[metric] = label
                    
                # Compute composability (sum of all changes)
                composability = sum(changes.values()) if changes else 0.0
                
                # Analyze global interference
                interference_analysis = self._analyze_global_interference(
                    prime_tools, combo_metrics, prime_metrics, metric_cols
                )
                
                # Tool effect analysis 
                tool_effect = self._analyze_tool_effect(
                    tool, combo_metrics, prime_metrics, metric_cols
                )
                
                # Assemble result
                result = {
                    "focus_tool": tool,
                    "combo_index": combo_id,
                    "combo_tools": "+".join(sorted(combo_tools)),
                    "prime_missing": False,
                    "prime_index": prime_id,
                    "prime_tools": "+".join(sorted(prime_tools)),
                    "composability": composability,
                    **interference_analysis,
                    **tool_effect
                }
                
                # Add metric changes and labels
                for metric in resolved_metrics:
                    m_out = self._output_metric_name(metric)
                    result[f"change_{m_out}"] = changes[metric]
                    result[f"label_{m_out}"] = labels[metric]
                    
                results.append(result)
                
        return results
        
    def _analyze_global_interference(self, prime_tools: Set[str], 
                                   combo_metrics: Dict[str, float],
                                   prime_metrics: Dict[str, float],
                                   metric_cols: Dict[str, str]) -> Dict:
        """Analyze global interference patterns."""
        interference_details = []
        has_mod, has_sev = False, False
        pos_flag, neg_flag = False, False
        
        if len(prime_tools) == 0:
            # All-noop prime - use acc_test_clean
            if "acc_test_clean" in metric_cols:
                to_val = combo_metrics["acc_test_clean"]
                from_val = prime_metrics["acc_test_clean"]
                if not (pd.isna(to_val) or pd.isna(from_val)):
                    change = self._desirability_change(to_val, from_val, True)
                    sign_label = self._basic_label(change)
                    raw_delta = to_val - from_val
                    thr_label = self._threshold_label(raw_delta, from_val, sign_label)
                    
                    interference_details.append(f"baseline:acc_test_{thr_label}")
                    
                    if thr_label.endswith("moderate"):
                        has_mod = True
                    if thr_label.endswith("severe"):
                        has_sev = True
                    if thr_label.startswith("pos_"):
                        pos_flag = True
                    if thr_label.startswith("neg_"):
                        neg_flag = True
        else:
            # Check each prime tool's main metric
            for other_tool in prime_tools:
                main_metric = self._get_tool_main_metric(other_tool)
                if not main_metric or main_metric not in metric_cols:
                    continue
                    
                to_val = combo_metrics[main_metric]
                from_val = prime_metrics[main_metric]
                
                if pd.isna(to_val) or pd.isna(from_val):
                    continue
                    
                higher_is_better = bool(self.metrics_config[main_metric]["higher_is_better"])
                change = self._desirability_change(to_val, from_val, higher_is_better)
                sign_label = self._basic_label(change)
                raw_delta = to_val - from_val
                thr_label = self._threshold_label(raw_delta, from_val, sign_label)
                
                metric_out = self._output_metric_name(main_metric)
                interference_details.append(f"{other_tool}:{metric_out}_{thr_label}")
                
                if thr_label.endswith("moderate"):
                    has_mod = True
                if thr_label.endswith("severe"):
                    has_sev = True
                if thr_label.startswith("pos_"):
                    pos_flag = True
                if thr_label.startswith("neg_"):
                    neg_flag = True
                    
        # Determine global interference level
        if not interference_details or all(s.endswith("_equal") for s in interference_details):
            global_interference = "False"
            interference_kind = "none"
        else:
            if pos_flag and not neg_flag:
                interference_kind = "positive"
            elif neg_flag and not pos_flag:
                interference_kind = "negative"
            elif pos_flag and neg_flag:
                interference_kind = "mixed"
            else:
                interference_kind = "none"
                
            if has_sev and has_mod:
                global_interference = "Mixed True"
            elif has_sev:
                global_interference = "Severe True"
            elif has_mod:
                global_interference = "Moderate True"
            else:
                global_interference = "False"
                
        return {
            "global_interference": global_interference,
            "interference_kind": interference_kind,
            "interference_causes": ";".join(interference_details)
        }
        
    def _analyze_tool_effect(self, tool: str, combo_metrics: Dict[str, float],
                           prime_metrics: Dict[str, float],
                           metric_cols: Dict[str, str]) -> Dict:
        """Analyze the effect of adding the focus tool."""
        main_metric = self._get_tool_main_metric(tool)
        
        if not main_metric or main_metric not in metric_cols:
            return {"tool_effect_label": "unknown_equal"}
            
        to_val = combo_metrics[main_metric]
        from_val = prime_metrics[main_metric]
        
        if pd.isna(to_val) or pd.isna(from_val):
            return {"tool_effect_label": "unknown_equal"}
            
        higher_is_better = bool(self.metrics_config[main_metric]["higher_is_better"])
        change = self._desirability_change(to_val, from_val, higher_is_better)
        sign_label = self._basic_label(change)
        
        metric_out = self._output_metric_name(main_metric)
        
        return {"tool_effect_label": f"{metric_out}_{sign_label}"}
        
    def _generate_outputs(self, results: List[Dict], output_dir: str, 
                         resolved_metrics: List[str]) -> List[str]:
        """Generate analysis output files."""
        output_files = []
        
        # Create detailed results DataFrame
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            # Define column order
            metric_cols_out = []
            for metric in resolved_metrics:
                m_out = self._output_metric_name(metric)
                metric_cols_out += [f"change_{m_out}", f"label_{m_out}"]
                
            ordered_cols = [
                "focus_tool", "combo_index", "combo_tools", "prime_missing", 
                "prime_index", "prime_tools"
            ] + metric_cols_out + [
                "composability", "global_interference", "interference_kind", 
                "interference_causes", "tool_effect_label"
            ]
            
            # Ensure all columns exist
            for col in ordered_cols:
                if col not in results_df.columns:
                    results_df[col] = None
                    
            results_df = results_df[ordered_cols]
            
        # Write detailed results
        detailed_path = os.path.join(output_dir, "global_interference_results.csv")
        results_df.to_csv(detailed_path, index=False)
        output_files.append(detailed_path)
        
        # Generate summary
        summary_path = os.path.join(output_dir, "global_interference_summary_by_tool.csv")
        if not results_df.empty:
            analyzed = results_df[results_df.get("prime_missing", True) == False].copy()
            
            if not analyzed.empty:
                agg_data = []
                for tool in analyzed["focus_tool"].unique():
                    tool_data = analyzed[analyzed["focus_tool"] == tool]
                    
                    interference_pct = (
                        tool_data["global_interference"]
                        .isin(["Moderate True", "Severe True", "Mixed True"])
                        .mean() * 100
                    )
                    
                    # Tool effect distribution
                    effect_counts = tool_data["tool_effect_label"].str.split("_").str[-1].value_counts()
                    total_effects = len(tool_data)
                    
                    agg_data.append({
                        "focus_tool": tool,
                        "combos_analyzed": len(tool_data),
                        "mean_composability": tool_data["composability"].mean(),
                        "median_composability": tool_data["composability"].median(),
                        "pct_global_interference": interference_pct,
                        "tool_effect_better": effect_counts.get("better", 0) / total_effects * 100,
                        "tool_effect_worse": effect_counts.get("worse", 0) / total_effects * 100,
                        "tool_effect_equal": effect_counts.get("equal", 0) / total_effects * 100,
                    })
                    
                summary_df = pd.DataFrame(agg_data)
            else:
                summary_df = pd.DataFrame(columns=[
                    "focus_tool", "combos_analyzed", "mean_composability", 
                    "median_composability", "pct_global_interference",
                    "tool_effect_better", "tool_effect_worse", "tool_effect_equal"
                ])
        else:
            summary_df = pd.DataFrame(columns=[
                "focus_tool", "combos_analyzed", "mean_composability",
                "median_composability", "pct_global_interference", 
                "tool_effect_better", "tool_effect_worse", "tool_effect_equal"
            ])
            
        summary_df.to_csv(summary_path, index=False)
        output_files.append(summary_path)
        
        # Write README
        readme_path = os.path.join(output_dir, "README.txt")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(self._generate_readme())
        output_files.append(readme_path)
        
        return output_files
        
    def _generate_summary(self, results: List[Dict]) -> Dict:
        """Generate a summary of analysis results."""
        if not results:
            return {"total_combinations": 0, "tools_analyzed": 0}
            
        df = pd.DataFrame(results)
        analyzed = df[df.get("prime_missing", True) == False]
        
        if analyzed.empty:
            return {
                "total_combinations": len(results),
                "analyzed_combinations": 0,
                "tools_analyzed": len(df["focus_tool"].unique()),
                "interference_rate": 0.0
            }
            
        interference_rate = (
            analyzed["global_interference"]
            .isin(["Moderate True", "Severe True", "Mixed True"])
            .mean()
        )
        
        return {
            "total_combinations": len(results),
            "analyzed_combinations": len(analyzed),
            "tools_analyzed": len(df["focus_tool"].unique()),
            "interference_rate": interference_rate,
            "mean_composability": analyzed["composability"].mean(),
            "tools_with_interference": len(analyzed[
                analyzed["global_interference"].isin(["Moderate True", "Severe True", "Mixed True"])
            ]["focus_tool"].unique())
        }
        
    def _generate_readme(self) -> str:
        """Generate README content for analysis outputs."""
        return f"""
Global Interference Analysis Results
===================================

This directory contains the results of global interference analysis performed on 
the Landseer ML defense pipeline results.

Files:
------
- global_interference_results.csv: Detailed per-combination analysis
- global_interference_summary_by_tool.csv: Summary statistics by tool
- README.txt: This file

Analysis Parameters:
-------------------
- Negligible threshold (t1): {self.t1:.1%}
- Moderate/Severe threshold (t2): {self.t2:.1%}

Column Descriptions:
-------------------

global_interference_results.csv:
- focus_tool: Tool being analyzed for interference
- combo_index: Combination identifier
- combo_tools: All tools in the combination
- prime_missing: Whether the prime (combination without focus tool) exists
- prime_index: Identifier of the prime combination
- prime_tools: Tools in the prime combination
- change_<metric>: Numeric change in metric (positive = better)
- label_<metric>: Threshold-based label (equal/pos_moderate/neg_moderate/pos_severe/neg_severe)
- composability: Sum of all metric changes
- global_interference: Overall interference classification
- interference_kind: Direction of interference (positive/negative/mixed/none)
- interference_causes: Detailed breakdown of interference sources
- tool_effect_label: Effect of adding the focus tool

global_interference_summary_by_tool.csv:
- focus_tool: Tool name
- combos_analyzed: Number of combinations analyzed
- mean_composability: Average composability score
- median_composability: Median composability score
- pct_global_interference: Percentage of combinations showing interference
- tool_effect_better/worse/equal: Distribution of tool effects

Interpretation:
--------------
- Positive composability indicates overall improvement when combining tools
- Negative composability indicates overall degradation
- Global interference "True" indicates significant tool interactions
- interference_causes provides specific details about which tools/metrics are affected

For questions about this analysis, consult the Landseer documentation.
"""


def run_analysis_cli():
    """Command-line interface for the analyzer."""
    parser = argparse.ArgumentParser(description="Global Interference Analysis for Landseer Pipeline")
    parser.add_argument("--input", required=True, help="Path to results CSV file")
    parser.add_argument("--outdir", required=True, help="Output directory for analysis results")
    parser.add_argument("--tool-config", help="JSON file with tool->metric mappings")
    parser.add_argument("--metrics-config", help="JSON file with metric configurations")
    parser.add_argument("--tools", nargs="*", help="Specific tools to analyze")
    parser.add_argument("--t1", type=float, default=0.02, help="Negligible threshold (default: 0.02)")
    parser.add_argument("--t2", type=float, default=0.05, help="Severe threshold (default: 0.05)")
    parser.add_argument("--extra-normalize", nargs="*", default=[], 
                       help="Extra normalizations as key=value pairs")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load configurations
    tool_config = None
    if args.tool_config:
        with open(args.tool_config, 'r') as f:
            tool_config = json.load(f)
            
    metrics_config = None
    if args.metrics_config:
        with open(args.metrics_config, 'r') as f:
            metrics_config = json.load(f)
            
    # Parse normalizations
    extra_normalize = {}
    for kv in args.extra_normalize:
        if "=" in kv:
            k, v = kv.split("=", 1)
            extra_normalize[k.strip().lower()] = v.strip().lower()
            
    # Run analysis
    analyzer = GlobalInterferenceAnalyzer(
        metrics_config=metrics_config,
        tool_config=tool_config,
        t1=args.t1,
        t2=args.t2
    )
    
    results = analyzer.analyze(
        input_csv=args.input,
        output_dir=args.outdir,
        focus_tools=args.tools,
        extra_normalize=extra_normalize
    )
    
    if results["success"]:
        print(f"Analysis completed successfully!")
        print(f"Analyzed {results['combinations_analyzed']} combinations")
        print(f"Output files: {results['output_files']}")
    else:
        print("Analysis failed!")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(run_analysis_cli())
