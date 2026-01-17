"""
Query Helper Functions for Landseer Pipeline Database

Provides convenient high-level query functions for common analysis tasks.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from decimal import Decimal

from sqlalchemy import create_engine, and_, or_, func, desc, asc, text
from sqlalchemy.orm import sessionmaker, Session

from .models import (
    Dataset, Model, Tool, ToolCategory, PipelineRun, PipelineAttacks,
    Combination, CombinationTools, CombinationMetrics, ToolExecution,
    ArtifactNode, StageEnum, RunStatusEnum, CombinationStatusEnum
)


class QueryHelper:
    """
    High-level query helper for common analysis tasks.
    
    Usage:
        helper = QueryHelper("mysql+mysqlconnector://user:pass@host/db")
        
        # Find best performing combinations
        results = helper.find_best_combinations(
            metric="acc_test_clean",
            dataset="cifar10",
            limit=10
        )
        
        # Compare tool effectiveness
        comparison = helper.compare_tools(
            tools=["in-trades", "in-pgd"],
            metrics=["acc_test_clean", "pgd_accuracy"]
        )
    """
    
    def __init__(self, database_url: str):
        """Initialize query helper with database connection."""
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def _get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    # ============================================================
    # Pipeline Run Queries
    # ============================================================
    
    def get_run_summary(self, run_id: int) -> Optional[Dict[str, Any]]:
        """Get comprehensive summary of a pipeline run."""
        session = self._get_session()
        try:
            run = session.query(PipelineRun).filter(
                PipelineRun.run_id == run_id
            ).first()
            
            if not run:
                return None
            
            # Get aggregate metrics
            metrics_result = session.query(
                func.count(Combination.combination_id).label('total'),
                func.avg(CombinationMetrics.acc_test_clean).label('avg_acc'),
                func.max(CombinationMetrics.acc_test_clean).label('max_acc'),
                func.avg(CombinationMetrics.pgd_accuracy).label('avg_pgd'),
                func.max(CombinationMetrics.pgd_accuracy).label('max_pgd'),
            ).join(
                CombinationMetrics,
                CombinationMetrics.combination_id == Combination.combination_id
            ).filter(
                Combination.run_id == run_id,
                Combination.status == CombinationStatusEnum.SUCCESS
            ).first()
            
            # Get cache stats
            cache_result = session.query(
                func.count(ToolExecution.execution_id).label('total'),
                func.sum(func.if_(ToolExecution.cache_hit, 1, 0)).label('hits'),
            ).join(Combination).filter(
                Combination.run_id == run_id
            ).first()
            
            return {
                'run_id': run.run_id,
                'pipeline_id': run.pipeline_id,
                'run_timestamp': run.run_timestamp.isoformat() if run.run_timestamp else None,
                'status': run.status.value if run.status else None,
                'dataset': run.dataset.dataset_name if run.dataset else None,
                'dataset_variant': run.dataset.variant if run.dataset else None,
                'total_combinations': run.total_combinations,
                'successful_combinations': run.successful_combinations,
                'failed_combinations': run.failed_combinations,
                'success_rate': round(run.success_rate, 2) if run.total_combinations > 0 else 0,
                'metrics': {
                    'avg_accuracy': float(metrics_result.avg_acc) if metrics_result.avg_acc else None,
                    'max_accuracy': float(metrics_result.max_acc) if metrics_result.max_acc else None,
                    'avg_pgd_accuracy': float(metrics_result.avg_pgd) if metrics_result.avg_pgd else None,
                    'max_pgd_accuracy': float(metrics_result.max_pgd) if metrics_result.max_pgd else None,
                },
                'cache': {
                    'total_executions': cache_result.total if cache_result else 0,
                    'cache_hits': int(cache_result.hits) if cache_result and cache_result.hits else 0,
                    'cache_hit_rate': round(
                        (int(cache_result.hits) / cache_result.total * 100) 
                        if cache_result and cache_result.total and cache_result.hits else 0, 2
                    ),
                },
            }
        finally:
            session.close()
    
    def list_recent_runs(
        self,
        limit: int = 10,
        dataset: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List recent pipeline runs with optional filters."""
        session = self._get_session()
        try:
            query = session.query(PipelineRun)
            
            if dataset:
                query = query.join(Dataset).filter(Dataset.dataset_name == dataset)
            
            if status:
                query = query.filter(PipelineRun.status == RunStatusEnum(status))
            
            runs = query.order_by(desc(PipelineRun.run_timestamp)).limit(limit).all()
            
            return [{
                'run_id': r.run_id,
                'pipeline_id': r.pipeline_id,
                'timestamp': r.run_timestamp.isoformat() if r.run_timestamp else None,
                'status': r.status.value if r.status else None,
                'dataset': r.dataset.dataset_name if r.dataset else None,
                'total': r.total_combinations,
                'success': r.successful_combinations,
                'failed': r.failed_combinations,
            } for r in runs]
        finally:
            session.close()
    
    # ============================================================
    # Combination Queries
    # ============================================================
    
    def find_best_combinations(
        self,
        metric: str,
        dataset: Optional[str] = None,
        limit: int = 10,
        min_accuracy: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Find best performing combinations for a given metric."""
        session = self._get_session()
        try:
            if not hasattr(CombinationMetrics, metric):
                raise ValueError(f"Unknown metric: {metric}")
            
            metric_col = getattr(CombinationMetrics, metric)
            
            query = session.query(
                Combination, CombinationMetrics, PipelineRun, Dataset
            ).join(
                CombinationMetrics
            ).join(
                PipelineRun
            ).join(
                Dataset, PipelineRun.dataset_id == Dataset.dataset_id
            ).filter(
                Combination.status == CombinationStatusEnum.SUCCESS,
                metric_col.isnot(None)
            )
            
            if dataset:
                query = query.filter(Dataset.dataset_name == dataset)
            
            if min_accuracy and metric in ['acc_test_clean', 'pgd_accuracy']:
                query = query.filter(metric_col >= min_accuracy)
            
            results = query.order_by(desc(metric_col)).limit(limit).all()
            
            return [{
                'combination_id': c.combination_id,
                'combination_code': c.combination_code,
                'pipeline_id': r.pipeline_id,
                'run_timestamp': r.run_timestamp.isoformat() if r.run_timestamp else None,
                'dataset': d.dataset_name,
                'variant': d.variant,
                metric: float(getattr(m, metric)) if getattr(m, metric) else None,
                'acc_test_clean': float(m.acc_test_clean) if m.acc_test_clean else None,
                'pgd_accuracy': float(m.pgd_accuracy) if m.pgd_accuracy else None,
            } for c, m, r, d in results]
        finally:
            session.close()
    
    def get_combination_details(self, combination_id: int) -> Optional[Dict[str, Any]]:
        """Get full details for a combination including tools and metrics."""
        session = self._get_session()
        try:
            comb = session.query(Combination).filter(
                Combination.combination_id == combination_id
            ).first()
            
            if not comb:
                return None
            
            # Get tools by stage
            tools_query = session.query(CombinationTools, Tool).join(Tool).filter(
                CombinationTools.combination_id == combination_id
            ).order_by(CombinationTools.stage, CombinationTools.tool_order).all()
            
            tools_by_stage = {}
            for ct, tool in tools_query:
                stage = ct.stage.value
                if stage not in tools_by_stage:
                    tools_by_stage[stage] = []
                tools_by_stage[stage].append(tool.tool_name)
            
            # Get metrics
            metrics = session.query(CombinationMetrics).filter(
                CombinationMetrics.combination_id == combination_id
            ).first()
            
            # Get run info
            run = session.query(PipelineRun, Dataset).join(Dataset).filter(
                PipelineRun.run_id == comb.run_id
            ).first()
            
            return {
                'combination_id': comb.combination_id,
                'combination_code': comb.combination_code,
                'status': comb.status.value if comb.status else None,
                'duration_sec': float(comb.total_duration_sec) if comb.total_duration_sec else None,
                'pipeline_id': run[0].pipeline_id if run else None,
                'dataset': run[1].dataset_name if run else None,
                'tools': tools_by_stage,
                'metrics': metrics.to_dict() if metrics else {},
            }
        finally:
            session.close()
    
    def find_combinations_with_tools(
        self,
        tools: List[str],
        match_all: bool = True,
        dataset: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find combinations that use specific tools."""
        session = self._get_session()
        try:
            if match_all:
                # Must have all tools
                query = session.query(Combination).join(PipelineRun)
                
                for tool_name in tools:
                    subquery = session.query(CombinationTools.combination_id).join(Tool).filter(
                        Tool.tool_name == tool_name
                    )
                    query = query.filter(Combination.combination_id.in_(subquery))
                
                if dataset:
                    query = query.join(Dataset).filter(Dataset.dataset_name == dataset)
                
                combs = query.filter(Combination.status == CombinationStatusEnum.SUCCESS).all()
            else:
                # Any of the tools
                query = session.query(Combination).join(
                    CombinationTools
                ).join(Tool).join(PipelineRun).filter(
                    Tool.tool_name.in_(tools),
                    Combination.status == CombinationStatusEnum.SUCCESS
                ).distinct()
                
                if dataset:
                    query = query.join(Dataset).filter(Dataset.dataset_name == dataset)
                
                combs = query.all()
            
            return [self.get_combination_details(c.combination_id) for c in combs]
        finally:
            session.close()
    
    # ============================================================
    # Tool Analysis Queries
    # ============================================================
    
    def compare_tools(
        self,
        tools: List[str],
        metrics: Optional[List[str]] = None,
        dataset: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Compare performance of different tools."""
        if metrics is None:
            metrics = ['acc_test_clean', 'pgd_accuracy', 'carlini_l2_accuracy', 'ood_auc']
        
        session = self._get_session()
        try:
            results = {}
            
            for tool_name in tools:
                # Build aggregation query
                agg_funcs = [func.count(CombinationMetrics.metric_id).label('count')]
                for metric in metrics:
                    if hasattr(CombinationMetrics, metric):
                        col = getattr(CombinationMetrics, metric)
                        agg_funcs.append(func.avg(col).label(f'avg_{metric}'))
                        agg_funcs.append(func.max(col).label(f'max_{metric}'))
                        agg_funcs.append(func.min(col).label(f'min_{metric}'))
                
                query = session.query(*agg_funcs).join(
                    Combination,
                    CombinationMetrics.combination_id == Combination.combination_id
                ).join(
                    CombinationTools
                ).join(
                    Tool
                ).filter(
                    Tool.tool_name == tool_name,
                    Combination.status == CombinationStatusEnum.SUCCESS
                )
                
                if dataset:
                    query = query.join(
                        PipelineRun
                    ).join(
                        Dataset
                    ).filter(Dataset.dataset_name == dataset)
                
                result = query.first()
                
                if result:
                    tool_results = {'count': result.count}
                    for metric in metrics:
                        if hasattr(result, f'avg_{metric}'):
                            tool_results[metric] = {
                                'avg': float(getattr(result, f'avg_{metric}')) if getattr(result, f'avg_{metric}') else None,
                                'max': float(getattr(result, f'max_{metric}')) if getattr(result, f'max_{metric}') else None,
                                'min': float(getattr(result, f'min_{metric}')) if getattr(result, f'min_{metric}') else None,
                            }
                    results[tool_name] = tool_results
            
            return results
        finally:
            session.close()
    
    def get_tool_performance_by_stage(self, stage: str) -> List[Dict[str, Any]]:
        """Get performance statistics for all tools in a stage."""
        session = self._get_session()
        try:
            stage_enum = StageEnum(stage)
            
            results = session.query(
                Tool.tool_name,
                ToolCategory.category_name,
                func.count(ToolExecution.execution_id).label('total_executions'),
                func.avg(ToolExecution.duration_sec).label('avg_duration'),
                func.sum(func.if_(ToolExecution.cache_hit, 1, 0)).label('cache_hits'),
                func.sum(func.if_(ToolExecution.status == 'success', 1, 0)).label('successful'),
            ).outerjoin(
                ToolCategory
            ).outerjoin(
                ToolExecution
            ).filter(
                Tool.stage == stage_enum
            ).group_by(
                Tool.tool_id
            ).all()
            
            return [{
                'tool_name': r.tool_name,
                'category': r.category_name,
                'total_executions': r.total_executions or 0,
                'avg_duration_sec': float(r.avg_duration) if r.avg_duration else 0,
                'cache_hits': int(r.cache_hits) if r.cache_hits else 0,
                'successful': int(r.successful) if r.successful else 0,
                'success_rate': round(
                    (int(r.successful) / r.total_executions * 100)
                    if r.total_executions and r.successful else 0, 2
                ),
            } for r in results]
        finally:
            session.close()
    
    # ============================================================
    # Robustness Analysis Queries
    # ============================================================
    
    def find_robust_combinations(
        self,
        min_clean_acc: float = 0.7,
        min_pgd_acc: float = 0.3,
        dataset: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Find combinations with good clean and adversarial accuracy."""
        session = self._get_session()
        try:
            query = session.query(
                Combination, CombinationMetrics, PipelineRun, Dataset
            ).join(
                CombinationMetrics
            ).join(
                PipelineRun
            ).join(
                Dataset, PipelineRun.dataset_id == Dataset.dataset_id
            ).filter(
                Combination.status == CombinationStatusEnum.SUCCESS,
                CombinationMetrics.acc_test_clean >= min_clean_acc,
                CombinationMetrics.pgd_accuracy >= min_pgd_acc
            )
            
            if dataset:
                query = query.filter(Dataset.dataset_name == dataset)
            
            # Order by composite score
            results = query.order_by(
                desc(
                    CombinationMetrics.acc_test_clean * 0.5 + 
                    CombinationMetrics.pgd_accuracy * 0.5
                )
            ).limit(limit).all()
            
            return [{
                'combination_id': c.combination_id,
                'combination_code': c.combination_code,
                'pipeline_id': r.pipeline_id,
                'dataset': d.dataset_name,
                'clean_accuracy': float(m.acc_test_clean) if m.acc_test_clean else None,
                'pgd_accuracy': float(m.pgd_accuracy) if m.pgd_accuracy else None,
                'carlini_accuracy': float(m.carlini_l2_accuracy) if m.carlini_l2_accuracy else None,
                'robustness_score': round(
                    (float(m.acc_test_clean or 0) + float(m.pgd_accuracy or 0)) / 2, 4
                ),
            } for c, m, r, d in results]
        finally:
            session.close()
    
    def analyze_accuracy_vs_robustness(
        self,
        dataset: Optional[str] = None
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Analyze trade-off between clean accuracy and adversarial robustness."""
        session = self._get_session()
        try:
            query = session.query(
                CombinationMetrics.acc_test_clean,
                CombinationMetrics.pgd_accuracy
            ).join(
                Combination
            ).filter(
                Combination.status == CombinationStatusEnum.SUCCESS,
                CombinationMetrics.acc_test_clean.isnot(None),
                CombinationMetrics.pgd_accuracy.isnot(None)
            )
            
            if dataset:
                query = query.join(
                    PipelineRun
                ).join(
                    Dataset
                ).filter(Dataset.dataset_name == dataset)
            
            results = query.all()
            
            return {
                'data_points': [
                    (float(r.acc_test_clean), float(r.pgd_accuracy))
                    for r in results
                ]
            }
        finally:
            session.close()
    
    # ============================================================
    # Cache Analysis Queries
    # ============================================================
    
    def get_cache_efficiency_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get comprehensive cache efficiency report."""
        session = self._get_session()
        try:
            query = session.query(
                func.count(ToolExecution.execution_id).label('total'),
                func.sum(func.if_(ToolExecution.cache_hit, 1, 0)).label('hits'),
                func.sum(ToolExecution.duration_sec).label('total_duration'),
                func.sum(func.if_(ToolExecution.cache_hit, ToolExecution.duration_sec, 0)).label('cached_duration'),
            ).join(Combination).join(PipelineRun)
            
            if start_date:
                query = query.filter(PipelineRun.run_timestamp >= start_date)
            if end_date:
                query = query.filter(PipelineRun.run_timestamp <= end_date)
            
            result = query.first()
            
            # Get per-tool cache stats
            tool_stats = session.query(
                Tool.tool_name,
                func.count(ToolExecution.execution_id).label('total'),
                func.sum(func.if_(ToolExecution.cache_hit, 1, 0)).label('hits'),
            ).join(ToolExecution).group_by(Tool.tool_id).all()
            
            return {
                'overall': {
                    'total_executions': result.total if result else 0,
                    'cache_hits': int(result.hits) if result and result.hits else 0,
                    'cache_hit_rate': round(
                        (int(result.hits) / result.total * 100)
                        if result and result.total and result.hits else 0, 2
                    ),
                    'total_duration_sec': float(result.total_duration) if result and result.total_duration else 0,
                    'time_saved_sec': float(result.cached_duration) if result and result.cached_duration else 0,
                },
                'by_tool': [{
                    'tool': t.tool_name,
                    'total': t.total,
                    'hits': int(t.hits) if t.hits else 0,
                    'hit_rate': round((int(t.hits) / t.total * 100) if t.total and t.hits else 0, 2),
                } for t in tool_stats]
            }
        finally:
            session.close()
    
    # ============================================================
    # Export Functions
    # ============================================================
    
    def export_run_to_csv(self, run_id: int, output_path: str) -> bool:
        """Export a pipeline run's results to CSV."""
        import csv
        
        session = self._get_session()
        try:
            # Get all combinations with metrics
            results = session.query(
                Combination, CombinationMetrics
            ).outerjoin(
                CombinationMetrics
            ).filter(
                Combination.run_id == run_id
            ).order_by(Combination.combination_index).all()
            
            if not results:
                return False
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow([
                    'combination_code', 'status', 'duration_sec',
                    'acc_train_clean', 'acc_test_clean', 'pgd_accuracy',
                    'carlini_l2_accuracy', 'ood_auc', 'fingerprinting_score',
                    'attack_success_rate', 'privacy_epsilon', 'mia_auc',
                    'watermark_accuracy'
                ])
                
                # Data rows
                for comb, metrics in results:
                    writer.writerow([
                        comb.combination_code,
                        comb.status.value if comb.status else '',
                        float(comb.total_duration_sec) if comb.total_duration_sec else '',
                        float(metrics.acc_train_clean) if metrics and metrics.acc_train_clean else '',
                        float(metrics.acc_test_clean) if metrics and metrics.acc_test_clean else '',
                        float(metrics.pgd_accuracy) if metrics and metrics.pgd_accuracy else '',
                        float(metrics.carlini_l2_accuracy) if metrics and metrics.carlini_l2_accuracy else '',
                        float(metrics.ood_auc) if metrics and metrics.ood_auc else '',
                        float(metrics.fingerprinting_score) if metrics and metrics.fingerprinting_score else '',
                        float(metrics.attack_success_rate) if metrics and metrics.attack_success_rate else '',
                        float(metrics.privacy_epsilon) if metrics and metrics.privacy_epsilon else '',
                        float(metrics.mia_auc) if metrics and metrics.mia_auc else '',
                        float(metrics.watermark_accuracy) if metrics and metrics.watermark_accuracy else '',
                    ])
            
            return True
        finally:
            session.close()
    
    def execute_raw_query(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Execute a raw SQL query and return results as dictionaries."""
        session = self._get_session()
        try:
            result = session.execute(text(query), params or {})
            columns = result.keys()
            return [dict(zip(columns, row)) for row in result.fetchall()]
        finally:
            session.close()
