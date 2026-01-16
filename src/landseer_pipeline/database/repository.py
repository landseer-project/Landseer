"""
Repository Pattern Implementation for Landseer Database

Provides high-level data access methods for common operations,
abstracting away the raw SQL queries.
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from decimal import Decimal

from sqlalchemy import and_, or_, func, desc, asc
from sqlalchemy.orm import Session, joinedload

from .models import (
    Base, Dataset, Model, Tool, ToolCategory,
    PipelineRun, PipelineAttacks, Combination, CombinationTools,
    CombinationMetrics, ToolExecution, ArtifactNode, ArtifactFile,
    OutputFileProvenance, StageEnum, RunStatusEnum, CombinationStatusEnum
)

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository with common CRUD operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def commit(self):
        """Commit current transaction."""
        self.session.commit()
    
    def rollback(self):
        """Rollback current transaction."""
        self.session.rollback()
    
    def flush(self):
        """Flush pending changes."""
        self.session.flush()


class DatasetRepository(BaseRepository):
    """Repository for Dataset operations."""
    
    def get_or_create(
        self, 
        name: str, 
        variant: str = "clean", 
        version: Optional[str] = None,
        **kwargs
    ) -> Dataset:
        """Get existing dataset or create new one."""
        dataset = self.session.query(Dataset).filter(
            and_(
                Dataset.dataset_name == name,
                Dataset.variant == variant,
                Dataset.version == version
            )
        ).first()
        
        if dataset is None:
            dataset = Dataset(
                dataset_name=name,
                variant=variant,
                version=version,
                **kwargs
            )
            self.session.add(dataset)
            self.session.flush()
        
        return dataset
    
    def list_all(self) -> List[Dataset]:
        """List all datasets."""
        return self.session.query(Dataset).all()


class ModelRepository(BaseRepository):
    """Repository for Model operations."""
    
    def get_or_create(
        self, 
        script_path: str, 
        content_hash: str,
        **kwargs
    ) -> Model:
        """Get existing model or create new one."""
        model = self.session.query(Model).filter(
            Model.content_hash == content_hash
        ).first()
        
        if model is None:
            model = Model(
                script_path=script_path,
                content_hash=content_hash,
                **kwargs
            )
            self.session.add(model)
            self.session.flush()
        
        return model


class ToolRepository(BaseRepository):
    """Repository for Tool operations."""
    
    def get_or_create(
        self, 
        tool_name: str, 
        stage: StageEnum,
        category_name: Optional[str] = None,
        **kwargs
    ) -> Tool:
        """Get existing tool or create new one."""
        tool = self.session.query(Tool).filter(
            Tool.tool_name == tool_name
        ).first()
        
        if tool is None:
            category_id = None
            if category_name:
                category = self.session.query(ToolCategory).filter(
                    ToolCategory.category_name == category_name
                ).first()
                if category:
                    category_id = category.category_id
            
            tool = Tool(
                tool_name=tool_name,
                stage=stage,
                category_id=category_id,
                **kwargs
            )
            self.session.add(tool)
            self.session.flush()
        
        return tool
    
    def get_by_name(self, tool_name: str) -> Optional[Tool]:
        """Get tool by name."""
        return self.session.query(Tool).filter(
            Tool.tool_name == tool_name
        ).first()
    
    def list_by_stage(self, stage: StageEnum) -> List[Tool]:
        """List all tools for a given stage."""
        return self.session.query(Tool).filter(
            Tool.stage == stage
        ).all()
    
    def list_all(self) -> List[Tool]:
        """List all tools."""
        return self.session.query(Tool).all()
    
    def get_tool_performance_stats(self, tool_name: str) -> Dict[str, Any]:
        """Get performance statistics for a tool."""
        result = self.session.query(
            func.count(ToolExecution.execution_id).label('total_executions'),
            func.sum(func.if_(ToolExecution.status == 'success', 1, 0)).label('successful'),
            func.sum(func.if_(ToolExecution.cache_hit, 1, 0)).label('cache_hits'),
            func.avg(ToolExecution.duration_sec).label('avg_duration'),
            func.min(ToolExecution.duration_sec).label('min_duration'),
            func.max(ToolExecution.duration_sec).label('max_duration'),
        ).join(Tool).filter(
            Tool.tool_name == tool_name
        ).first()
        
        if result:
            return {
                'total_executions': result.total_executions or 0,
                'successful': result.successful or 0,
                'cache_hits': result.cache_hits or 0,
                'avg_duration': float(result.avg_duration) if result.avg_duration else 0,
                'min_duration': float(result.min_duration) if result.min_duration else 0,
                'max_duration': float(result.max_duration) if result.max_duration else 0,
            }
        return {}


class PipelineRunRepository(BaseRepository):
    """Repository for PipelineRun operations."""
    
    def create(
        self,
        pipeline_id: str,
        run_timestamp: datetime,
        dataset: Dataset,
        model: Optional[Model] = None,
        config_file_path: Optional[str] = None,
        attack_config_path: Optional[str] = None,
        **kwargs
    ) -> PipelineRun:
        """Create a new pipeline run."""
        run = PipelineRun(
            pipeline_id=pipeline_id,
            run_timestamp=run_timestamp,
            dataset_id=dataset.dataset_id,
            model_id=model.model_id if model else None,
            config_file_path=config_file_path,
            attack_config_path=attack_config_path,
            **kwargs
        )
        self.session.add(run)
        self.session.flush()
        return run
    
    def get_by_id(self, run_id: int) -> Optional[PipelineRun]:
        """Get pipeline run by ID."""
        return self.session.query(PipelineRun).options(
            joinedload(PipelineRun.dataset),
            joinedload(PipelineRun.model),
            joinedload(PipelineRun.attacks)
        ).filter(
            PipelineRun.run_id == run_id
        ).first()
    
    def get_by_pipeline_id(self, pipeline_id: str) -> List[PipelineRun]:
        """Get all runs for a pipeline ID."""
        return self.session.query(PipelineRun).filter(
            PipelineRun.pipeline_id == pipeline_id
        ).order_by(desc(PipelineRun.run_timestamp)).all()
    
    def get_latest(self, pipeline_id: Optional[str] = None) -> Optional[PipelineRun]:
        """Get the latest pipeline run."""
        query = self.session.query(PipelineRun)
        if pipeline_id:
            query = query.filter(PipelineRun.pipeline_id == pipeline_id)
        return query.order_by(desc(PipelineRun.run_timestamp)).first()
    
    def update_status(
        self, 
        run_id: int, 
        status: RunStatusEnum,
        error_message: Optional[str] = None
    ) -> None:
        """Update run status."""
        run = self.get_by_id(run_id)
        if run:
            run.status = status
            if error_message:
                run.error_message = error_message
            if status in [RunStatusEnum.COMPLETED, RunStatusEnum.FAILED]:
                run.completed_at = datetime.utcnow()
            self.session.flush()
    
    def update_combination_counts(self, run_id: int) -> None:
        """Update combination success/failure counts."""
        run = self.get_by_id(run_id)
        if run:
            total = self.session.query(func.count(Combination.combination_id)).filter(
                Combination.run_id == run_id
            ).scalar()
            
            successful = self.session.query(func.count(Combination.combination_id)).filter(
                and_(
                    Combination.run_id == run_id,
                    Combination.status == CombinationStatusEnum.SUCCESS
                )
            ).scalar()
            
            failed = self.session.query(func.count(Combination.combination_id)).filter(
                and_(
                    Combination.run_id == run_id,
                    Combination.status == CombinationStatusEnum.FAILURE
                )
            ).scalar()
            
            run.total_combinations = total
            run.successful_combinations = successful
            run.failed_combinations = failed
            self.session.flush()
    
    def list_recent(self, limit: int = 10) -> List[PipelineRun]:
        """List recent pipeline runs."""
        return self.session.query(PipelineRun).options(
            joinedload(PipelineRun.dataset)
        ).order_by(
            desc(PipelineRun.run_timestamp)
        ).limit(limit).all()
    
    def search(
        self,
        dataset_name: Optional[str] = None,
        status: Optional[RunStatusEnum] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[PipelineRun]:
        """Search pipeline runs with filters."""
        query = self.session.query(PipelineRun).options(
            joinedload(PipelineRun.dataset)
        )
        
        if dataset_name:
            query = query.join(Dataset).filter(Dataset.dataset_name == dataset_name)
        if status:
            query = query.filter(PipelineRun.status == status)
        if start_date:
            query = query.filter(PipelineRun.run_timestamp >= start_date)
        if end_date:
            query = query.filter(PipelineRun.run_timestamp <= end_date)
        
        return query.order_by(desc(PipelineRun.run_timestamp)).all()


class CombinationRepository(BaseRepository):
    """Repository for Combination operations."""
    
    def create(
        self,
        run: PipelineRun,
        combination_code: str,
        combination_index: int,
        tools_by_stage: Dict[str, List[str]],
        **kwargs
    ) -> Combination:
        """Create a new combination with its tools."""
        combination = Combination(
            run_id=run.run_id,
            combination_code=combination_code,
            combination_index=combination_index,
            **kwargs
        )
        self.session.add(combination)
        self.session.flush()
        
        # Add tools for each stage
        tool_repo = ToolRepository(self.session)
        for stage_name, tool_names in tools_by_stage.items():
            stage = StageEnum(stage_name)
            for order, tool_name in enumerate(tool_names):
                tool = tool_repo.get_by_name(tool_name)
                if tool:
                    ct = CombinationTools(
                        combination_id=combination.combination_id,
                        tool_id=tool.tool_id,
                        stage=stage,
                        tool_order=order
                    )
                    self.session.add(ct)
        
        self.session.flush()
        return combination
    
    def get_by_id(self, combination_id: int) -> Optional[Combination]:
        """Get combination by ID with all related data."""
        return self.session.query(Combination).options(
            joinedload(Combination.tools).joinedload(CombinationTools.tool),
            joinedload(Combination.metrics),
            joinedload(Combination.pipeline_run)
        ).filter(
            Combination.combination_id == combination_id
        ).first()
    
    def get_by_code(self, run_id: int, combination_code: str) -> Optional[Combination]:
        """Get combination by run ID and code."""
        return self.session.query(Combination).filter(
            and_(
                Combination.run_id == run_id,
                Combination.combination_code == combination_code
            )
        ).first()
    
    def list_for_run(self, run_id: int) -> List[Combination]:
        """List all combinations for a pipeline run."""
        return self.session.query(Combination).options(
            joinedload(Combination.tools).joinedload(CombinationTools.tool),
            joinedload(Combination.metrics)
        ).filter(
            Combination.run_id == run_id
        ).order_by(Combination.combination_index).all()
    
    def update_status(
        self,
        combination_id: int,
        status: CombinationStatusEnum,
        error_message: Optional[str] = None,
        duration_sec: Optional[float] = None
    ) -> None:
        """Update combination status."""
        comb = self.session.query(Combination).filter(
            Combination.combination_id == combination_id
        ).first()
        
        if comb:
            comb.status = status
            if error_message:
                comb.error_message = error_message
            if duration_sec:
                comb.total_duration_sec = Decimal(str(duration_sec))
            if status in [CombinationStatusEnum.SUCCESS, CombinationStatusEnum.FAILURE]:
                comb.completed_at = datetime.utcnow()
            self.session.flush()
    
    def find_with_tool(
        self,
        tool_name: str,
        stage: Optional[StageEnum] = None
    ) -> List[Combination]:
        """Find all combinations using a specific tool."""
        query = self.session.query(Combination).join(
            CombinationTools
        ).join(Tool).filter(
            Tool.tool_name == tool_name
        )
        
        if stage:
            query = query.filter(CombinationTools.stage == stage)
        
        return query.all()


class MetricsRepository(BaseRepository):
    """Repository for CombinationMetrics operations."""
    
    def create_or_update(
        self,
        combination_id: int,
        **metrics
    ) -> CombinationMetrics:
        """Create or update metrics for a combination."""
        existing = self.session.query(CombinationMetrics).filter(
            CombinationMetrics.combination_id == combination_id
        ).first()
        
        if existing:
            for key, value in metrics.items():
                if hasattr(existing, key) and value is not None:
                    setattr(existing, key, Decimal(str(value)) if value != -1 else None)
            self.session.flush()
            return existing
        
        # Convert -1 values to None
        clean_metrics = {}
        for key, value in metrics.items():
            if value is not None and value != -1:
                clean_metrics[key] = Decimal(str(value))
            else:
                clean_metrics[key] = None
        
        metric = CombinationMetrics(
            combination_id=combination_id,
            **clean_metrics
        )
        self.session.add(metric)
        self.session.flush()
        return metric
    
    def get_for_combination(self, combination_id: int) -> Optional[CombinationMetrics]:
        """Get metrics for a specific combination."""
        return self.session.query(CombinationMetrics).filter(
            CombinationMetrics.combination_id == combination_id
        ).first()
    
    def find_best_by_metric(
        self,
        metric_name: str,
        dataset_name: Optional[str] = None,
        limit: int = 10,
        ascending: bool = False
    ) -> List[Tuple[Combination, CombinationMetrics]]:
        """Find combinations with best values for a metric."""
        if not hasattr(CombinationMetrics, metric_name):
            raise ValueError(f"Unknown metric: {metric_name}")
        
        metric_col = getattr(CombinationMetrics, metric_name)
        
        query = self.session.query(Combination, CombinationMetrics).join(
            CombinationMetrics
        ).join(
            PipelineRun
        ).filter(
            metric_col.isnot(None),
            Combination.status == CombinationStatusEnum.SUCCESS
        )
        
        if dataset_name:
            query = query.join(Dataset).filter(Dataset.dataset_name == dataset_name)
        
        order = asc(metric_col) if ascending else desc(metric_col)
        return query.order_by(order).limit(limit).all()
    
    def compare_tools(
        self,
        tool_names: List[str],
        dataset_name: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compare average metrics across tools."""
        results = {}
        
        for tool_name in tool_names:
            query = self.session.query(
                func.avg(CombinationMetrics.acc_test_clean).label('avg_acc'),
                func.avg(CombinationMetrics.pgd_accuracy).label('avg_pgd'),
                func.avg(CombinationMetrics.carlini_l2_accuracy).label('avg_carlini'),
                func.avg(CombinationMetrics.ood_auc).label('avg_ood'),
            ).join(
                Combination
            ).join(
                CombinationTools
            ).join(
                Tool
            ).filter(
                Tool.tool_name == tool_name,
                Combination.status == CombinationStatusEnum.SUCCESS
            )
            
            if dataset_name:
                query = query.join(
                    PipelineRun, Combination.run_id == PipelineRun.run_id
                ).join(Dataset).filter(Dataset.dataset_name == dataset_name)
            
            result = query.first()
            if result:
                results[tool_name] = {
                    'avg_accuracy': float(result.avg_acc) if result.avg_acc else None,
                    'avg_pgd_accuracy': float(result.avg_pgd) if result.avg_pgd else None,
                    'avg_carlini_accuracy': float(result.avg_carlini) if result.avg_carlini else None,
                    'avg_ood_auc': float(result.avg_ood) if result.avg_ood else None,
                }
        
        return results
    
    def get_aggregate_stats(self, run_id: int) -> Dict[str, Any]:
        """Get aggregate statistics for a pipeline run."""
        result = self.session.query(
            func.avg(CombinationMetrics.acc_test_clean).label('avg_acc'),
            func.avg(CombinationMetrics.pgd_accuracy).label('avg_pgd'),
            func.avg(CombinationMetrics.carlini_l2_accuracy).label('avg_carlini'),
            func.avg(CombinationMetrics.ood_auc).label('avg_ood'),
            func.max(CombinationMetrics.acc_test_clean).label('max_acc'),
            func.max(CombinationMetrics.pgd_accuracy).label('max_pgd'),
            func.min(CombinationMetrics.acc_test_clean).label('min_acc'),
        ).join(Combination).filter(
            Combination.run_id == run_id,
            Combination.status == CombinationStatusEnum.SUCCESS
        ).first()
        
        if result:
            return {
                'avg_accuracy': float(result.avg_acc) if result.avg_acc else None,
                'avg_pgd_accuracy': float(result.avg_pgd) if result.avg_pgd else None,
                'avg_carlini_accuracy': float(result.avg_carlini) if result.avg_carlini else None,
                'avg_ood_auc': float(result.avg_ood) if result.avg_ood else None,
                'max_accuracy': float(result.max_acc) if result.max_acc else None,
                'max_pgd_accuracy': float(result.max_pgd) if result.max_pgd else None,
                'min_accuracy': float(result.min_acc) if result.min_acc else None,
            }
        return {}


class ToolExecutionRepository(BaseRepository):
    """Repository for ToolExecution operations."""
    
    def create(
        self,
        combination_id: int,
        tool_id: int,
        stage: StageEnum,
        **kwargs
    ) -> ToolExecution:
        """Create a new tool execution record."""
        execution = ToolExecution(
            combination_id=combination_id,
            tool_id=tool_id,
            stage=stage,
            **kwargs
        )
        self.session.add(execution)
        self.session.flush()
        return execution
    
    def get_for_combination(self, combination_id: int) -> List[ToolExecution]:
        """Get all tool executions for a combination."""
        return self.session.query(ToolExecution).options(
            joinedload(ToolExecution.tool)
        ).filter(
            ToolExecution.combination_id == combination_id
        ).order_by(ToolExecution.stage).all()
    
    def get_cache_stats(self, run_id: int) -> Dict[str, Any]:
        """Get cache statistics for a pipeline run."""
        result = self.session.query(
            func.count(ToolExecution.execution_id).label('total'),
            func.sum(func.if_(ToolExecution.cache_hit, 1, 0)).label('cache_hits'),
            func.sum(func.if_(ToolExecution.cache_hit, ToolExecution.duration_sec, 0)).label('saved_time'),
            func.sum(func.if_(~ToolExecution.cache_hit, ToolExecution.duration_sec, 0)).label('compute_time'),
        ).join(Combination).filter(
            Combination.run_id == run_id
        ).first()
        
        if result:
            total = result.total or 0
            cache_hits = result.cache_hits or 0
            return {
                'total_executions': total,
                'cache_hits': cache_hits,
                'cache_misses': total - cache_hits,
                'cache_hit_rate': (cache_hits / total * 100) if total > 0 else 0,
                'saved_time_sec': float(result.saved_time) if result.saved_time else 0,
                'compute_time_sec': float(result.compute_time) if result.compute_time else 0,
            }
        return {}


class ArtifactRepository(BaseRepository):
    """Repository for ArtifactNode operations."""
    
    def get_or_create(self, node_hash: str, **kwargs) -> ArtifactNode:
        """Get existing artifact node or create new one."""
        node = self.session.query(ArtifactNode).filter(
            ArtifactNode.node_hash == node_hash
        ).first()
        
        if node is None:
            node = ArtifactNode(node_hash=node_hash, **kwargs)
            self.session.add(node)
            self.session.flush()
        else:
            # Update access count
            node.access_count += 1
            self.session.flush()
        
        return node
    
    def add_file(
        self,
        node: ArtifactNode,
        relative_path: str,
        file_size_bytes: int,
        file_hash: Optional[str] = None,
        mime_type: Optional[str] = None
    ) -> ArtifactFile:
        """Add a file to an artifact node."""
        file = ArtifactFile(
            node_id=node.node_id,
            relative_path=relative_path,
            file_size_bytes=file_size_bytes,
            file_hash=file_hash,
            mime_type=mime_type
        )
        self.session.add(file)
        self.session.flush()
        return file
    
    def get_by_hash(self, node_hash: str) -> Optional[ArtifactNode]:
        """Get artifact node by hash."""
        return self.session.query(ArtifactNode).options(
            joinedload(ArtifactNode.files)
        ).filter(
            ArtifactNode.node_hash == node_hash
        ).first()
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get overall storage statistics."""
        result = self.session.query(
            func.count(ArtifactNode.node_id).label('total_nodes'),
            func.sum(ArtifactNode.total_size_bytes).label('total_size'),
            func.sum(ArtifactNode.access_count).label('total_accesses'),
        ).first()
        
        if result:
            return {
                'total_nodes': result.total_nodes or 0,
                'total_size_bytes': result.total_size or 0,
                'total_size_gb': (result.total_size or 0) / (1024**3),
                'total_accesses': result.total_accesses or 0,
            }
        return {}
