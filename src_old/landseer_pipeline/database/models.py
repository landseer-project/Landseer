"""
SQLAlchemy ORM Models for Landseer Pipeline Database

These models provide an object-oriented interface to the database tables,
making it easier to work with pipeline data in Python code.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from decimal import Decimal
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, Integer, String, Text, Boolean, DateTime, 
    ForeignKey, Enum, DECIMAL, BIGINT, JSON, Index,
    UniqueConstraint, create_engine
)
from sqlalchemy.orm import (
    declarative_base, relationship, sessionmaker, Session
)
from sqlalchemy.sql import func


Base = declarative_base()


# ============================================================
# Enums
# ============================================================

class StageEnum(str, PyEnum):
    """Pipeline stages."""
    PRE_TRAINING = "pre_training"
    DURING_TRAINING = "during_training"
    POST_TRAINING = "post_training"
    DEPLOYMENT = "deployment"


class RunStatusEnum(str, PyEnum):
    """Pipeline run status."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class CombinationStatusEnum(str, PyEnum):
    """Combination execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


class ExecutionStatusEnum(str, PyEnum):
    """Tool execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    CACHED = "cached"


class FrameworkEnum(str, PyEnum):
    """ML framework types."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    OTHER = "other"


# ============================================================
# Reference/Lookup Tables
# ============================================================

class ToolCategory(Base):
    """Tool categories for classification."""
    __tablename__ = "tool_categories"
    
    category_id = Column(Integer, primary_key=True, autoincrement=True)
    category_name = Column(String(50), unique=True, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    tools = relationship("Tool", back_populates="category")
    
    def __repr__(self):
        return f"<ToolCategory(id={self.category_id}, name='{self.category_name}')>"


class Tool(Base):
    """Registry of available tools."""
    __tablename__ = "tools"
    
    tool_id = Column(Integer, primary_key=True, autoincrement=True)
    tool_name = Column(String(100), unique=True, nullable=False)
    category_id = Column(Integer, ForeignKey("tool_categories.category_id"))
    stage = Column(Enum(StageEnum), nullable=False)
    container_image = Column(String(255))
    container_command = Column(String(500))
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relationships
    category = relationship("ToolCategory", back_populates="tools")
    combination_tools = relationship("CombinationTools", back_populates="tool")
    executions = relationship("ToolExecution", back_populates="tool")
    
    __table_args__ = (
        Index("idx_tool_stage", "stage"),
        Index("idx_tool_category", "category_id"),
    )
    
    def __repr__(self):
        return f"<Tool(id={self.tool_id}, name='{self.tool_name}', stage='{self.stage}')>"


class Dataset(Base):
    """Dataset registry."""
    __tablename__ = "datasets"
    
    dataset_id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_name = Column(String(100), nullable=False)
    variant = Column(String(50), nullable=False, default="clean")
    version = Column(String(50))
    num_classes = Column(Integer)
    num_train_samples = Column(Integer)
    num_test_samples = Column(Integer)
    input_shape = Column(String(50))
    description = Column(Text)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    pipeline_runs = relationship("PipelineRun", back_populates="dataset")
    
    __table_args__ = (
        UniqueConstraint("dataset_name", "variant", "version", name="uk_dataset"),
        Index("idx_dataset_name", "dataset_name"),
        Index("idx_dataset_variant", "variant"),
    )
    
    def __repr__(self):
        return f"<Dataset(id={self.dataset_id}, name='{self.dataset_name}', variant='{self.variant}')>"


class Model(Base):
    """Model configurations registry."""
    __tablename__ = "models"
    
    model_id = Column(Integer, primary_key=True, autoincrement=True)
    script_path = Column(String(500), nullable=False)
    content_hash = Column(String(64), unique=True, nullable=False)
    framework = Column(Enum(FrameworkEnum), default=FrameworkEnum.PYTORCH)
    architecture_name = Column(String(100))
    num_parameters = Column(BIGINT)
    description = Column(Text)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    pipeline_runs = relationship("PipelineRun", back_populates="model")
    
    __table_args__ = (
        Index("idx_model_hash", "content_hash"),
        Index("idx_model_framework", "framework"),
    )
    
    def __repr__(self):
        return f"<Model(id={self.model_id}, arch='{self.architecture_name}', hash='{self.content_hash[:8]}...')>"


# ============================================================
# Pipeline Run Tables
# ============================================================

class PipelineRun(Base):
    """Main pipeline runs table."""
    __tablename__ = "pipeline_runs"
    
    run_id = Column(Integer, primary_key=True, autoincrement=True)
    pipeline_id = Column(String(16), nullable=False)
    run_timestamp = Column(DateTime, nullable=False)
    config_file_path = Column(String(500))
    attack_config_path = Column(String(500))
    config_hash = Column(String(64))
    
    # Foreign keys
    dataset_id = Column(Integer, ForeignKey("datasets.dataset_id"))
    model_id = Column(Integer, ForeignKey("models.model_id"))
    
    # Run metadata
    total_combinations = Column(Integer, default=0)
    successful_combinations = Column(Integer, default=0)
    failed_combinations = Column(Integer, default=0)
    total_duration_sec = Column(DECIMAL(12, 3))
    
    # Status
    status = Column(Enum(RunStatusEnum), default=RunStatusEnum.RUNNING)
    error_message = Column(Text)
    
    # Version info
    git_commit = Column(String(40))
    landseer_version = Column(String(20))
    
    created_at = Column(DateTime, default=func.current_timestamp())
    completed_at = Column(DateTime)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="pipeline_runs")
    model = relationship("Model", back_populates="pipeline_runs")
    attacks = relationship("PipelineAttacks", back_populates="pipeline_run", uselist=False)
    combinations = relationship("Combination", back_populates="pipeline_run", cascade="all, delete-orphan")
    
    __table_args__ = (
        UniqueConstraint("pipeline_id", "run_timestamp", name="uk_pipeline_run"),
        Index("idx_pipeline_id", "pipeline_id"),
        Index("idx_run_timestamp", "run_timestamp"),
        Index("idx_run_status", "status"),
        Index("idx_dataset", "dataset_id"),
    )
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_combinations == 0:
            return 0.0
        return (self.successful_combinations / self.total_combinations) * 100
    
    def __repr__(self):
        return f"<PipelineRun(id={self.run_id}, pipeline='{self.pipeline_id}', status='{self.status}')>"


class PipelineAttacks(Base):
    """Attack configuration for pipeline runs."""
    __tablename__ = "pipeline_attacks"
    
    attack_id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("pipeline_runs.run_id", ondelete="CASCADE"), nullable=False)
    
    # Attack types enabled
    backdoor_enabled = Column(Boolean, default=False)
    adversarial_enabled = Column(Boolean, default=False)
    outlier_enabled = Column(Boolean, default=False)
    carlini_enabled = Column(Boolean, default=False)
    watermarking_enabled = Column(Boolean, default=False)
    fingerprinting_enabled = Column(Boolean, default=False)
    inference_enabled = Column(Boolean, default=False)
    other_enabled = Column(Boolean, default=False)
    
    # Flexible attack parameters
    attack_params = Column(JSON)
    
    # Relationships
    pipeline_run = relationship("PipelineRun", back_populates="attacks")
    
    __table_args__ = (
        Index("idx_attack_run", "run_id"),
    )
    
    def __repr__(self):
        return f"<PipelineAttacks(run_id={self.run_id})>"


# ============================================================
# Combination Tables
# ============================================================

class Combination(Base):
    """Pipeline combinations (tool combinations)."""
    __tablename__ = "combinations"
    
    combination_id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("pipeline_runs.run_id", ondelete="CASCADE"), nullable=False)
    combination_code = Column(String(20), nullable=False)
    combination_index = Column(Integer, nullable=False)
    
    # Execution metadata
    status = Column(Enum(CombinationStatusEnum), default=CombinationStatusEnum.PENDING)
    error_message = Column(Text)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    total_duration_sec = Column(DECIMAL(10, 3))
    
    # Output path
    output_directory = Column(String(500))
    
    # Relationships
    pipeline_run = relationship("PipelineRun", back_populates="combinations")
    tools = relationship("CombinationTools", back_populates="combination", cascade="all, delete-orphan")
    metrics = relationship("CombinationMetrics", back_populates="combination", uselist=False, cascade="all, delete-orphan")
    executions = relationship("ToolExecution", back_populates="combination", cascade="all, delete-orphan")
    output_files = relationship("OutputFileProvenance", back_populates="combination", cascade="all, delete-orphan")
    
    __table_args__ = (
        UniqueConstraint("run_id", "combination_code", name="uk_combination"),
        Index("idx_comb_run", "run_id"),
        Index("idx_comb_status", "status"),
        Index("idx_comb_code", "combination_code"),
    )
    
    def get_tools_by_stage(self) -> Dict[str, List[str]]:
        """Get tools organized by stage."""
        result = {stage.value: [] for stage in StageEnum}
        for ct in sorted(self.tools, key=lambda x: (x.stage, x.tool_order)):
            result[ct.stage].append(ct.tool.tool_name)
        return result
    
    def __repr__(self):
        return f"<Combination(id={self.combination_id}, code='{self.combination_code}', status='{self.status}')>"


class CombinationTools(Base):
    """Tools used in each combination (many-to-many)."""
    __tablename__ = "combination_tools"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    combination_id = Column(Integer, ForeignKey("combinations.combination_id", ondelete="CASCADE"), nullable=False)
    tool_id = Column(Integer, ForeignKey("tools.tool_id"), nullable=False)
    stage = Column(Enum(StageEnum), nullable=False)
    tool_order = Column(Integer, nullable=False, default=0)
    
    # Relationships
    combination = relationship("Combination", back_populates="tools")
    tool = relationship("Tool", back_populates="combination_tools")
    
    __table_args__ = (
        UniqueConstraint("combination_id", "tool_id", "stage", "tool_order", name="uk_comb_tool_stage"),
        Index("idx_comb_tool", "combination_id"),
        Index("idx_tool", "tool_id"),
        Index("idx_stage", "stage"),
    )
    
    def __repr__(self):
        return f"<CombinationTools(comb={self.combination_id}, tool={self.tool_id}, stage='{self.stage}')>"


class CombinationMetrics(Base):
    """Evaluation metrics for each combination."""
    __tablename__ = "combination_metrics"
    
    metric_id = Column(Integer, primary_key=True, autoincrement=True)
    combination_id = Column(Integer, ForeignKey("combinations.combination_id", ondelete="CASCADE"), unique=True, nullable=False)
    
    # Accuracy metrics
    acc_train_clean = Column(DECIMAL(6, 4))
    acc_test_clean = Column(DECIMAL(6, 4))
    
    # Adversarial robustness metrics
    pgd_accuracy = Column(DECIMAL(6, 4))
    carlini_l2_accuracy = Column(DECIMAL(6, 4))
    fgsm_accuracy = Column(DECIMAL(6, 4))
    autoattack_accuracy = Column(DECIMAL(6, 4))
    
    # Out-of-distribution detection
    ood_auc = Column(DECIMAL(6, 4))
    ood_fpr_at_95_tpr = Column(DECIMAL(6, 4))
    
    # Model fingerprinting
    fingerprinting_score = Column(DECIMAL(6, 4))
    
    # Backdoor attack metrics
    attack_success_rate = Column(DECIMAL(6, 4))
    clean_accuracy_after_attack = Column(DECIMAL(6, 4))
    
    # Privacy metrics
    privacy_epsilon = Column(DECIMAL(10, 4))
    dp_accuracy = Column(DECIMAL(6, 4))
    mia_auc = Column(DECIMAL(6, 4))
    eps_estimate = Column(DECIMAL(10, 4))
    
    # Watermarking metrics
    watermark_accuracy = Column(DECIMAL(6, 4))
    watermark_detection_rate = Column(DECIMAL(6, 4))
    
    # Model efficiency metrics
    model_size_mb = Column(DECIMAL(10, 3))
    inference_time_ms = Column(DECIMAL(10, 3))
    flops = Column(BIGINT)
    
    # Timestamps
    evaluated_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    combination = relationship("Combination", back_populates="metrics")
    
    __table_args__ = (
        Index("idx_acc_test", "acc_test_clean"),
        Index("idx_pgd", "pgd_accuracy"),
        Index("idx_carlini", "carlini_l2_accuracy"),
        Index("idx_ood", "ood_auc"),
        Index("idx_mia", "mia_auc"),
        Index("idx_watermark", "watermark_accuracy"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "acc_train_clean": float(self.acc_train_clean) if self.acc_train_clean else None,
            "acc_test_clean": float(self.acc_test_clean) if self.acc_test_clean else None,
            "pgd_accuracy": float(self.pgd_accuracy) if self.pgd_accuracy else None,
            "carlini_l2_accuracy": float(self.carlini_l2_accuracy) if self.carlini_l2_accuracy else None,
            "ood_auc": float(self.ood_auc) if self.ood_auc else None,
            "fingerprinting_score": float(self.fingerprinting_score) if self.fingerprinting_score else None,
            "attack_success_rate": float(self.attack_success_rate) if self.attack_success_rate else None,
            "privacy_epsilon": float(self.privacy_epsilon) if self.privacy_epsilon else None,
            "dp_accuracy": float(self.dp_accuracy) if self.dp_accuracy else None,
            "mia_auc": float(self.mia_auc) if self.mia_auc else None,
            "watermark_accuracy": float(self.watermark_accuracy) if self.watermark_accuracy else None,
        }
    
    def __repr__(self):
        return f"<CombinationMetrics(comb={self.combination_id}, acc={self.acc_test_clean})>"


# ============================================================
# Tool Execution Tables
# ============================================================

class ToolExecution(Base):
    """Individual tool execution records."""
    __tablename__ = "tool_executions"
    
    execution_id = Column(Integer, primary_key=True, autoincrement=True)
    combination_id = Column(Integer, ForeignKey("combinations.combination_id", ondelete="CASCADE"), nullable=False)
    tool_id = Column(Integer, ForeignKey("tools.tool_id"), nullable=False)
    stage = Column(Enum(StageEnum), nullable=False)
    
    # Cache information
    cache_key = Column(String(64))
    cache_hit = Column(Boolean, default=False)
    
    # Execution details
    status = Column(Enum(ExecutionStatusEnum), default=ExecutionStatusEnum.PENDING)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    duration_sec = Column(DECIMAL(10, 3))
    
    # Resource usage
    peak_memory_mb = Column(DECIMAL(10, 3))
    gpu_memory_mb = Column(DECIMAL(10, 3))
    cpu_percent = Column(DECIMAL(5, 2))
    
    # Output information
    output_path = Column(String(500))
    log_path = Column(String(500))
    error_message = Column(Text)
    
    # Container info
    container_id = Column(String(64))
    exit_code = Column(Integer)
    
    # Relationships
    combination = relationship("Combination", back_populates="executions")
    tool = relationship("Tool", back_populates="executions")
    
    __table_args__ = (
        Index("idx_exec_comb", "combination_id"),
        Index("idx_exec_tool", "tool_id"),
        Index("idx_exec_stage", "stage"),
        Index("idx_exec_status", "status"),
        Index("idx_exec_cache_key", "cache_key"),
        Index("idx_exec_cache_hit", "cache_hit"),
    )
    
    def __repr__(self):
        return f"<ToolExecution(id={self.execution_id}, tool={self.tool_id}, status='{self.status}')>"


# ============================================================
# Artifact Tables
# ============================================================

class ArtifactNode(Base):
    """Artifact nodes (content-addressable storage)."""
    __tablename__ = "artifact_nodes"
    
    node_id = Column(Integer, primary_key=True, autoincrement=True)
    node_hash = Column(String(64), unique=True, nullable=False)
    tool_identity_hash = Column(String(64))
    tool_name = Column(String(100))
    stage = Column(Enum(StageEnum))
    
    # Parent nodes (stored as JSON array of hashes)
    parent_hashes = Column(JSON)
    
    # Execution info
    duration_sec = Column(DECIMAL(10, 3))
    total_size_bytes = Column(BIGINT)
    
    # Storage path
    storage_path = Column(String(500))
    
    created_at = Column(DateTime, default=func.current_timestamp())
    last_accessed_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    access_count = Column(Integer, default=1)
    
    # Relationships
    files = relationship("ArtifactFile", back_populates="node", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_node_hash", "node_hash"),
        Index("idx_tool_identity", "tool_identity_hash"),
        Index("idx_tool_name", "tool_name"),
        Index("idx_node_stage", "stage"),
    )
    
    def __repr__(self):
        return f"<ArtifactNode(hash='{self.node_hash[:8]}...', tool='{self.tool_name}')>"


class ArtifactFile(Base):
    """Files within artifact nodes."""
    __tablename__ = "artifact_files"
    
    file_id = Column(Integer, primary_key=True, autoincrement=True)
    node_id = Column(Integer, ForeignKey("artifact_nodes.node_id", ondelete="CASCADE"), nullable=False)
    relative_path = Column(String(500), nullable=False)
    file_size_bytes = Column(BIGINT)
    file_hash = Column(String(64))
    mime_type = Column(String(100))
    
    # Relationships
    node = relationship("ArtifactNode", back_populates="files")
    
    __table_args__ = (
        UniqueConstraint("node_id", "relative_path", name="uk_node_file"),
        Index("idx_file_node", "node_id"),
        Index("idx_file_hash", "file_hash"),
    )
    
    def __repr__(self):
        return f"<ArtifactFile(path='{self.relative_path}', size={self.file_size_bytes})>"


class OutputFileProvenance(Base):
    """Output file provenance tracking."""
    __tablename__ = "output_file_provenance"
    
    provenance_id = Column(Integer, primary_key=True, autoincrement=True)
    combination_id = Column(Integer, ForeignKey("combinations.combination_id", ondelete="CASCADE"), nullable=False)
    file_name = Column(String(255), nullable=False)
    source_path = Column(String(500))
    stage = Column(Enum(StageEnum))
    tool_name = Column(String(100))
    was_copied = Column(Boolean, default=False)
    
    # Relationships
    combination = relationship("Combination", back_populates="output_files")
    
    __table_args__ = (
        UniqueConstraint("combination_id", "file_name", name="uk_comb_file"),
        Index("idx_prov_comb", "combination_id"),
        Index("idx_prov_tool", "tool_name"),
    )
    
    def __repr__(self):
        return f"<OutputFileProvenance(file='{self.file_name}', tool='{self.tool_name}')>"


# ============================================================
# Helper Functions
# ============================================================

def create_session(database_url: str) -> Session:
    """
    Create a new database session.
    
    Args:
        database_url: SQLAlchemy database URL 
                     (e.g., "mysql+mysqlconnector://user:pass@host/db")
    
    Returns:
        SQLAlchemy Session object
    """
    engine = create_engine(database_url, echo=False, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def init_database(database_url: str) -> None:
    """
    Initialize database tables.
    
    Args:
        database_url: SQLAlchemy database URL
    """
    engine = create_engine(database_url, echo=True)
    Base.metadata.create_all(bind=engine)
