"""
Tests for scheduler's converter tool invocation.

From docs/Workflow.md:
- "In the in-stages there might be some tools which actually deal with other 
  model format. so they take input/give output in format other than pytorch 
  which is supported by pipeline."
- "Converter tool is only invoked when: 
  a) tool takes input in format other than pytorch and 
  b) tool gives output in format other than pytorch."
- "In case a tool outputs tensorflow model, then the converter tool should 
  convert the model to pytorch format."
- "Similarly, if a tool takes input in tensorflow format, then the converter 
  tool should convert the saved pytorch model to tensorflow format and pass 
  it as input to the tool."
- "The scheduler needs to invoke the converter tool when it is needed."

These tests verify that the scheduler correctly identifies when converter tools
are needed and invokes them appropriately.
"""

import pytest
from typing import List, Optional

artifacts_mod = pytest.importorskip(
    "src.pipeline.artifacts",
    reason="Converter/artifact framework module is not available in current pipeline package",
)
converter_mod = pytest.importorskip(
    "src.pipeline.converter",
    reason="Converter module is not available in current pipeline package",
)

from src.pipeline.tasks import (
    Task,
    TaskStatus,
    TaskType,
    TaskFactory,
    PostTrainingTask,
    InTrainingTask,
    DeploymentTask,
    clear_task_registry,
)
from src.pipeline.tools import ToolDefinition, ContainerConfig
from src.pipeline.workflow import Workflow, WorkflowFactory
from src.pipeline.pipeline import DefenseEvaluationPipeline
ModelFramework = artifacts_mod.ModelFramework
ConversionPlanner = converter_mod.ConversionPlanner
ConversionSpec = converter_mod.ConversionSpec
ConverterTool = converter_mod.ConverterTool
CONVERTER_IMAGES = converter_mod.CONVERTER_IMAGES
from src.backend.scheduler.priority_scheduler import PriorityScheduler


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def pytorch_tool():
    """A tool that works with PyTorch models (the standard format)."""
    return ToolDefinition(
        name="pytorch_tool",
        container=ContainerConfig(
            image="test/pytorch_tool:v1",
            command="python run.py"
        )
    )


@pytest.fixture
def tensorflow_tool():
    """A tool that requires TensorFlow format."""
    return ToolDefinition(
        name="tensorflow_tool",
        container=ContainerConfig(
            image="test/tensorflow_tool:v1",
            command="python run.py"
        ),
        framework_label="tensorflow"
    )


@pytest.fixture
def keras_tool():
    """A tool that requires Keras format."""
    return ToolDefinition(
        name="keras_tool",
        container=ContainerConfig(
            image="test/keras_tool:v1",
            command="python run.py"
        ),
        framework_label="keras"
    )


@pytest.fixture
def onnx_tool():
    """A tool that requires ONNX format."""
    return ToolDefinition(
        name="onnx_tool",
        container=ContainerConfig(
            image="test/onnx_tool:v1",
            command="python run.py"
        ),
        framework_label="onnx"
    )


@pytest.fixture(autouse=True)
def cleanup_task_registry():
    """Clear task registry before each test."""
    clear_task_registry()
    yield
    clear_task_registry()


def create_in_training_task(
    tool: ToolDefinition,
    output_framework: str = "pytorch",
    dependencies: List[Task] = None
) -> InTrainingTask:
    """Create an in-training task with specified output framework."""
    # Directly instantiate to set framework fields
    task = InTrainingTask(
        tool=tool,
        config={"tool": tool.name},
        dependencies=dependencies or [],
        output_framework=output_framework
    )
    return task


def create_post_training_task(
    tool: ToolDefinition,
    required_framework: str = "pytorch",
    output_framework: str = "pytorch",
    dependencies: List[Task] = None
) -> PostTrainingTask:
    """Create a post-training task with specified frameworks."""
    # Directly instantiate to set framework fields
    task = PostTrainingTask(
        tool=tool,
        config={"tool": tool.name},
        dependencies=dependencies or [],
        required_framework=required_framework,
        output_framework=output_framework
    )
    return task


def create_deployment_task(
    tool: ToolDefinition,
    required_framework: str = "pytorch",
    output_framework: str = "pytorch",
    dependencies: List[Task] = None
) -> DeploymentTask:
    """Create a deployment task with specified frameworks."""
    # Directly instantiate to set framework fields
    task = DeploymentTask(
        tool=tool,
        config={"tool": tool.name},
        dependencies=dependencies or [],
        required_framework=required_framework,
        output_framework=output_framework
    )
    return task


# ============================================================================
# Test: ConversionPlanner - Detecting When Conversion is Needed
# ============================================================================


class TestConversionPlannerDetection:
    """
    Tests for ConversionPlanner's ability to detect when conversions are needed.
    
    From Workflow.md:
    "Converter tool is only invoked when:
    a) tool takes input in format other than pytorch
    b) tool gives output in format other than pytorch"
    """
    
    def test_no_conversion_needed_for_pytorch_to_pytorch(self):
        """No conversion when both input and output are PyTorch."""
        planner = ConversionPlanner()
        
        assert not planner.needs_pre_conversion(
            ModelFramework.PYTORCH, ModelFramework.PYTORCH
        )
        assert not planner.needs_post_conversion(ModelFramework.PYTORCH)
    
    def test_pre_conversion_needed_when_tool_requires_tensorflow(self):
        """
        From Workflow.md: "if a tool takes input in tensorflow format, then 
        the converter tool should convert the saved pytorch model to 
        tensorflow format"
        """
        planner = ConversionPlanner()
        
        # Current model is PyTorch, tool requires TensorFlow
        assert planner.needs_pre_conversion(
            ModelFramework.PYTORCH, ModelFramework.TENSORFLOW
        )
    
    def test_post_conversion_needed_when_tool_outputs_tensorflow(self):
        """
        From Workflow.md: "In case a tool outputs tensorflow model, then the 
        converter tool should convert the model to pytorch format."
        """
        planner = ConversionPlanner()
        
        assert planner.needs_post_conversion(ModelFramework.TENSORFLOW)
    
    def test_no_pre_conversion_when_formats_match(self):
        """No pre-conversion when tool's required format matches current."""
        planner = ConversionPlanner()
        
        # Both TensorFlow - no conversion needed
        assert not planner.needs_pre_conversion(
            ModelFramework.TENSORFLOW, ModelFramework.TENSORFLOW
        )
    
    def test_conversion_needed_for_all_non_pytorch_outputs(self):
        """All non-PyTorch outputs need post-conversion."""
        planner = ConversionPlanner()
        
        non_pytorch_frameworks = [
            ModelFramework.TENSORFLOW,
            ModelFramework.KERAS,
            ModelFramework.ONNX,
        ]
        
        for framework in non_pytorch_frameworks:
            assert planner.needs_post_conversion(framework), \
                f"Expected post-conversion for {framework}"


# ============================================================================
# Test: ConversionPlanner - Analyzing Workflow Conversions
# ============================================================================


class TestConversionPlannerAnalysis:
    """Tests for ConversionPlanner's workflow analysis."""
    
    def test_analyze_pytorch_only_workflow(self, pytorch_tool):
        """Workflow with only PyTorch tasks needs no conversions."""
        task1 = create_post_training_task(pytorch_tool)
        task2 = create_post_training_task(
            pytorch_tool, dependencies=[task1]
        )
        
        planner = ConversionPlanner()
        conversions = planner.analyze_workflow_conversions([task1, task2])
        
        assert len(conversions) == 0
    
    def test_analyze_workflow_with_tensorflow_input(self, pytorch_tool, tensorflow_tool):
        """Workflow with TensorFlow-requiring task needs pre-conversion."""
        task_pytorch = create_post_training_task(pytorch_tool)
        task_tf = create_post_training_task(
            tensorflow_tool, 
            required_framework="tensorflow",
            output_framework="tensorflow",
            dependencies=[task_pytorch]
        )
        
        planner = ConversionPlanner()
        conversions = planner.analyze_workflow_conversions(
            [task_pytorch, task_tf]
        )
        
        # Should have: pre-conversion (PyTorch->TF) and post-conversion (TF->PyTorch)
        assert len(conversions) >= 1
        
        # Find the pre-conversion for task_tf
        pre_conversions = [c for c in conversions if c["position"] == "before"]
        assert len(pre_conversions) >= 1
        
        pre_conv = pre_conversions[0]
        assert pre_conv["source"] == ModelFramework.PYTORCH
        assert pre_conv["target"] == ModelFramework.TENSORFLOW
    
    def test_analyze_workflow_with_tensorflow_output(self, tensorflow_tool):
        """Workflow with TensorFlow output needs post-conversion."""
        task_tf = create_post_training_task(
            tensorflow_tool,
            required_framework="tensorflow",
            output_framework="tensorflow"
        )
        
        planner = ConversionPlanner()
        # Initial framework is TensorFlow (pretend we already converted)
        conversions = planner.analyze_workflow_conversions(
            [task_tf],
            initial_framework=ModelFramework.TENSORFLOW
        )
        
        # Should have post-conversion (TF->PyTorch)
        post_conversions = [c for c in conversions if c["position"] == "after"]
        assert len(post_conversions) == 1
        
        post_conv = post_conversions[0]
        assert post_conv["source"] == ModelFramework.TENSORFLOW
        assert post_conv["target"] == ModelFramework.PYTORCH
    
    def test_multiple_conversions_in_workflow(
        self, pytorch_tool, tensorflow_tool, keras_tool
    ):
        """
        Complex workflow with multiple framework changes needs multiple conversions.
        
        PyTorch task -> TensorFlow task -> Keras task
        Should have conversions at each boundary.
        """
        task_pt = create_post_training_task(pytorch_tool)
        task_tf = create_post_training_task(
            tensorflow_tool,
            required_framework="tensorflow",
            output_framework="tensorflow",
            dependencies=[task_pt]
        )
        task_keras = create_post_training_task(
            keras_tool,
            required_framework="keras",
            output_framework="keras",
            dependencies=[task_tf]
        )
        
        planner = ConversionPlanner()
        conversions = planner.analyze_workflow_conversions(
            [task_pt, task_tf, task_keras]
        )
        
        # Should have multiple conversions
        assert len(conversions) >= 3  # PT->TF, TF->PT, PT->Keras, Keras->PT


# ============================================================================
# Test: ConverterTool Creation
# ============================================================================


class TestConverterToolCreation:
    """Tests for ConverterTool creation and configuration."""
    
    def test_tensorflow_to_pytorch_converter(self):
        """Create a TensorFlow to PyTorch converter."""
        converter = ConverterTool(
            source_framework=ModelFramework.TENSORFLOW,
            target_framework=ModelFramework.PYTORCH
        )
        
        assert converter.source_framework == ModelFramework.TENSORFLOW
        assert converter.target_framework == ModelFramework.PYTORCH
        assert "tf2pt" in converter.container_image or "converter" in converter.container_image
    
    def test_pytorch_to_tensorflow_converter(self):
        """Create a PyTorch to TensorFlow converter."""
        converter = ConverterTool(
            source_framework=ModelFramework.PYTORCH,
            target_framework=ModelFramework.TENSORFLOW
        )
        
        assert converter.source_framework == ModelFramework.PYTORCH
        assert converter.target_framework == ModelFramework.TENSORFLOW
        assert "pt2tf" in converter.container_image or "converter" in converter.container_image
    
    def test_converter_to_tool_definition(self):
        """Converter can be converted to ToolDefinition for task creation."""
        converter = ConverterTool(
            source_framework=ModelFramework.TENSORFLOW,
            target_framework=ModelFramework.PYTORCH
        )
        
        tool_def = converter.to_tool_definition()
        
        assert isinstance(tool_def, ToolDefinition)
        assert "converter" in tool_def.name
        assert "tensorflow" in tool_def.name
        assert "pytorch" in tool_def.name
    
    def test_converter_command_includes_frameworks(self):
        """Converter command includes source and target frameworks."""
        converter = ConverterTool(
            source_framework=ModelFramework.KERAS,
            target_framework=ModelFramework.PYTORCH
        )
        
        command = converter.get_command("/input/model.keras", "/output/model.pt")
        
        assert "keras" in command
        assert "pytorch" in command
        assert "/input/model.keras" in command
        assert "/output/model.pt" in command


# ============================================================================
# Test: ConversionSpec
# ============================================================================


class TestConversionSpec:
    """Tests for ConversionSpec functionality."""
    
    def test_is_needed_when_frameworks_differ(self):
        """Conversion is needed when frameworks differ."""
        spec = ConversionSpec(
            source_framework=ModelFramework.PYTORCH,
            target_framework=ModelFramework.TENSORFLOW
        )
        
        assert spec.is_needed()
    
    def test_is_not_needed_when_frameworks_same(self):
        """Conversion is not needed when frameworks are the same."""
        spec = ConversionSpec(
            source_framework=ModelFramework.PYTORCH,
            target_framework=ModelFramework.PYTORCH
        )
        
        assert not spec.is_needed()
    
    def test_conversion_key_format(self):
        """Conversion key follows expected format."""
        spec = ConversionSpec(
            source_framework=ModelFramework.TENSORFLOW,
            target_framework=ModelFramework.PYTORCH
        )
        
        key = spec.get_conversion_key()
        assert key == "tensorflow_to_pytorch"


# ============================================================================
# Test: Task Framework Requirements
# ============================================================================


class TestTaskFrameworkRequirements:
    """Tests verifying tasks correctly report their framework requirements."""
    
    def test_post_training_task_reports_required_framework(self, tensorflow_tool):
        """Post-training task reports its required input framework."""
        task = create_post_training_task(
            tensorflow_tool,
            required_framework="tensorflow"
        )
        
        required = task.get_required_framework()
        assert required == ModelFramework.TENSORFLOW
    
    def test_post_training_task_reports_output_framework(self, tensorflow_tool):
        """Post-training task reports its output framework."""
        task = create_post_training_task(
            tensorflow_tool,
            output_framework="tensorflow"
        )
        
        output = task.get_output_framework()
        assert output == ModelFramework.TENSORFLOW
    
    def test_in_training_task_reports_output_framework(self, tensorflow_tool):
        """In-training task reports its output framework."""
        task = create_in_training_task(
            tensorflow_tool,
            output_framework="tensorflow"
        )
        
        output = task.get_output_framework()
        assert output == ModelFramework.TENSORFLOW
    
    def test_deployment_task_reports_frameworks(self, keras_tool):
        """Deployment task reports both required and output frameworks."""
        task = create_deployment_task(
            keras_tool,
            required_framework="keras",
            output_framework="keras"
        )
        
        assert task.get_required_framework() == ModelFramework.KERAS
        assert task.get_output_framework() == ModelFramework.KERAS
    
    def test_pytorch_is_default_framework(self, pytorch_tool):
        """PyTorch is the default framework when not specified."""
        task = create_post_training_task(pytorch_tool)
        
        required = task.get_required_framework()
        output = task.get_output_framework()
        
        assert required == ModelFramework.PYTORCH
        assert output == ModelFramework.PYTORCH


# ============================================================================
# Test: Scheduler Integration with Conversion Planning
# ============================================================================


class TestSchedulerConversionIntegration:
    """
    Tests for scheduler's integration with conversion planning.
    
    These tests verify that the scheduler correctly handles workflows
    that require model format conversions.
    """
    
    def test_scheduler_initializes_with_mixed_framework_workflow(
        self, pytorch_tool, tensorflow_tool
    ):
        """Scheduler can initialize with workflows containing different frameworks."""
        task_pt = create_post_training_task(pytorch_tool)
        task_tf = create_post_training_task(
            tensorflow_tool,
            required_framework="tensorflow",
            output_framework="tensorflow",
            dependencies=[task_pt]
        )
        
        workflow = WorkflowFactory.create_workflow(
            name="mixed_framework",
            tasks=[task_pt, task_tf]
        )
        pipeline = DefenseEvaluationPipeline(
            name="test",
            workflows=[workflow]
        )
        
        scheduler = PriorityScheduler(pipeline)
        
        assert len(scheduler.get_all_tasks()) == 2
    
    def test_scheduler_maintains_task_order_with_framework_changes(
        self, pytorch_tool, tensorflow_tool
    ):
        """
        Scheduler maintains correct task order even with framework changes.
        Dependencies must still be completed before dependent tasks.
        """
        task_pt = create_post_training_task(pytorch_tool)
        task_tf = create_post_training_task(
            tensorflow_tool,
            required_framework="tensorflow",
            dependencies=[task_pt]
        )
        
        workflow = WorkflowFactory.create_workflow(
            name="ordered",
            tasks=[task_pt, task_tf]
        )
        pipeline = DefenseEvaluationPipeline(
            name="test",
            workflows=[workflow]
        )
        
        scheduler = PriorityScheduler(pipeline)
        
        # First task should be PyTorch task (no deps)
        first = scheduler.get_next_task()
        assert first is task_pt
        
        # TensorFlow task should not be ready yet
        assert scheduler.get_next_task() is None
        
        # Complete PyTorch task
        scheduler.update_task_status(task_pt.id, TaskStatus.COMPLETED)
        
        # Now TensorFlow task should be ready
        second = scheduler.get_next_task()
        assert second is task_tf
    
    def test_workflow_can_query_conversion_requirements(
        self, pytorch_tool, tensorflow_tool, keras_tool
    ):
        """
        Workflow can be analyzed for conversion requirements.
        
        This is crucial for the scheduler to know when to invoke converters.
        """
        task_pt = create_post_training_task(pytorch_tool)
        task_tf = create_post_training_task(
            tensorflow_tool,
            required_framework="tensorflow",
            output_framework="tensorflow",
            dependencies=[task_pt]
        )
        task_keras = create_post_training_task(
            keras_tool,
            required_framework="keras",
            output_framework="pytorch",  # Outputs back to PyTorch
            dependencies=[task_tf]
        )
        
        workflow = WorkflowFactory.create_workflow(
            name="multi_convert",
            tasks=[task_pt, task_tf, task_keras]
        )
        
        # Analyze conversions needed
        planner = ConversionPlanner()
        conversions = planner.analyze_workflow_conversions(workflow.tasks)
        
        # Should identify conversions:
        # 1. Before task_tf: PyTorch -> TensorFlow
        # 2. After task_tf: TensorFlow -> PyTorch (for storage)
        # 3. Before task_keras: PyTorch -> Keras
        # (task_keras outputs PyTorch, so no post-conversion)
        
        assert len(conversions) >= 3
        
        # Check conversion positions
        task_ids_needing_conversion = [c["task_id"] for c in conversions]
        assert task_tf.id in task_ids_needing_conversion
        assert task_keras.id in task_ids_needing_conversion


# ============================================================================
# Test: Converter Container Images
# ============================================================================


class TestConverterContainerImages:
    """Tests for converter container image configuration."""
    
    def test_converter_images_exist_for_common_conversions(self):
        """Converter images are configured for common framework conversions."""
        expected_conversions = [
            "tensorflow_to_pytorch",
            "pytorch_to_tensorflow",
            "keras_to_pytorch",
            "pytorch_to_keras",
            "onnx_to_pytorch",
            "pytorch_to_onnx",
        ]
        
        for conversion in expected_conversions:
            assert conversion in CONVERTER_IMAGES, \
                f"Missing converter image for {conversion}"
    
    def test_converter_images_are_valid_docker_references(self):
        """Converter images look like valid Docker image references."""
        for conversion, image in CONVERTER_IMAGES.items():
            assert "/" in image, f"Image {image} should have registry/path"
            assert ":" in image, f"Image {image} should have a tag"
            assert "landseer" in image.lower(), \
                f"Image {image} should be from landseer project"


# ============================================================================
# Test: End-to-End Conversion Scenario
# ============================================================================


class TestEndToEndConversionScenario:
    """
    End-to-end tests for complete conversion scenarios.
    
    These tests verify the complete flow as described in Workflow.md.
    """
    
    def test_tensorflow_tool_in_pytorch_pipeline(
        self, pytorch_tool, tensorflow_tool
    ):
        """
        A TensorFlow tool in a PyTorch pipeline triggers appropriate conversions.
        
        Scenario:
        1. PyTorch task outputs model in .pt format
        2. TensorFlow task needs .h5 format
        3. Converter should be invoked: PyTorch -> TensorFlow
        4. TensorFlow task outputs .h5
        5. Converter should be invoked: TensorFlow -> PyTorch
        """
        # Setup tasks
        task_pt = create_post_training_task(pytorch_tool)
        task_tf = create_post_training_task(
            tensorflow_tool,
            required_framework="tensorflow",
            output_framework="tensorflow",
            dependencies=[task_pt]
        )
        task_final = create_post_training_task(
            pytorch_tool,
            dependencies=[task_tf]
        )
        
        # Analyze conversions
        planner = ConversionPlanner()
        conversions = planner.analyze_workflow_conversions(
            [task_pt, task_tf, task_final]
        )
        
        # Find conversions around task_tf
        tf_conversions = [c for c in conversions if c["task_id"] == task_tf.id]
        
        # Should have both pre and post conversions
        positions = {c["position"] for c in tf_conversions}
        assert "before" in positions, "Should have pre-conversion for TF task"
        assert "after" in positions, "Should have post-conversion for TF task"
        
        # Verify conversion frameworks
        pre_conv = next(c for c in tf_conversions if c["position"] == "before")
        assert pre_conv["source"] == ModelFramework.PYTORCH
        assert pre_conv["target"] == ModelFramework.TENSORFLOW
        
        post_conv = next(c for c in tf_conversions if c["position"] == "after")
        assert post_conv["source"] == ModelFramework.TENSORFLOW
        assert post_conv["target"] == ModelFramework.PYTORCH
    
    def test_onnx_export_workflow(self, pytorch_tool, onnx_tool):
        """
        Workflow that exports to ONNX for deployment.
        
        Common scenario: train in PyTorch, deploy with ONNX.
        """
        task_pt = create_post_training_task(pytorch_tool)
        task_onnx = create_deployment_task(
            onnx_tool,
            required_framework="onnx",
            output_framework="onnx",
            dependencies=[task_pt]
        )
        
        planner = ConversionPlanner()
        conversions = planner.analyze_workflow_conversions([task_pt, task_onnx])
        
        # Should have pre-conversion: PyTorch -> ONNX
        pre_conversions = [c for c in conversions if c["position"] == "before"]
        assert len(pre_conversions) >= 1
        
        pre_conv = pre_conversions[0]
        assert pre_conv["source"] == ModelFramework.PYTORCH
        assert pre_conv["target"] == ModelFramework.ONNX
