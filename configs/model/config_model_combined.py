"""
Combined PyTorch/TensorFlow ResNet configuration with lazy loading.
This module automatically detects available frameworks and provides the appropriate model implementation.
"""

import sys
import importlib.util

# Framework detection flags
_torch_available = False
_tensorflow_available = False
_active_framework = None

# Lazy imports - will be set when framework is detected
torch = None
nn = None
F = None
init = None
Variable = None
tf = None
layers = None

def _check_framework_availability():
    """Check which frameworks are available in the environment."""
    global _torch_available, _tensorflow_available
    
    # Check PyTorch
    if importlib.util.find_spec("torch") is not None:
        _torch_available = True
    
    # Check TensorFlow
    if importlib.util.find_spec("tensorflow") is not None:
        _tensorflow_available = True

def _load_pytorch():
    """Lazy load PyTorch modules."""
    global torch, nn, F, init, Variable, _active_framework
    
    if not _torch_available:
        raise ImportError("PyTorch is not available in the environment")
    
    import torch as _torch
    import torch.nn as _nn
    import torch.nn.functional as _F
    import torch.nn.init as _init
    from torch.autograd import Variable as _Variable

    torch = _torch
    nn = _nn
    F = _F
    init = _init
    Variable = _Variable
    _active_framework = "pytorch"

def _load_tensorflow():
    """Lazy load TensorFlow modules."""
    global tf, layers, _active_framework
    
    if not _tensorflow_available:
        raise ImportError("TensorFlow is not available in the environment")
    
    import tensorflow as _tf
    from tensorflow.keras import layers as _layers
    
    tf = _tf
    layers = _layers
    _active_framework = "tensorflow"

# Initialize framework availability check
_check_framework_availability()

# ============================================================================
# PyTorch Implementation
# ============================================================================

def _weights_init(m):
    """Initialize weights for PyTorch model."""
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

# PyTorch classes defined only when PyTorch is loaded
def _create_pytorch_classes():
    """Create PyTorch model classes when needed"""
    
    class PyTorchLambdaLayer(nn.Module):
        def __init__(self, lambd):
            super(PyTorchLambdaLayer, self).__init__()
            self.lambd = lambd

        def forward(self, x):
            return self.lambd(x)

    class PyTorchBasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1, option='B'):
            super(PyTorchBasicBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                if option == 'A':
                    self.shortcut = PyTorchLambdaLayer(lambda x:
                                                F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
                elif option == 'B':
                    self.shortcut = nn.Sequential(
                         nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                         nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)
                    )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + self.shortcut(x)
            out = F.relu(out)
            return out

    class PyTorchResNet(nn.Module):
        def __init__(self, block, num_blocks, num_classes=10):
            super(PyTorchResNet, self).__init__()
            self.in_planes = 16

            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16, track_running_stats=False)
            self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
            self.linear = nn.Linear(64, num_classes)

            self.apply(_weights_init)

        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = F.avg_pool2d(out, out.size()[3])
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out
    
    return PyTorchLambdaLayer, PyTorchBasicBlock, PyTorchResNet

# ============================================================================
# TensorFlow Implementation
# ============================================================================

def _create_tensorflow_classes():
    """Create TensorFlow model classes when needed"""
    
    class TensorFlowIdentity(layers.Layer):
        """Identity layer for shortcut connections."""
        def call(self, x, training=False):
            return x

    class TensorFlowBasicBlock(tf.keras.Model):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1):
            super(TensorFlowBasicBlock, self).__init__()
            self.conv1 = layers.Conv2D(planes, kernel_size=3, strides=stride,
                                       padding='same', use_bias=False)
            self.bn1 = layers.BatchNormalization(momentum=0.1, epsilon=1e-5)

            self.conv2 = layers.Conv2D(planes, kernel_size=3, strides=1,
                                       padding='same', use_bias=False)
            self.bn2 = layers.BatchNormalization(momentum=0.1, epsilon=1e-5)

            # Shortcut connection
            if stride != 1 or in_planes != planes:
                self.shortcut = tf.keras.Sequential([
                    layers.Conv2D(planes, kernel_size=1, strides=stride,
                                  padding='valid', use_bias=False),
                    layers.BatchNormalization(momentum=0.1, epsilon=1e-5)
                ])
            else:
                self.shortcut = TensorFlowIdentity()

        def call(self, x, training=False):
            out = self.conv1(x)
            out = self.bn1(out, training=training)
            out = tf.nn.relu(out)

            out = self.conv2(out)
            out = self.bn2(out, training=training)

            out += self.shortcut(x, training=training)
            out = tf.nn.relu(out)
            return out

    class TensorFlowResNet(tf.keras.Model):
        def __init__(self, block, num_blocks, num_classes=10):
            super(TensorFlowResNet, self).__init__()
            self.in_planes = 16

            self.conv1 = layers.Conv2D(16, kernel_size=3, strides=1, padding='same', use_bias=False)
            self.bn1 = layers.BatchNormalization(momentum=0.1, epsilon=1e-5)

            self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

            self.avgpool = layers.GlobalAveragePooling2D()
            self.fc = layers.Dense(num_classes)

        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1]*(num_blocks - 1)
            layers_list = []
            for s in strides:
                layers_list.append(block(self.in_planes, planes, s))
                self.in_planes = planes * block.expansion
            return tf.keras.Sequential(layers_list)

        def call(self, x, training=False):
            out = self.conv1(x)
            out = self.bn1(out, training=training)
            out = tf.nn.relu(out)

            out = self.layer1(out, training=training)
            out = self.layer2(out, training=training)
            out = self.layer3(out, training=training)

            out = self.avgpool(out)
            out = self.fc(out)
            return out
    
    return TensorFlowIdentity, TensorFlowBasicBlock, TensorFlowResNet

# ============================================================================
# Model Factory Functions
# ============================================================================

def _create_pytorch_resnet(block_config):
    """Create PyTorch ResNet model."""
    if not _torch_available:
        raise RuntimeError("PyTorch is not available in the environment")
    
    if _active_framework != "pytorch":
        _load_pytorch()
    
    # Create classes dynamically
    PyTorchLambdaLayer, PyTorchBasicBlock, PyTorchResNet = _create_pytorch_classes()
    
    return PyTorchResNet(PyTorchBasicBlock, block_config)

def _create_tensorflow_resnet(block_config):
    """Create TensorFlow ResNet model."""
    if not _tensorflow_available:
        raise RuntimeError("TensorFlow is not available in the environment")
    
    if _active_framework != "tensorflow":
        _load_tensorflow()
    
    # Create classes dynamically
    TensorFlowIdentity, TensorFlowBasicBlock, TensorFlowResNet = _create_tensorflow_classes()
    
    return TensorFlowResNet(TensorFlowBasicBlock, block_config)

def _create_resnet(block_config, framework_preference=None):
    """
    Create ResNet model with automatic framework selection.
    
    Args:
        block_config: List specifying the number of blocks in each layer
        framework_preference: 'pytorch', 'tensorflow', or None for auto-detection
    
    Returns:
        ResNet model instance
    """
    if framework_preference == "pytorch":
        return _create_pytorch_resnet(block_config)
    elif framework_preference == "tensorflow":
        return _create_tensorflow_resnet(block_config)
    else:
        # Auto-select framework based on availability
        if _torch_available:
            return _create_pytorch_resnet(block_config)
        elif _tensorflow_available:
            return _create_tensorflow_resnet(block_config)
        else:
            raise RuntimeError("Neither PyTorch nor TensorFlow is available in the environment")

# Public API - ResNet variants
def resnet20(framework_preference=None):
    """Create ResNet-20 model."""
    return _create_resnet([3, 3, 3], framework_preference)

def resnet32(framework_preference=None):
    """Create ResNet-32 model."""
    return _create_resnet([5, 5, 5], framework_preference)

def resnet44(framework_preference=None):
    """Create ResNet-44 model."""
    return _create_resnet([7, 7, 7], framework_preference)

def resnet56(framework_preference=None):
    """Create ResNet-56 model."""
    return _create_resnet([9, 9, 9], framework_preference)

def resnet110(framework_preference=None):
    """Create ResNet-110 model."""
    return _create_resnet([18, 18, 18], framework_preference)

def resnet1202(framework_preference=None):
    """Create ResNet-1202 model."""
    return _create_resnet([200, 200, 200], framework_preference)

# Main configuration function
def config(framework_preference=None):
    """
    Main configuration function that returns a ResNet-20 model.
    
    Args:
        framework_preference: 'pytorch', 'tensorflow', or None for auto-detection
    
    Returns:
        ResNet-20 model instance using the available/preferred framework
    """
    return resnet20(framework_preference)

# Utility functions
def get_available_frameworks():
    """Return list of available frameworks."""
    frameworks = []
    if _torch_available:
        frameworks.append("pytorch")
    if _tensorflow_available:
        frameworks.append("tensorflow")
    return frameworks

def get_active_framework():
    """Return the currently active framework."""
    return _active_framework