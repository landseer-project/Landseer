"""
TensorFlow/Keras equivalent of configs/model/config_model_resnet20_celeba.py.

Identical to config_model_resnet20_tf.py except for NUM_CLASSES (CelebA is
binary in Landseer); model scripts are uploaded as single files, so the two
cannot share code.

``config()`` returns a Keras model with the same architecture, input layout,
initialisation and BatchNorm behaviour as the PyTorch ``config()`` (ResNet20,
option B shortcuts, 2 classes), so a TensorFlow tool can train the pipeline's
model natively and its weights map 1:1 onto the PyTorch state_dict:

- Input is NCHW float32 in [0, 1] like the dataset (data.npy); the model
  transposes to NHWC internally because TensorFlow's CPU convolutions only
  support channels-last.
- 3x3 convolutions pad 1 pixel on every side (ZeroPadding2D + "valid"), as
  PyTorch ``padding=1`` does. Keras ``padding="same"`` with stride 2 pads only
  bottom/right and would shift every downsampling layer by one pixel.
- BatchNormalization uses PyTorch's eps=1e-5 and momentum (Keras 0.9 ==
  PyTorch 0.1); Keras defaults are 1e-3 and 0.99.
- Conv/Dense kernels use untruncated He-normal (std sqrt(2/fan_in)), as
  ``init.kaiming_normal_``; Keras ``he_normal`` is truncated. The Dense bias
  is uniform(+-1/sqrt(fan_in)), PyTorch's nn.Linear default.
- The final pooling reproduces ``F.avg_pool2d(out, out.size()[3])``: a
  width x width window, i.e. global average pooling for square inputs.

Layers are named after the PyTorch modules ("layer2.0.shortcut.0", ...), and
``load_torch_state_dict`` / ``to_torch_state_dict`` convert weights both ways.

Known difference: during training PyTorch updates running_var with the
unbiased batch variance (n/(n-1)) and Keras with the biased one. With
n = batch*H*W this is a relative 1e-5 to 1e-4 per update; forward passes and
gradients are identical.
"""

from typing import Dict, Sequence, Tuple

import numpy as np
import tensorflow as tf

NUM_CLASSES = 2  # CelebA default setup in Landseer is binary classification
NUM_BLOCKS = (3, 3, 3)
INPUT_SHAPE = (3, 32, 32)  # C, H, W as in data.npy

BN_EPSILON = 1e-5
BN_MOMENTUM = 0.9  # Keras convention; PyTorch momentum=0.1


def _kernel_init(seed=None):
    # init.kaiming_normal_(w): normal(0, sqrt(2 / fan_in)), not truncated.
    return tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_in", distribution="untruncated_normal", seed=seed)


def _conv(x, filters, kernel_size, stride, name, seed=None):
    if kernel_size == 3:
        x = tf.keras.layers.ZeroPadding2D(padding=1, name=f"{name}.pad")(x)
    return tf.keras.layers.Conv2D(
        filters, kernel_size, strides=stride, padding="valid", use_bias=False,
        kernel_initializer=_kernel_init(seed), name=name,
    )(x)


def _bn(x, name):
    return tf.keras.layers.BatchNormalization(axis=-1, epsilon=BN_EPSILON, momentum=BN_MOMENTUM, name=name)(x)


def _basic_block(x, in_planes, planes, stride, name, seed=None):
    out = _conv(x, planes, 3, stride, f"{name}.conv1", seed)
    out = tf.keras.layers.ReLU(name=f"{name}.relu1")(_bn(out, f"{name}.bn1"))
    out = _bn(_conv(out, planes, 3, 1, f"{name}.conv2", seed), f"{name}.bn2")
    if stride != 1 or in_planes != planes:
        # Option B (the PyTorch config's default): 1x1 conv + BN.
        shortcut = _bn(_conv(x, planes, 1, stride, f"{name}.shortcut.0", seed), f"{name}.shortcut.1")
    else:
        shortcut = x
    out = tf.keras.layers.Add(name=f"{name}.add")([out, shortcut])
    return tf.keras.layers.ReLU(name=f"{name}.relu2")(out)


def _torch_avg_pool_by_width(x, height, width):
    """F.avg_pool2d(x, x.size()[3]) followed by .view(N, -1) into nn.Linear(64)."""
    if height < width or height >= 2 * width:
        # PyTorch fails here too: the window is larger than the input, or the
        # flattened features would not match nn.Linear(64, ...).
        raise ValueError(f"ResNet20 needs width <= height < 2*width after the stem, got {height}x{width}")
    if height != width:
        x = tf.keras.layers.Cropping2D(((0, height - width), (0, 0)), name="avgpool.crop")(x)
    return tf.keras.layers.GlobalAveragePooling2D(name="avgpool")(x)


def resnet(
    num_blocks: Sequence[int] = NUM_BLOCKS,
    num_classes: int = NUM_CLASSES,
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    seed=None,
) -> tf.keras.Model:
    channels, height, width = input_shape
    inputs = tf.keras.Input(shape=(channels, height, width), name="input")
    x = tf.keras.layers.Permute((2, 3, 1), name="to_nhwc")(inputs)
    x = _conv(x, 16, 3, 1, "conv1", seed)
    x = tf.keras.layers.ReLU(name="relu")(_bn(x, "bn1"))

    in_planes = 16
    for stage, (planes, blocks, first_stride) in enumerate(zip((16, 32, 64), num_blocks, (1, 2, 2)), start=1):
        for index, stride in enumerate([first_stride] + [1] * (blocks - 1)):
            x = _basic_block(x, in_planes, planes, stride, f"layer{stage}.{index}", seed)
            in_planes = planes
            if stride == 2:
                # Conv(k=3, pad=1, s=2) and Conv(k=1, s=2) both give ceil(n / 2).
                height, width = (height + 1) // 2, (width + 1) // 2

    x = _torch_avg_pool_by_width(x, height, width)
    fan_in = 64
    bound = 1.0 / np.sqrt(fan_in)
    outputs = tf.keras.layers.Dense(
        num_classes, kernel_initializer=_kernel_init(seed),
        bias_initializer=tf.keras.initializers.RandomUniform(-bound, bound, seed=seed), name="linear",
    )(x)
    return tf.keras.Model(inputs, outputs, name="resnet20_celeba")


def config() -> tf.keras.Model:
    return resnet()


# ---------------------------------------------------------------------------
# PyTorch state_dict <-> Keras weights
# ---------------------------------------------------------------------------

# PyTorch tensor suffix -> (Keras weight index in layer.weights, axis permutation torch -> keras)
_CONV = {"weight": (0, (2, 3, 1, 0))}                 # (out, in, kh, kw) -> (kh, kw, in, out)
_DENSE = {"weight": (0, (1, 0)), "bias": (1, None)}   # (out, in) -> (in, out)
_BN = {"weight": (0, None), "bias": (1, None), "running_mean": (2, None), "running_var": (3, None)}


def weight_map(model: tf.keras.Model) -> Dict[str, Tuple[str, int, tuple]]:
    """PyTorch state_dict key -> (Keras layer name, weight index, permutation)."""
    mapping = {}
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            table = _CONV
        elif isinstance(layer, tf.keras.layers.Dense):
            table = _DENSE
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            table = _BN
        else:
            continue
        for suffix, (index, perm) in table.items():
            mapping[f"{layer.name}.{suffix}"] = (layer.name, index, perm)
    return mapping


def load_torch_state_dict(model: tf.keras.Model, state_dict) -> None:
    """Copy a PyTorch ResNet20 state_dict (tensors or arrays) into ``model``.

    Every tensor except ``num_batches_tracked`` must be used, and every Keras
    weight must be set; shapes are checked.
    """
    mapping = weight_map(model)
    arrays = {k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)) for k, v in state_dict.items()}
    extra = sorted(k for k in arrays if k not in mapping and not k.endswith("num_batches_tracked"))
    missing = sorted(k for k in mapping if k not in arrays)
    if extra or missing:
        raise ValueError(f"state_dict does not match the Keras ResNet: unexpected {extra[:5]}, missing {missing[:5]}")
    by_layer: Dict[str, Dict[int, np.ndarray]] = {}
    for key, (layer_name, index, perm) in mapping.items():
        value = arrays[key] if perm is None else np.transpose(arrays[key], perm)
        by_layer.setdefault(layer_name, {})[index] = value
    for layer_name, values in by_layer.items():
        layer = model.get_layer(layer_name)
        current = layer.get_weights()
        if len(values) != len(current):
            raise ValueError(f"{layer_name}: {len(values)} tensors for {len(current)} weights")
        new = []
        for index, old in enumerate(current):
            if values[index].shape != old.shape:
                raise ValueError(f"{layer_name}[{index}]: shape {values[index].shape} != {old.shape}")
            new.append(values[index].astype(old.dtype))
        layer.set_weights(new)


def to_torch_state_dict(model: tf.keras.Model) -> Dict[str, np.ndarray]:
    """Keras weights as PyTorch state_dict arrays (without num_batches_tracked)."""
    state = {}
    for key, (layer_name, index, perm) in weight_map(model).items():
        value = model.get_layer(layer_name).get_weights()[index]
        state[key] = np.ascontiguousarray(value if perm is None else np.transpose(value, np.argsort(perm)))
    return state
