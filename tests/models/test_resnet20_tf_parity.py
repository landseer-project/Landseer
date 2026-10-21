"""
Parity of the Keras ResNet20 (configs/model/config_model_resnet20_tf.py) with the
PyTorch one (configs/model/config_model_resnet20.py), and of the CelebA pair.

Needs torch and tensorflow, e.g. in the adapter image:

    docker run --rm -v "$PWD":/repo -w /repo landseer-local/adapter_pytorch_tensorflow:dev \
        sh -c "pip install -q pytest && python -m pytest tests/models -q -p no:cacheprovider"

Tests cover:
- Same tensors: every state_dict entry maps to one Keras weight with a matching shape; equal parameter counts
- Eval-mode logits with random weights and non-trivial BatchNorm statistics, on 32x32, odd and non-square inputs
- Train-mode logits (batch statistics) and gradients of the loss for every weight
- In float64 logits, loss, gradients and an SGD step agree to ~1e-12, so the
  models are the same function; float32 checks allow for rounding, which
  train-mode BatchNorm amplifies in gradients (up to ~0.4% of a tensor's norm)
- Running statistics after a training step (up to PyTorch's unbiased variance)
- Round trip Keras -> state_dict -> PyTorch with strict loading; .keras save/load
- Initialisation distributions and BatchNorm hyperparameters
"""

import contextlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
tf = pytest.importorskip("tensorflow")

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS = REPO_ROOT / "configs" / "model"

PAIRS = {
    "cifar10": ("config_model_resnet20.py", "config_model_resnet20_tf.py", 10),
    "celeba": ("config_model_resnet20_celeba.py", "config_model_resnet20_celeba_tf.py", 2),
}


def _module(filename):
    spec = importlib.util.spec_from_file_location(filename[:-3], MODELS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=sorted(PAIRS), scope="module")
def pair(request):
    torch_file, tf_file, classes = PAIRS[request.param]
    return _module(torch_file), _module(tf_file), classes


def _randomize(net, seed=0):
    """Random weights and BatchNorm statistics far from the defaults."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, tensor in net.state_dict().items():
            if name.endswith("num_batches_tracked"):
                continue
            if name.endswith("running_var"):
                tensor.copy_(torch.rand(tensor.shape, generator=gen) * 1.5 + 0.25)
            elif name.endswith("running_mean"):
                tensor.copy_(torch.randn(tensor.shape, generator=gen) * 0.3)
            elif ".bn" in name or "shortcut.1" in name or name.startswith("bn1"):
                tensor.copy_(torch.randn(tensor.shape, generator=gen) * 0.2 + (1.0 if name.endswith("weight") else 0.0))
            else:
                tensor.copy_(torch.randn(tensor.shape, generator=gen) * (2.0 / max(1, tensor[0].numel())) ** 0.5)
    return net


def _pair_models(pair, input_shape=(3, 32, 32), seed=0, dtype="float32"):
    torch_mod, tf_mod, classes = pair
    torch.manual_seed(seed)
    net = _randomize(torch_mod.config(), seed).to(getattr(torch, dtype))
    with _keras_floatx(dtype):
        keras = tf_mod.resnet(num_classes=classes, input_shape=input_shape)
    tf_mod.load_torch_state_dict(keras, net.state_dict())
    return net, keras, tf_mod


@contextlib.contextmanager
def _keras_floatx(dtype):
    previous = tf.keras.backend.floatx()
    tf.keras.backend.set_floatx(dtype)
    try:
        yield
    finally:
        tf.keras.backend.set_floatx(previous)


def _assert_norm_close(a, b, rel):
    """||a - b|| <= rel * ||b||: robust to rounding in individual small entries."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert np.linalg.norm(a - b) <= rel * max(np.linalg.norm(b), 1e-12)


def _x(n=16, shape=(3, 32, 32), seed=1, dtype="float32"):
    return np.random.default_rng(seed).random((n, *shape), dtype=np.float32).astype(dtype)


def _assert_close(a, b, rtol=1e-4, atol=1e-4):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    scale = max(1.0, float(np.abs(a).max()))
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol * scale)


class TestStructure:
    def test_every_tensor_maps_to_one_weight(self, pair):
        torch_mod, tf_mod, classes = pair
        net, keras = torch_mod.config(), tf_mod.config()
        mapping = tf_mod.weight_map(keras)
        torch_keys = {k for k in net.state_dict() if not k.endswith("num_batches_tracked")}
        assert set(mapping) == torch_keys
        targets = [(layer, index) for layer, index, _ in mapping.values()]
        assert len(targets) == len(set(targets))
        assert len(targets) == len(keras.weights)
        for key, (layer, index, perm) in mapping.items():
            shape = tuple(net.state_dict()[key].shape)
            expected = shape if perm is None else tuple(shape[p] for p in perm)
            assert tuple(keras.get_layer(layer).weights[index].shape) == expected, key

    def test_parameter_counts(self, pair):
        torch_mod, tf_mod, classes = pair
        net, keras = torch_mod.config(), tf_mod.config()
        trainable = sum(p.numel() for p in net.parameters())
        assert trainable == int(sum(np.prod(w.shape) for w in keras.trainable_weights))
        buffers = sum(b.numel() for n, b in net.named_buffers() if not n.endswith("num_batches_tracked"))
        assert buffers == int(sum(np.prod(w.shape) for w in keras.non_trainable_weights))
        assert keras.output_shape == (None, classes)

    def test_batchnorm_hyperparameters(self, pair):
        torch_mod, tf_mod, _ = pair
        torch_bn = [m for m in torch_mod.config().modules() if isinstance(m, torch.nn.BatchNorm2d)]
        keras_bn = [l for l in tf_mod.config().layers if isinstance(l, tf.keras.layers.BatchNormalization)]
        assert len(torch_bn) == len(keras_bn)
        for t, k in zip(torch_bn, keras_bn):
            assert k.epsilon == pytest.approx(t.eps)
            assert k.momentum == pytest.approx(1.0 - t.momentum)
            assert k.center and k.scale and t.affine and t.track_running_stats


class TestForward:
    def test_eval_logits(self, pair):
        net, keras, _ = _pair_models(pair)
        x = _x()
        with torch.no_grad():
            expected = net.eval()(torch.from_numpy(x)).numpy()
        _assert_close(keras(x, training=False).numpy(), expected)
        # Batch-size independent, and predict() agrees.
        _assert_close(keras.predict(x[:3], verbose=0), expected[:3])

    @pytest.mark.parametrize("shape", [(3, 33, 33), (3, 40, 32), (3, 64, 64)])
    def test_other_input_sizes(self, pair, shape):
        """Odd sizes check the symmetric padding; non-square checks the width-kernel pooling."""
        net, keras, _ = _pair_models(pair, input_shape=shape)
        x = _x(4, shape)
        with torch.no_grad():
            expected = net.eval()(torch.from_numpy(x)).numpy()
        _assert_close(keras(x, training=False).numpy(), expected)

    def test_unsupported_shape_rejected_like_pytorch(self, pair):
        torch_mod, tf_mod, classes = pair
        with pytest.raises(Exception):
            torch_mod.config().eval()(torch.zeros(1, 3, 64, 32))
        with pytest.raises(ValueError):
            tf_mod.resnet(num_classes=classes, input_shape=(3, 64, 32))

    def test_train_mode_logits(self, pair):
        net, keras, _ = _pair_models(pair)
        x = _x(32)
        expected = net.train()(torch.from_numpy(x)).detach().numpy()
        _assert_close(keras(x, training=True).numpy(), expected, rtol=1e-3, atol=1e-4)


def _torch_loss_grads(net, x, y):
    net.train()
    net.zero_grad()
    loss = torch.nn.functional.cross_entropy(net(torch.from_numpy(x)), torch.from_numpy(y))
    loss.backward()
    return float(loss), {n: p.grad.numpy().copy() for n, p in net.named_parameters()}


def _keras_loss_grads(keras, x, y):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    with tf.GradientTape() as tape:
        loss = loss_fn(y, keras(x, training=True))
    grads = tape.gradient(loss, keras.trainable_weights)
    return float(loss), dict(zip([w.name for w in keras.trainable_weights], grads))


class TestTraining:
    @staticmethod
    def _grad_pairs(pair, dtype):
        net, keras, tf_mod = _pair_models(pair, dtype=dtype)
        classes = pair[2]
        x = _x(32, dtype=dtype)
        y = np.random.default_rng(2).integers(0, classes, 32).astype(np.int64)
        t_loss, t_grads = _torch_loss_grads(net, x, y)
        k_loss, k_grads = _keras_loss_grads(keras, x, y)
        mapping = tf_mod.weight_map(keras)
        pairs = []
        for key, grad in t_grads.items():
            layer, index, perm = mapping[key]
            weight = keras.get_layer(layer).weights[index]
            pairs.append((key, k_grads[weight.name].numpy(), grad if perm is None else np.transpose(grad, perm)))
        assert len(pairs) == len(keras.trainable_weights)
        return t_loss, k_loss, pairs

    def test_gradients_float64_identical(self, pair):
        t_loss, k_loss, pairs = self._grad_pairs(pair, "float64")
        assert k_loss == pytest.approx(t_loss, rel=1e-12)
        for key, k_grad, t_grad in pairs:
            _assert_close(k_grad, t_grad, rtol=1e-9, atol=1e-11)

    def test_gradients_float32(self, pair):
        t_loss, k_loss, pairs = self._grad_pairs(pair, "float32")
        assert k_loss == pytest.approx(t_loss, rel=1e-5)
        for key, k_grad, t_grad in pairs:
            _assert_norm_close(k_grad, t_grad, rel=2e-2)

    def test_eval_logits_float64_identical(self, pair):
        net, keras, _ = _pair_models(pair, dtype="float64")
        x = _x(dtype="float64")
        with torch.no_grad():
            expected = net.eval()(torch.from_numpy(x)).numpy()
        _assert_close(keras(x, training=False).numpy(), expected, rtol=1e-11, atol=1e-12)

    def test_running_stats_after_training_step(self, pair):
        net, keras, tf_mod = _pair_models(pair)
        x = _x(32)
        net.train()(torch.from_numpy(x))
        keras(x, training=True)
        updated = tf_mod.to_torch_state_dict(keras)
        for key, value in net.state_dict().items():
            if key.endswith("running_mean"):
                _assert_close(updated[key], value.numpy(), rtol=1e-4, atol=1e-5)
            elif key.endswith("running_var"):
                # PyTorch uses the unbiased batch variance (n / (n - 1)), Keras the biased one.
                _assert_close(updated[key], value.numpy(), rtol=2e-3, atol=1e-5)

    def test_sgd_step_float64(self, pair):
        net, keras, tf_mod = _pair_models(pair, dtype="float64")
        classes = pair[2]
        x = _x(32, dtype="float64")
        y = np.random.default_rng(3).integers(0, classes, 32).astype(np.int64)
        # Keras keeps the learning rate in float32; 0.0625 is exact there (0.05 is not).
        lr = 0.0625
        _, t_grads = _torch_loss_grads(net, x, y)
        with torch.no_grad():
            for name, p in net.named_parameters():
                p -= lr * p.grad
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        with tf.GradientTape() as tape:
            loss = loss_fn(y, keras(x, training=True))
        optimizer.apply_gradients(zip(tape.gradient(loss, keras.trainable_weights), keras.trainable_weights))
        updated = tf_mod.to_torch_state_dict(keras)
        for name, p in net.named_parameters():
            _assert_close(updated[name], p.detach().numpy(), rtol=1e-10, atol=1e-12)


class TestRoundTrip:
    def test_keras_to_torch_strict(self, pair):
        torch_mod, tf_mod, classes = pair
        keras = tf_mod.config()
        x = _x(8)
        keras(x, training=True)  # move BN statistics off their initial values
        state = {k: torch.from_numpy(v) for k, v in tf_mod.to_torch_state_dict(keras).items()}
        net = torch_mod.config()
        for key, value in net.state_dict().items():
            if key.endswith("num_batches_tracked"):
                state[key] = value
        net.load_state_dict(state, strict=True)
        with torch.no_grad():
            expected = net.eval()(torch.from_numpy(x)).numpy()
        _assert_close(keras(x, training=False).numpy(), expected)

    def test_save_and_load_keras_file(self, pair, tmp_path):
        net, keras, _ = _pair_models(pair)
        path = tmp_path / "model.keras"
        keras.save(path)
        loaded = tf.keras.models.load_model(path, compile=False)
        x = _x(4)
        _assert_close(loaded(x, training=False).numpy(), keras(x, training=False).numpy(), rtol=0, atol=0)


class TestInitialisation:
    def test_kernel_and_bias_distributions(self, pair):
        _, tf_mod, classes = pair
        keras = tf_mod.config()
        for layer in keras.layers:
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
                kernel = layer.get_weights()[0]
                fan_in = int(np.prod(kernel.shape[:-1]))
                expected_std = (2.0 / fan_in) ** 0.5
                # Untruncated: values beyond 2 std exist (truncated he_normal has none).
                if kernel.size >= 2000:
                    assert kernel.std() == pytest.approx(expected_std, rel=0.1), layer.name
                    assert np.abs(kernel).max() > 2.2 * expected_std, layer.name
        bias = keras.get_layer("linear").get_weights()[1]
        assert np.all(np.abs(bias) <= 1 / 8) and bias.shape == (classes,)
        for layer in keras.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                gamma, beta, mean, var = layer.get_weights()
                assert np.all(gamma == 1) and np.all(beta == 0) and np.all(mean == 0) and np.all(var == 1)
