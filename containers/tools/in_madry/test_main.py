"""Checks that main.py keeps the semantics of the official TF1 code (MadryLab/cifar10_challenge@f15682d).

Needs TensorFlow, so it runs inside the tool image, not in the repository's pytest suite:
    docker run --rm -v $PWD/containers/tools/in_madry:/app -w /app <in_madry image> python -m pytest -q test_main.py
(the TF image has no pytest; `pip install pytest` first, or run `python test_main.py`).
"""

import numpy as np
import tensorflow as tf

import main


def small_model(num_classes=3, bn=True):
    inputs = tf.keras.Input((3, 8, 8))
    x = tf.keras.layers.Permute((2, 3, 1))(inputs)
    x = tf.keras.layers.Conv2D(4, 3, padding="same", use_bias=False)(x)
    if bn:
        x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inputs, tf.keras.layers.Dense(num_classes)(x))


def reference_perturb(model, x_nat, y, epsilon, num_steps, step_size, x_start, loss_func="xent"):
    """LinfPGDAttack.perturb from pgd_attack.py with the TF1 gradient replaced by a GradientTape."""
    x = x_start.copy()
    for _ in range(num_steps):
        xt = tf.constant(x)
        with tf.GradientTape() as tape:
            tape.watch(xt)
            logits = model(xt, training=False)
            if loss_func == "xent":
                loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
            else:
                mask = tf.one_hot(y, logits.shape[1])
                correct = tf.reduce_sum(mask * logits, axis=1)
                wrong = tf.reduce_max((1 - mask) * logits - 1e4 * mask, axis=1)
                loss = -tf.nn.relu(correct - wrong + 50)
        grad = tape.gradient(loss, xt).numpy()
        x = x + step_size * np.sign(grad)
        x = np.clip(x, x_nat - epsilon, x_nat + epsilon)
        x = np.clip(x, 0, 1)
    return x


def test_piecewise_constant_matches_tf1():
    boundaries, values = [40000, 60000], [0.1, 0.01, 0.001]
    step = tf.Variable(0, dtype=tf.int64)
    lr = tf.compat.v1.train.piecewise_constant(step, boundaries, values)
    for s in [0, 1, 39999, 40000, 40001, 59999, 60000, 60001, 79999]:
        step.assign(s)
        expected = float(lr() if callable(lr) else lr)
        got = values[sum(s > b for b in boundaries)]
        assert np.isclose(got, expected), (s, got, expected)


def test_attack_matches_reference_and_stays_in_ball():
    rng = np.random.default_rng(0)
    model = small_model(bn=False)
    x_nat = rng.random((16, 3, 8, 8), dtype=np.float32)
    y = rng.integers(0, 3, 16)
    eps, step = 8 / 255, 2 / 255
    for loss_func in ("xent", "cw"):
        attack = main.make_attack(model, eps, 7, step, False, loss_func, training=False)
        got = attack(x_nat, y)
        ref = reference_perturb(model, x_nat, y, eps, 7, step, x_nat, loss_func)
        assert np.abs(got - ref).max() < 1e-6, loss_func
        assert np.abs(got - x_nat).max() <= eps + 1e-6 and got.min() >= 0 and got.max() <= 1


def test_random_start_is_uniform_in_ball_and_clipped():
    np.random.seed(1)
    model = small_model(bn=False)
    x_nat = np.zeros((64, 3, 8, 8), np.float32)  # clip at 0 must apply
    attack = main.make_attack(model, 8 / 255, 0, 2 / 255, True, "xent", training=False)
    x = attack(x_nat, np.zeros(64, np.int64))
    assert x.min() >= 0 and x.max() <= 8 / 255 + 1e-7 and (x > 0).mean() > 0.4


def test_training_mode_attack_updates_batch_norm_like_tf1():
    """tf.contrib.layers.batch_norm(updates_collections=None, is_training=True) updates the moving
    averages on every forward pass, so each attack step moves them; eval mode must not."""
    model = small_model()
    bn = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.BatchNormalization)][0]
    x = np.random.default_rng(2).random((8, 3, 8, 8), dtype=np.float32)
    y = np.zeros(8, np.int64)
    before = bn.moving_mean.numpy().copy()
    main.make_attack(model, 8 / 255, 3, 2 / 255, False, "xent", training=False)(x, y)
    assert np.array_equal(before, bn.moving_mean.numpy())
    # Step size 0 keeps the input fixed: three updates with decay 0.9 from 0 give mean * (1 - 0.9 ** 3).
    main.make_attack(model, 8 / 255, 3, 0.0, False, "xent", training=True)(x, y)
    conv_out = model.layers[2](model.layers[1](x)).numpy()
    expected = np.mean(conv_out, axis=(0, 1, 2)) * (1 - 0.9 ** 3)
    assert np.allclose(bn.moving_mean.numpy(), expected, atol=1e-5)


def test_momentum_optimizer_is_tf1_momentum():
    """v <- m * v + g; w <- w - lr * v (tf.train.MomentumOptimizer), also across an lr change."""
    w = tf.Variable([1.0, -2.0])
    lr = tf.Variable(0.1)
    opt = tf.compat.v1.train.MomentumOptimizer(lr, 0.9)
    v, w_ref = np.zeros(2), np.array([1.0, -2.0])
    for lr_value, g in [(0.1, [0.5, 1.0]), (0.1, [0.2, -1.0]), (0.01, [1.0, 1.0])]:
        lr.assign(lr_value)
        opt.apply_gradients([(tf.constant(g), w)])
        v = 0.9 * v + np.array(g)
        w_ref = w_ref - lr_value * v
        assert np.allclose(w.numpy(), w_ref, atol=1e-6)


def test_data_subset_matches_official_multiple_passes():
    xs, ys = np.arange(10)[:, None], np.arange(10)
    np.random.seed(3)
    ours = main.DataSubset(xs, ys)
    batches = [ours.get_next_batch(4)[1] for _ in range(5)]
    np.random.seed(3)
    order = np.random.permutation(10)
    expected = [order[0:4], order[4:8]]
    order = np.random.permutation(10)  # 2 left < 4: reshuffle, drop the tail
    expected += [order[0:4], order[4:8]]
    order = np.random.permutation(10)
    expected += [order[0:4]]
    for got, exp in zip(batches, expected):
        assert np.array_equal(got, exp)


def test_augment_is_pad2_crop_flip():
    np.random.seed(4)
    x = np.random.default_rng(5).random((32, 3, 8, 8), dtype=np.float32)
    out = main.augment(x)
    padded = np.pad(x, ((0, 0), (0, 0), (2, 2), (2, 2)))
    for i in range(len(x)):
        crops = [padded[i, :, t:t + 8, l:l + 8] for t in range(5) for l in range(5)]
        assert any(np.array_equal(out[i], c) or np.array_equal(out[i], c[:, :, ::-1]) for c in crops)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
