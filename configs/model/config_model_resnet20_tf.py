import tensorflow as tf
from tensorflow.keras import layers

# Identity layer for shortcut (fixes empty Sequential issue)
class Identity(layers.Layer):
    def call(self, x, training=False):
        return x


# Basic building block for ResNet
class BasicBlock(tf.keras.Model):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
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
            self.shortcut = Identity()  # FIX: replace empty Sequential

    def call(self, x, training=False):
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = tf.nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        out += self.shortcut(x, training=training)
        out = tf.nn.relu(out)
        return out


# ResNet model for CIFAR-10
class ResNet(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
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


# Model configuration functions
def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])

def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])

def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])

def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])

def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])

def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])

def config():
    return resnet20()
