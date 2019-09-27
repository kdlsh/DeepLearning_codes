# https://www.tensorflow.org/beta/tutorials/eager/custom_layers

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

# 대부분의 layer는 첫번째 인수로 출력 차원(크기) 또는 채널을 취합니다.
layer = tf.keras.layers.Dense(100)
# 일부 복잡한 모델에서는 수동으로 입력 차원의 수를 제공하는것이 유용할 수 있습니다.
layer = tf.keras.layers.Dense(10, input_shape=(None, 5))
# Dense(완전 연결 층), Conv2D, LSTM, BatchNormalization, Dropout, 등
# https://www.tensorflow.org/api_docs/python/tf/keras/layers

# 층을 사용하려면, 간단하게 호출합니다.
layer(tf.zeros([10, 5]))

print(layer.variables)
print(layer.trainable_variables)
print(layer.kernel)
print(layer.bias)

## Custum layer
# __init__ 에서 층에 필요한 매개변수를 입력 받습니다
# build, 입력 텐서의 크기를 얻고 남은 초기화를 진행할 수 있습니다
# call, 정방향 연산(forward computation)
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel",
                                        shape=[int(input_shape[-1]),
                                            self.num_outputs])

    def call(self, input):
        return tf.matmul(input, self.kernel)

layer = MyDenseLayer(10)
print(layer(tf.zeros([10, 5])))
print(layer.trainable_variables)

## Model: layer
class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)

block = ResnetIdentityBlock(1, [1, 2, 3])
print(block(tf.zeros([1, 2, 3, 3])))
print([x.name for x in block.trainable_variables])

## Simple Sequential
my_seq = tf.keras.Sequential([tf.keras.layers.Conv2D(1, (1, 1),
                                                    input_shape=(
                                                        None, None, 3)),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Conv2D(2, 1,
                                                    padding='same'),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Conv2D(3, (1, 1)),
                             tf.keras.layers.BatchNormalization()])
my_seq(tf.zeros([1, 2, 3, 3]))