# https://www.tensorflow.org/tutorials/eager/eager_basics

from __future__ import absolute_import, division, print_function
import tensorflow as tf

tf.executing_eagerly()

print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))
print(tf.encode_base64("hello world"))

# 연산자의 오버로딩(overloding) 또한 지원합니다.
print(tf.square(2) + tf.square(3))

