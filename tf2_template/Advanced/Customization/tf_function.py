# https://www.tensorflow.org/beta/tutorials/eager/tf_function

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import contextlib

# Some helper code to demonstrate the kinds of errors you might encounter.
@contextlib.contextmanager
def assert_raises(error_class):
    try:
        yield
    except error_class as e:
        print('Caught expected exception \n  {}: {}'.format(error_class, e))
    except Exception as e:
        print('Got unexpected exception \n  {}: {}'.format(type(e), e))
    else:
        raise Exception('Expected {} to be raised but no error was raised!'.format(
            error_class))

# A function is like an op
@tf.function
def add(a, b):
    return a + b

print(add(tf.ones([2, 2]), tf.ones([2, 2])))  #  [[2., 2.], [2., 2.]]

# Functions have gradients
v = tf.Variable(1.0)
with tf.GradientTape() as tape:
    result = add(v, 1.0)
print(tape.gradient(result, v))

# You can use functions inside functions
@tf.function
def dense_layer(x, w, b):
    return add(tf.matmul(x, w), b)

print(dense_layer(tf.ones([3, 2]), tf.ones([2, 2]), tf.ones([2])))

## Tracing and polymorphism
@tf.function
def double(a):
    print("Tracing with", a)
    return a + a

print(double(tf.constant(1)))
print()
print(double(tf.constant(1.1)))
print()
print(double(tf.constant("a")))
print()

print("Obtaining concrete trace")
double_strings = double.get_concrete_function(tf.TensorSpec(shape=None, dtype=tf.string))
print("Executing traced function")
print(double_strings(tf.constant("a")))
print(double_strings(a=tf.constant("b")))
print("Using a concrete trace with incompatible types will throw an error")
with assert_raises(tf.errors.InvalidArgumentError):
    double_strings(tf.constant(1))

@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
def next_collatz(x):
    print("Tracing with", x)
    return tf.where(x % 2 == 0, x // 2, 3 * x + 1)

print(next_collatz(tf.constant([1, 2])))
# We specified a 1-D tensor in the input signature, so this should fail.
with assert_raises(ValueError):
    next_collatz(tf.constant([[1, 2], [3, 4]]))