# https://www.tensorflow.org/tutorials/load_data/tfrecord

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import numpy as np
import IPython.display as display

## tf.Example
# tf.Example is a {"string": tf.train.Feature} mapping