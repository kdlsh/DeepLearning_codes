#https://tensorflow.blog/2019/03/06/tensorflow-2-0-keras-api-overview/

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
 
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

class MyLayer(layers.Layer):
 
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # weight variable
        self.kernel = self.add_weight(name='kernel',
            shape=(input_shape[1], self.output_dim),
            initializer='uniform',
            trainable=True)
    
    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)
    
    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

model = tf.keras.Sequential([
        MyLayer(10),
        layers.Activation('softmax')])
 
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
 
model.fit(data, labels, batch_size=32, epochs=5)

