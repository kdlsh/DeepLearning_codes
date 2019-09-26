#https://tensorflow.blog/2019/03/06/tensorflow-2-0-keras-api-overview/

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
 
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

class MyModel(tf.keras.Model):
 
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes        
        self.dense_1 = layers.Dense(32, activation='relu')
        self.dense_2 = layers.Dense(num_classes, activation='sigmoid')
 
    def call(self, inputs):        
        x = self.dense_1(inputs)
        return self.dense_2(x)

model = MyModel(num_classes=10)
 
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
loss='categorical_crossentropy',
metrics=['accuracy'])
 
model.fit(data, labels, batch_size=32, epochs=5)