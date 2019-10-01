
# https://www.tensorflow.org/beta/guide/using_gpu

import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

import tensorflow as tf
# from keras.backend import tensorflow_backend as K

# ## tf backend config
# gpu_fraction = 0.45
# config = tf.compat.v1.ConfigProto()
# #config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction=gpu_fraction
# K.set_session(tf.compat.v1.Session(config=config))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    #tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)