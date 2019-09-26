#python version check

import os
import tensorflow as tf
import keras as kr
import tensorflow_hub as hub
import tensorflow_datasets as tfds

os.chdir(os.path.dirname(os.path.realpath(__file__)))

cuDNN_lib="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0\\include\\cudnn.h"

tf_ver = tf.__version__
tf_keras_ver = tf.keras.__version__
keras_ver = kr.__version__
hub_ver = hub.__version__

popen_out=os.popen("nvcc --version").read()
cuda_ver = popen_out.split('\n')[-2].split(',')[-1].lstrip()

ver_li = ['define CUDNN_MAJOR', 'define CUDNN_MINOR', 'define CUDNN_PATCHLEVEL']
cudnn_ver = '.'.join([line.split()[-1] for ver in ver_li for line in open(cuDNN_lib) if ver in line])

print("GPU ", "사용 가능" if tf.test.is_gpu_available() else "사용 불가능")
print_li = ['TensorFlow: '+tf_ver, 'TF_Keras: '+tf_keras_ver, 'TF_hub: '+hub_ver,
         'Keras: '+keras_ver, 'CUDA: '+cuda_ver, 'cuDNN: '+cudnn_ver]
for i in print_li: print(i)

#tensorflow2 linux docker recommend (linux)