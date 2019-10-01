# https://www.tensorflow.org/guide/saved_model

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

## Creating a SavedModel from Keras
file = tf.keras.utils.get_file(
    "grace_hopper.jpg",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")
img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])
plt.imshow(img)
plt.axis('off')
plt.show()

x = tf.keras.preprocessing.image.img_to_array(img)
x = tf.keras.applications.mobilenet.preprocess_input(x[tf.newaxis,...])

## pre-trained model, save
labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

pretrained_model = tf.keras.applications.MobileNet()
result_before_save = pretrained_model(x)
print()

decoded = imagenet_labels[np.argsort(result_before_save)[0,::-1][:5]+1]
print("Result before saving:\n", decoded)

tf.saved_model.save(pretrained_model, "./mobilenet/1/")

## Load SavedModel
loaded = tf.saved_model.load("./mobilenet/1/")
print(list(loaded.signatures.keys()))  # ["serving_default"]

infer = loaded.signatures["serving_default"]
print(infer.structured_outputs)

labeling = infer(tf.constant(x))[pretrained_model.output_names[0]]
decoded = imagenet_labels[np.argsort(labeling)[0,::-1][:5]+1]
print("Result after saving and loading:\n", decoded)

## Exporting custom models
class CustomModule(tf.Module):

    def __init__(self):
        super(CustomModule, self).__init__()
        self.v = tf.Variable(1.)

    @tf.function
    def __call__(self, x):
        return x * self.v

    @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
    def mutate(self, new_v):
        self.v.assign(new_v)

module = CustomModule()

module(tf.constant(0.))
tf.saved_model.save(module, "./module_no_signatures/")

imported = tf.saved_model.load("./module_no_signatures/")
assert 3. == imported(tf.constant(3.)).numpy()
imported.mutate(tf.constant(2.))
assert 6. == imported(tf.constant(3.)).numpy()
module.__call__.get_concrete_function(x=tf.TensorSpec([None], tf.float32))
tf.saved_model.save(module, "./module_no_signatures/")
imported = tf.saved_model.load("./module_no_signatures/")
assert [3.] == imported(tf.constant([3.])).numpy()


## Reusing SavedModels in Python
# Basic fine tuning
optimizer = tf.optimizers.SGD(0.05)

def train_step():
    with tf.GradientTape() as tape:
        loss = (10. - imported(tf.constant(2.))) ** 2
    variables = tape.watched_variables()
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))
    return loss

for _ in range(10):
    # "v" approaches 5, "loss" approaches 0
    print("loss={:.2f} v={:.2f}".format(train_step(), imported.v.numpy()))

# General fine-tuning
loaded = tf.saved_model.load("./mobilenet/1/")
print("MobileNet has {} trainable variables: {}, ...".format(
            len(loaded.trainable_variables),
            ", ".join([v.name for v in loaded.trainable_variables[:5]])))

trainable_variable_ids = {id(v) for v in loaded.trainable_variables}
non_trainable_variables = [v for v in loaded.variables
                            if id(v) not in trainable_variable_ids]
print("MobileNet also has {} non-trainable variables: {}, ...".format(
            len(non_trainable_variables),
            ", ".join([v.name for v in non_trainable_variables[:3]])))

## Control flow in SavedModels
@tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
def control_flow(x):
    if x < 0:
        tf.print("Invalid!")
    else:
        tf.print(x % 3)

to_export = tf.Module()
to_export.control_flow = control_flow
tf.saved_model.save(to_export, "./control_flow/")

imported = tf.saved_model.load("./control_flow")
imported.control_flow(tf.constant(-1))  # Invalid!
imported.control_flow(tf.constant(2))   # 2
imported.control_flow(tf.constant(3))   # 0