# https://www.tensorflow.org/guide/data_performance

import tensorflow as tf

def parse_fn(example):
    "Parse TFExample records and perform simple data augmentation."
    example_fmt = {
        "image": tf.io.FixedLenFeature((), tf.string, ""),
        "label": tf.io.FixedLenFeature((), tf.int64, -1)
    }
    parsed = tf.io.parse_single_example(example, example_fmt)
    image = tf.io.decode_image(parsed["image"])
    image = _augment_helper(image)  # augments image using slice, reshape, resize_bilinear
    return image, parsed["label"]

def make_dataset():
    dataset = tf.data.TFRecordDataset("/path/to/dataset/train-*.tfrecord")
    dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size)
    dataset = dataset.map(map_func=parse_fn)
    dataset = dataset.batch(batch_size=FLAGS.batch_size)
    return dataset

## Pipelining
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

## Parallelize data transformation
#dataset = dataset.map(map_func=parse_fn)
dataset = dataset.map(map_func=parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

## Parallelize data extraction
#dataset = tf.data.TFRecordDataset("/path/to/dataset/train-*.tfrecord")
files = tf.data.Dataset.list_files("/path/to/dataset/train-*.tfrecord")
dataset = files.interleave(
    tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_reads,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Use the prefetch transformation to overlap the work of a producer and consumer. 
# In particular, we recommend adding prefetch to the end of your input pipeline to overlap 
# the transformations performed on the CPU with the training done on the accelerator. 
# Either manually tuning the buffer size, or using tf.data.experimental.AUTOTUNE to delegate 
# the decision to the tf.data runtime.

# Parallelize the map transformation by setting the num_parallel_calls argument.
# Either manually tuning the level of parallelism, or using tf.data.experimental.AUTOTUNE 
# to delegate the decision to the tf.data runtime.

# If you are working with data stored remotely and/or requiring deserialization, 
# we recommend using the interleave transformation to parallelize the reading 
# (and deserialization) of data from different files.

# Vectorize cheap user-defined functions passed in to the map transformation 
# to amortize the overhead associated with scheduling and executing the function.

# If your data can fit into memory, use the cache transformation to cache it in memory 
# during the first epoch, so that subsequent epochs can avoid the overhead associated 
# with reading, parsing, and transforming it.

# If your pre-processing increases the size of your data, we recommend applying 
# the interleave, prefetch, and shuffle first (if possible) to reduce memory usage.

# We recommend applying the shuffle transformation before the repeat transformation.

