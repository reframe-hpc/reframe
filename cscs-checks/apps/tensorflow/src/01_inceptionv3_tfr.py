#!/usr/bin/env python3
# coding: utf-8
# from https://github.com/eth-cscs/tensorflow-training/tree/master/imagenet/

import os
import sys
import glob
import tensorflow as tf
from datetime import datetime


image_shape = (224, 224)
batch_size = int(sys.argv[1])


def decode(serialized_example):
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/class/label': tf.io.FixedLenFeature([], tf.int64),
        })
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    image = tf.image.resize(image, image_shape, method='bicubic')
    label = tf.cast(features['image/class/label'], tf.int64) - 1  # [0-999]
    return image, label


list_of_files = glob.glob(sys.argv[2])
print(f"# batch_size={batch_size} {type(batch_size)}")
# print(f"# list_of_files={list_of_files} {type(list_of_files)}")
AUTO = tf.data.experimental.AUTOTUNE
dataset = (tf.data.TFRecordDataset(list_of_files, num_parallel_reads=AUTO)
           .map(decode, num_parallel_calls=AUTO)
           .batch(batch_size)
           .prefetch(AUTO))
model = tf.keras.applications.InceptionV3(weights=None,
                                          input_shape=(*image_shape, 3),
                                          classes=1000)

optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# model.summary()

tb_callback = \
    tf.keras.callbacks.TensorBoard(log_dir=os.path.join('inceptionv3_logs',
                                   datetime.now().strftime("%d-%H%M")),
                                   histogram_freq=1, profile_batch='80,100')

fit = model.fit(dataset.take(100),
                epochs=1,
                callbacks=[tb_callback])
