# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from official.benchmark.models import trivial_model
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils
from official.utils.misc import model_helpers
from official.vision.image_classification import common
from official.vision.image_classification import imagenet_preprocessing
from official.vision.image_classification import alexnet_model


def dataset_fn(_):
  is_training = True
  data_dir = '/home/cc/nfs/imagenet/tf_records/train/'
  num_epochs = 5
  batch_size = 512
  dtype = tf.float32
  shuffle_buffer = 100000

  filenames = imagenet_preprocessing.get_shuffled_filenames(is_training, data_dir, num_epochs)
  dataset = tf.data.Dataset.from_tensor_slices(filenames)
  dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=40, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  dataset = dataset.shuffle(shuffle_buffer).repeat()
  dataset = dataset.map(
        lambda value: imagenet_preprocessing.parse_record(value, is_training, dtype),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size, drop_remainder=False)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  # x = tf.random.uniform((10, 10))
  # y = tf.random.uniform((10,))

  # dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10).repeat()
  # dataset = dataset.batch(global_batch_size)
  # dataset = dataset.prefetch(2)
  return dataset

def run(flags_obj):
  os.environ["TF_CONFIG"] = json.dumps({
      "cluster": {
            "worker": ["10.31.0.37:6433", "10.31.0.43:6434"],
            "ps": ["10.31.0.44:6435"],
            "chief": ["10.31.0.41:6436"]
        },
      "task": {"type": "chief", "index": 0}
  })


  print("1 --- ")
  cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
  if cluster_resolver.task_type in ("worker", "ps"):
      # Start a TensorFlow server and wait.
      os.environ["GRPC_FAIL_FAST"] = "use_caller"

      server = tf.distribute.Server(
          cluster_resolver.cluster_spec(),
          job_name=cluster_resolver.task_type,
          task_index=cluster_resolver.task_id,
          protocol=cluster_resolver.rpc_layer or "grpc",
          start=True)
      server.join()

  strategy = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver, 
    variable_partitioner=None)
  print("Cluster initialized")

  lr_schedule = 0.1
  # if flags_obj.use_tensor_lr:
  if True:
    lr_schedule = common.PiecewiseConstantDecayWithWarmup(
        batch_size=flags_obj.batch_size,
        epoch_size=imagenet_preprocessing.NUM_IMAGES['train'],
        warmup_epochs=common.LR_SCHEDULE[0][1],
        boundaries=list(p[1] for p in common.LR_SCHEDULE[1:]),
        multipliers=list(p[0] for p in common.LR_SCHEDULE),
        compute_lr_on_cpu=True)
  
  print(type(lr_schedule),lr_schedule)

  # filenames = imagenet_preprocessing.get_shuffled_filenames(True, flags_obj.data_dir, 2)
  # dataset = tf.data.Dataset.from_tensor_slices(filenames).shuffle(10).repeat().batch(64)

  # if input_context:
  #   logging.info(
  #       'Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d',
  #       input_context.input_pipeline_id, input_context.num_input_pipelines)
  #   dataset = dataset.shard(input_context.num_input_pipelines,
  #                           input_context.input_pipeline_id)
  # dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=10, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  # dataset = dataset.shuffle(buffer_size=2)
  # dataset = dataset.map(
  #       lambda value: imagenet_preprocessing.parse_record(value, True, tf.float32),
  #       num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # dataset = dataset.batch(128)
  # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  # input_options = tf.distribute.InputOptions(
  #   experimental_fetch_to_device=True,
  #   experimental_per_replica_buffer_size=2)

  # print(type(dataset), dataset)

  

  with strategy.scope():
    model = alexnet_model.alexnet()
    # optimizer = common.get_optimizer(lr_schedule)
    optimizer = tf.keras.optimizers.legacy.SGD()
    model.compile(optimizer, loss = "mse")

  steps_per_epoch=imagenet_preprocessing.NUM_IMAGES['train'] // flags_obj.batch_size

  dataset_creator = tf.keras.utils.experimental.DatasetCreator(dataset_fn)

  model.fit(dataset_creator, epochs=flags_obj.train_epochs, steps_per_epoch=steps_per_epoch)

  return 

def define_imagenet_keras_flags():
  common.define_keras_flags()
  flags_core.set_defaults()
  flags.adopt_module_key_flags(common)


def main(_):
  model_helpers.apply_clean(flags.FLAGS)
  with logger.benchmark_context(flags.FLAGS):
    stats = run(flags.FLAGS)
  logging.info('Run stats:\n%s', stats)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  define_imagenet_keras_flags()
  app.run(main)
