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
from official.vision.image_classification import vgg_model


def dataset_fn(_):
  is_training = True
  data_dir = '/home/cc/datasets/imagenet/tf_records/train/'
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

def run_(flags_obj):
  """Run ResNet ImageNet training and eval loop using native Keras APIs.

  Args:
    flags_obj: An object containing parsed flag values.

  Raises:
    ValueError: If fp16 is passed as it is not currently supported.

  Returns:
    Dictionary of training and eval stats.
  """
  keras_utils.set_session_config(
      enable_eager=flags_obj.enable_eager,
      enable_xla=flags_obj.enable_xla)

  # Execute flag override logic for better model performance
  if flags_obj.tf_gpu_thread_mode:
    keras_utils.set_gpu_thread_mode_and_count(
        per_gpu_thread_count=flags_obj.per_gpu_thread_count,
        gpu_thread_mode=flags_obj.tf_gpu_thread_mode,
        num_gpus=flags_obj.num_gpus,
        datasets_num_private_threads=flags_obj.datasets_num_private_threads)
  common.set_cudnn_batchnorm_mode()

  dtype = flags_core.get_tf_dtype(flags_obj)
  if dtype == tf.float16:
    loss_scale = flags_core.get_loss_scale(flags_obj, default_for_fp16=128)
    policy = tf.compat.v2.keras.mixed_precision.experimental.Policy(
        'mixed_float16', loss_scale=loss_scale)
    tf.compat.v2.keras.mixed_precision.experimental.set_policy(policy)
    if not keras_utils.is_v2_0():
      raise ValueError('--dtype=fp16 is not supported in TensorFlow 1.')
  elif dtype == tf.bfloat16:
    policy = tf.compat.v2.keras.mixed_precision.experimental.Policy(
        'mixed_bfloat16')
    tf.compat.v2.keras.mixed_precision.experimental.set_policy(policy)

  data_format = flags_obj.data_format
  if data_format is None:
    data_format = ('channels_first'
                   if tf.test.is_built_with_cuda() else 'channels_last')
  tf.keras.backend.set_image_data_format(data_format)

  # Configures cluster spec for distribution strategy.
  num_workers = distribution_utils.configure_cluster(flags_obj.worker_hosts,
                                                     flags_obj.task_index)

  strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=flags_obj.distribution_strategy,
      num_gpus=flags_obj.num_gpus,
      num_workers=num_workers,
      all_reduce_alg=flags_obj.all_reduce_alg,
      num_packs=flags_obj.num_packs,
      tpu_address=flags_obj.tpu)

  if strategy:
    # flags_obj.enable_get_next_as_optional controls whether enabling
    # get_next_as_optional behavior in DistributedIterator. If true, last
    # partial batch can be supported.
    strategy.extended.experimental_enable_get_next_as_optional = (
        flags_obj.enable_get_next_as_optional
    )

  strategy_scope = distribution_utils.get_strategy_scope(strategy)

  # pylint: disable=protected-access
  if flags_obj.use_synthetic_data:
    distribution_utils.set_up_synthetic_data()
    input_fn = common.get_synth_input_fn(
        height=imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
        width=imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
        num_channels=imagenet_preprocessing.NUM_CHANNELS,
        num_classes=imagenet_preprocessing.NUM_CLASSES,
        dtype=dtype,
        drop_remainder=True)
  else:
    distribution_utils.undo_set_up_synthetic_data()
    input_fn = imagenet_preprocessing.input_fn

  # When `enable_xla` is True, we always drop the remainder of the batches
  # in the dataset, as XLA-GPU doesn't support dynamic shapes.
  drop_remainder = flags_obj.enable_xla

  train_input_dataset = input_fn(
      is_training=True,
      data_dir=flags_obj.data_dir,
      batch_size=flags_obj.batch_size,
      num_epochs=flags_obj.train_epochs,
      parse_record_fn=imagenet_preprocessing.parse_record,
      datasets_num_private_threads=flags_obj.datasets_num_private_threads,
      dtype=dtype,
      drop_remainder=drop_remainder,
      tf_data_experimental_slack=flags_obj.tf_data_experimental_slack,
      training_dataset_cache=flags_obj.training_dataset_cache,
  )

  eval_input_dataset = None
  if not flags_obj.skip_eval:
    eval_input_dataset = input_fn(
        is_training=False,
        data_dir=flags_obj.data_dir,
        batch_size=flags_obj.batch_size,
        num_epochs=flags_obj.train_epochs,
        parse_record_fn=imagenet_preprocessing.parse_record,
        dtype=dtype,
        drop_remainder=drop_remainder)

  lr_schedule = 0.1
  if flags_obj.use_tensor_lr:
    lr_schedule = common.PiecewiseConstantDecayWithWarmup(
        batch_size=flags_obj.batch_size,
        epoch_size=imagenet_preprocessing.NUM_IMAGES['train'],
        warmup_epochs=common.LR_SCHEDULE[0][1],
        boundaries=list(p[1] for p in common.LR_SCHEDULE[1:]),
        multipliers=list(p[0] for p in common.LR_SCHEDULE),
        compute_lr_on_cpu=True)

  with strategy_scope:
    optimizer = common.get_optimizer(lr_schedule)
    if flags_obj.fp16_implementation == 'graph_rewrite':
      # Note: when flags_obj.fp16_implementation == "graph_rewrite", dtype as
      # determined by flags_core.get_tf_dtype(flags_obj) would be 'float32'
      # which will ensure tf.compat.v2.keras.mixed_precision and
      # tf.train.experimental.enable_mixed_precision_graph_rewrite do not double
      # up.
      optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
          optimizer)

    # TODO(hongkuny): Remove trivial model usage and move it to benchmark.
    if flags_obj.use_trivial_model:
      model = trivial_model.trivial_model(
          imagenet_preprocessing.NUM_CLASSES)
    else:
      model = alexnet_model.alexnet()

    # TODO(b/138957587): Remove when force_v2_in_keras_compile is on longer
    # a valid arg for this model. Also remove as a valid flag.
    if flags_obj.force_v2_in_keras_compile is not None:
      model.compile(
          loss='sparse_categorical_crossentropy',
          optimizer=optimizer,
          metrics=(['sparse_categorical_accuracy']
                   if flags_obj.report_accuracy_metrics else None),
          run_eagerly=flags_obj.run_eagerly,
          experimental_run_tf_function=flags_obj.force_v2_in_keras_compile)
    else:
      model.compile(
          loss='sparse_categorical_crossentropy',
          optimizer=optimizer,
          metrics=(['sparse_categorical_accuracy']
                   if flags_obj.report_accuracy_metrics else None),
          run_eagerly=flags_obj.run_eagerly)

  steps_per_epoch = (
      imagenet_preprocessing.NUM_IMAGES['train'] // flags_obj.batch_size)
  train_epochs = flags_obj.train_epochs

  callbacks = common.get_callbacks(steps_per_epoch,
                                   common.learning_rate_schedule)
  if flags_obj.enable_checkpoint_and_export:
    ckpt_full_path = os.path.join(flags_obj.model_dir, 'model.ckpt-{epoch:04d}')
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(ckpt_full_path,
                                                        save_weights_only=True))

  # if mutliple epochs, ignore the train_steps flag.
  if train_epochs <= 1 and flags_obj.train_steps:
    steps_per_epoch = min(flags_obj.train_steps, steps_per_epoch)
    train_epochs = 1

  num_eval_steps = (
      imagenet_preprocessing.NUM_IMAGES['validation'] // flags_obj.batch_size)

  validation_data = eval_input_dataset
  if flags_obj.skip_eval:
    # Only build the training graph. This reduces memory usage introduced by
    # control flow ops in layers that have different implementations for
    # training and inference (e.g., batch norm).
    if flags_obj.set_learning_phase_to_train:
      # TODO(haoyuzhang): Understand slowdown of setting learning phase when
      # not using distribution strategy.
      tf.keras.backend.set_learning_phase(1)
    num_eval_steps = None
    validation_data = None

  if not strategy and flags_obj.explicit_gpu_placement:
    # TODO(b/135607227): Add device scope automatically in Keras training loop
    # when not using distribition strategy.
    no_dist_strat_device = tf.device('/device:GPU:0')
    no_dist_strat_device.__enter__()

  

  # input_options = tf.distribute.InputOptions(
  #   experimental_fetch_to_device=False,
  #   experimental_per_replica_buffer_size=1)

  history = model.fit(input_fn,
                      epochs=train_epochs,
                      steps_per_epoch=steps_per_epoch,
                      callbacks=callbacks,
                      validation_steps=num_eval_steps,
                      validation_data=validation_data,
                      validation_freq=flags_obj.epochs_between_evals,
                      verbose=2)
  if flags_obj.enable_checkpoint_and_export:
    if dtype == tf.bfloat16:
      logging.warning("Keras model.save does not support bfloat16 dtype.")
    else:
      # Keras model.save assumes a float32 input designature.
      export_path = os.path.join(flags_obj.model_dir, 'saved_model')
      model.save(export_path, include_optimizer=False)

  eval_output = None
  if not flags_obj.skip_eval:
    eval_output = model.evaluate(eval_input_dataset,
                                 steps=num_eval_steps,
                                 verbose=2)

  if not strategy and flags_obj.explicit_gpu_placement:
    no_dist_strat_device.__exit__()

  stats = common.build_stats(history, eval_output, callbacks)
  return stats

def run(flags_obj):
  os.environ["TF_CONFIG"] = json.dumps({
      "cluster": {
            "worker": ["10.31.0.72:6433", "10.31.0.74:6434"],
            "ps": ["10.31.0.73:6435"],
            "chief": ["10.31.0.76:6436"]
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
    model = vgg_model.vgg16(num_classes=1000)
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
