from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import json

from tensorflow import keras
#import resnet

features = [[1., 1.5], [2., 2.5], [3., 3.5]]
labels = [[0.3], [0.5], [0.7]]
eval_features = [[4., 4.5], [5., 5.5], [6., 6.5]]
eval_labels = [[0.8], [0.9], [1.]]

"""
Remember to set the TF_CONFIG envrionment variable.

For example:

export TF_CONFIG='{"cluster": {"worker": ["10.1.10.58:12345", "10.1.10.250:12345"]}, "task": {"index": 0, "type": "worker"}}'
"""

os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
            "worker": ["10.31.0.20:6433", "10.31.0.19:6434"],
            "ps": ["10.31.0.30:6435"],
            "chief": ["10.31.0.28:6436"]
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

# elif cluster_resolver.task_type == "evaluator":
#     # Run sidecar evaluation
# else:
#     # Run the coordinator.

print("2 --- ")
strategy = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver, 
    variable_partitioner=None)
print("3 --- ")
# coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy)
print("Cluster initialized")

# global_batch_size = 32

# # A dataset function takes a `input_context` and returns a `Dataset`
# def dataset_fn(input_context):
#     x = tf.random.uniform((10, 10))
#     y = tf.random.uniform((10,))

#     dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10).repeat()
#     dataset = dataset.batch(global_batch_size)
#     dataset = dataset.prefetch(2)
#     return dataset

# # With `Model.fit`, a `DatasetCreator` needs to be used.
# input = tf.keras.utils.experimental.DatasetCreator(dataset_fn=dataset_fn)


# # x = tf.random.uniform((10, 10))
# # y = tf.random.uniform((10,))

# # dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10).repeat()
# # dataset = dataset.batch(global_batch_size)
# # dataset = dataset.prefetch(2)

# print("Dataset initialized")

# with strategy.scope():
#     model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])

#     model.compile(tf.keras.optimizers.legacy.SGD(), loss="mse", steps_per_execution=10)

# working_dir = "/tmp/working_dir"
# log_dir = os.path.join(working_dir, "log")
# ckpt_filepath = os.path.join(working_dir, "ckpt")
# backup_dir = os.path.join(working_dir, "backup")

# callbacks = [
#     tf.keras.callbacks.TensorBoard(log_dir=log_dir),
#     tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_filepath),
#     tf.keras.callbacks.BackupAndRestore(backup_dir=backup_dir),
# ]

# print("Training initialized")
# model.fit(input, epochs=5, steps_per_epoch=20, callbacks=callbacks)

dataset = tf.data.Dataset.from_tensor_slices(
      (features, labels)).shuffle(10).repeat().batch(64)

eval_dataset = tf.data.Dataset.from_tensor_slices(
      (eval_features, eval_labels)).repeat().batch(1)

print(type(dataset), dataset)

with strategy.scope():
  model = tf.keras.models.Sequential([tf.keras.layers.Dense(1)])
  optimizer = tf.keras.optimizers.legacy.Adagrad(learning_rate=0.05)
  model.compile(optimizer, "mse")

model.fit(dataset, epochs=5, steps_per_epoch=10)
