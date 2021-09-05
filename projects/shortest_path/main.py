# Copyright 2021 Petrov, Danil <ddbihbka@gmail.com>
# Author: Petrov, Danil <ddbihbka@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main file to run the training."""

import tensorflow as tf
# Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
# it unavailable to JAX.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        tf.config.experimental.set_visible_devices([], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

from absl import app
from absl import flags
from absl import logging

import jax
import clu.platform
import ml_collections.config_flags

import projects.shortest_path.train as train

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
ml_collections.config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    logging.info('JAX process: %d / %d', jax.process_index(),
                 jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    # Add a note so that we can tell which task is which JAX host.
    # (Depending on the platform task 0 is not guaranteed to be host 0)
    clu.platform.work_unit().set_task_status(
        f'process_index: {jax.process_index()}, '
        f'process_count: {jax.process_count()}')
    clu.platform.work_unit().create_artifact(
        clu.platform.ArtifactType.DIRECTORY, FLAGS.workdir, 'workdir')

    train.train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == '__main__':
    flags.mark_flags_as_required(['config', 'workdir'])
    app.run(main)
