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

from typing import Iterator, Generator, NamedTuple

import jax
import jax.numpy as jnp

import jraph

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

from graph_nets import utils_tf

import projects.shortest_path.batching_utils as batching_utils


def create_split(dataset_builder, train: bool, cache=False):
    """Creates a split from the ShortestPath dataset using TensorFlow Datasets.
    Args:
        dataset_builder: TFDS dataset builder for ShortestPath.
        batch_size: the batch size returned by the data pipeline.
        train: Whether to load the train or evaluation split.
        cache: Whether to cache the dataset.
    Returns:
        A `tf.data.Dataset`.
    """
    if train:
        train_examples = dataset_builder.info.splits['train'].num_examples
        split_size = train_examples // jax.process_count()
        start = jax.process_index() * split_size
        split = 'train[{}:{}]'.format(start, start + split_size)
    else:
        validate_examples = dataset_builder.info.splits['test'].num_examples
        split_size = validate_examples // jax.process_count()
        start = jax.process_index() * split_size
        split = 'test[{}:{}]'.format(start, start + split_size)

    def decode_example(example):
        return utils_tf.data_dicts_to_graphs_tuple([example['graph']])

    ds = dataset_builder.as_dataset(split=split)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 48
    ds = ds.with_options(options)

    if cache:
        ds = ds.cache()

    if train:
        ds = ds.repeat()
        ds = ds.shuffle(16 * 8, seed=0)

    ds = ds.map(decode_example,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if not train:
        ds = ds.repeat()

    ds = ds.prefetch(10)

    return ds


class Batch(NamedTuple):
    graph: jraph.GraphsTuple
    distance: jnp.ndarray
    node_labels: jnp.ndarray
    edge_labels: jnp.ndarray


def create_input_iter(dataset_builder: tfds.core.DatasetBuilder,
                      batch_size: int, n_node_per_graph: int,
                      n_edge_per_graph: int, dtype: tf.dtypes.DType,
                      train: bool, cache: bool) -> Iterator:
    """Create iterable dataset."""

    def intermediate_graph_to_batch(graph):
        """NamedTuple to represent batches of data."""
        node_labels = graph.nodes[:, -1]
        edge_labels = graph.edges[:, -1]
        distance = graph.globals

        graph = graph._replace(globals=jnp.zeros_like(graph.globals),
                               nodes=graph.nodes[:, :-1],
                               edges=graph.edges[:, :-1])

        return Batch(graph=graph,
                     distance=distance,
                     node_labels=node_labels,
                     edge_labels=edge_labels)

    del dtype

    np_ds = iter(
        tfds.as_numpy(create_split(dataset_builder, train=train, cache=cache)))
    batch_list = []
    for batch in batching_utils.dynamically_batch(
            np_ds,
            n_node=batch_size * n_node_per_graph,
            n_edge=batch_size * n_edge_per_graph,
            n_graph=batch_size):
        with jax.profiler.StepTraceAnnotation('batch_postprocessing'):
            batch = intermediate_graph_to_batch(batch)
        batch_list.append(batch)
        if len(batch_list) == jax.local_device_count():
            yield jax.device_put_sharded(batch_list, jax.local_devices())
            batch_list = []
