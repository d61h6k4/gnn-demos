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

import pathlib

import jax
import jax.numpy as jnp
import jraph
import ml_collections.config_flags

import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

import projects.shortest_path.input_pipeline as input_pipeline
import projects.shortest_path.train as train
import visualization.graphs.shortest_path as vis_shortest_path

from graph_nets import utils_np

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
ml_collections.config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)


def gt_to_nx(graph):

    nx_graph = utils_np.graphs_tuple_to_networkxs(graph)[0]

    for node_ix, node_feature in nx_graph.nodes(data=True):
        nx_graph.add_node(node_ix,
                          pos=(node_feature["features"][0],
                               node_feature["features"][1]),
                          weight=node_feature["features"][2],
                          start=bool(node_feature["features"][3]),
                          end=bool(node_feature["features"][4]),
                          solution=bool(node_feature["features"][5]))

    for receiver, sender, edge_feature in nx_graph.edges(data=True):
        if "features" not in edge_feature:
            continue

        nx_graph.add_edge(sender,
                          receiver,
                          distance=edge_feature["features"][0],
                          solution=bool(edge_feature["features"][1]))

    return nx_graph


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    config = FLAGS.config
    rng = jax.random.PRNGKey(0)

    ds = tfds.load(config.dataset, split="test")

    base_learning_rate = config.learning_rate * config.batch_size / 256.
    model = train.GraphNetwork(mlp_features=(16,),
                               latent_size=16,
                               message_passing_num=3)
    learning_rate_fn = train.create_learning_rate_fn(config, base_learning_rate,
                                                     1)

    state = train.create_train_state(rng, config, model, learning_rate_fn)
    state = train.restore_checkpoint(state, FLAGS.workdir)

    graphs = []
    batch = []

    examples_num = 10
    for ex in ds.take(examples_num):
        input_graph = jraph.GraphsTuple(
            **jax.tree_map(lambda x: x._numpy(), ex["graph"]))
        graphs.append(
            gt_to_nx(
                jraph.batch([
                    input_graph._replace(
                        n_node=jnp.asarray([input_graph.n_node]),
                        n_edge=jnp.asarray([input_graph.n_edge]))
                ])))

        input_graph = input_graph._replace(
            nodes=input_graph.nodes[:, :-1],
            edges=input_graph.edges[:, :-1],
            globals=jnp.zeros((1, 1)),
            n_node=jnp.asarray([input_graph.n_node]),
            n_edge=jnp.asarray([input_graph.n_edge]))
        batch.append(input_graph)
        preds = state.apply_fn({'params': state.params},
                               jraph.batch(batch),
                               mutable=False)
        graph = preds._replace(
            nodes=jnp.concatenate((batch[0].nodes,
                                   jnp.reshape(jnp.argmax(preds.nodes, axis=-1),
                                               (-1, 1))),
                                  axis=-1),
            edges=jnp.concatenate((batch[0].edges,
                                   jnp.reshape(jnp.argmax(preds.edges, axis=-1),
                                               (-1, 1))),
                                  axis=-1))
        batch = []
        graphs.append(gt_to_nx(graph))

    fig = vis_shortest_path.show(graphs, rows=examples_num, cols=2)
    fig.savefig(
        pathlib.Path(FLAGS.workdir) / "solution_shortest_path_graphs.png")


if __name__ == '__main__':
    flags.mark_flags_as_required(['config', 'workdir'])
    app.run(main)
