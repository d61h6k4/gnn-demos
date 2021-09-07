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
"""Datasets."""

from typing import Tuple, Dict

import collections
import itertools

import numpy as np
import networkx as nx

from scipy import spatial
from graph_nets import utils_np

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """\
The dataset to learn shortest path in graph.
We follow the shortest path demo from
https://github.com/deepmind/graph_nets
"""

DISTANCE_WEIGHT_NAME = "distance"    # The name for the distance edge attribute.


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def set_diff(seq0, seq1):
    """Return the set difference between 2 sequences as a list."""
    return list(set(seq0) - set(seq1))


def generate_graph(rand: np.random.Generator,
                   min_num_nodes: int,
                   max_num_nodes: int,
                   dimensions: int = 2,
                   theta: float = 1000.0,
                   rate: float = 1.0) -> Tuple[nx.Graph, nx.Graph, nx.Graph]:
    """Creates a connected graph.

      The graphs are geographic threshold graphs, but with added edges via a
      minimum spanning tree algorithm, to ensure all nodes are connected.

      Args:
      rand: A random seed for the graph generator. Default= None.
      num_nodes_min: A min number of nodes per graph.
      num_nodes_max: A max number of nodes per graph.
      dimensions: (optional) An `int` number of dimensions for the positions.
        Default= 2.
      theta: (optional) A `float` threshold parameters for the geographic
        threshold graph's threshold. Large values (1000+) make mostly trees. Try
        20-60 for good non-trees. Default=1000.0.
      rate: (optional) A rate parameter for the node weight exponential sampling
        distribution. Default= 1.0.

      Returns:
        The graph.
    """
    # Sample num_nodes.
    num_nodes = rand.integers(min_num_nodes, max_num_nodes)

    # Create geographic threshold graph.
    pos_array = rand.uniform(size=(num_nodes, dimensions))
    pos = dict(enumerate(pos_array))
    weight = dict(enumerate(rand.exponential(rate, size=num_nodes)))
    geo_graph = nx.geographical_threshold_graph(num_nodes,
                                                theta,
                                                pos=pos,
                                                weight=weight)

    # Create minimum spanning tree across geo_graph's nodes.
    distances = spatial.distance.squareform(spatial.distance.pdist(pos_array))
    i_, j_ = np.meshgrid(range(num_nodes), range(num_nodes), indexing="ij")
    weighted_edges = list(zip(i_.ravel(), j_.ravel(), distances.ravel()))
    mst_graph = nx.Graph()
    mst_graph.add_weighted_edges_from(weighted_edges,
                                      weight=DISTANCE_WEIGHT_NAME)
    mst_graph = nx.minimum_spanning_tree(mst_graph, weight=DISTANCE_WEIGHT_NAME)
    # Put geo_graph's node attributes into the mst_graph.
    for i in mst_graph.nodes():
        mst_graph.nodes[i].update(geo_graph.nodes[i])

    # Compose the graphs.
    combined_graph = nx.compose_all((mst_graph, geo_graph.copy()))
    # Put all distance weights into edge attributes.
    for i, j in combined_graph.edges():
        combined_graph.get_edge_data(i, j).setdefault(DISTANCE_WEIGHT_NAME,
                                                      distances[i, j])
    return combined_graph, geo_graph, mst_graph


def add_shortest_path(rand: np.random.Generator,
                      graph: nx.Graph,
                      min_length: int = 1) -> nx.DiGraph:
    """Samples a shortest path from A to B and adds attributes to indicate it.
  
    Args:
      rand: A random seed for the graph generator. Default= None.
      graph: A `nx.Graph`.
      min_length: (optional) An `int` minimum number of edges in the shortest
        path. Default= 1.
  
    Returns:
      The `nx.DiGraph` with the shortest path added.
  
    Raises:
      ValueError: All shortest paths are below the minimum length
    """
    lengths = nx.all_pairs_shortest_path_length(graph)

    pair_to_length_dict = {}
    for x, yy in lengths:
        for y, l in yy.items():
            if l >= min_length:
                pair_to_length_dict[x, y] = l
    if max(pair_to_length_dict.values()) < min_length:
        raise ValueError("All shortest paths are below the minimum length")
    # The node pairs which exceed the minimum length.
    node_pairs = list(pair_to_length_dict)

    # Computes probabilities per pair, to enforce uniform sampling of each
    # shortest path lengths.
    # The counts of pairs per length.
    counts = collections.Counter(pair_to_length_dict.values())
    prob_per_length = 1.0 / len(counts)
    probabilities = [
        prob_per_length / counts[pair_to_length_dict[x]] for x in node_pairs
    ]

    # Choose the start and end points.
    i = rand.choice(len(node_pairs), p=probabilities)
    start, end = node_pairs[i]
    path = nx.shortest_path(graph,
                            source=start,
                            target=end,
                            weight=DISTANCE_WEIGHT_NAME)

    # Creates a directed graph, to store the directed path from start to end.
    digraph = graph.to_directed()

    # Add the "start", "end", and "solution" attributes to the nodes and edges.
    digraph.add_node(start, start=True)
    digraph.add_node(end, end=True)
    digraph.add_nodes_from(set_diff(digraph.nodes(), [start]), start=False)
    digraph.add_nodes_from(set_diff(digraph.nodes(), [end]), end=False)
    digraph.add_nodes_from(set_diff(digraph.nodes(), path), solution=False)
    digraph.add_nodes_from(path, solution=True)
    path_edges = list(pairwise(path))
    digraph.add_edges_from(set_diff(digraph.edges(), path_edges),
                           solution=False)
    digraph.add_edges_from(path_edges, solution=True)

    return digraph


def graph_to_input_target(graph):
    """Returns 2 graphs with input and target feature vectors for training.
  
    Args:
      graph: An `nx.DiGraph` instance.
  
    Returns:
      The input `nx.DiGraph` instance.
      The target `nx.DiGraph` instance.
  
    Raises:
      ValueError: unknown node type
    """

    def create_feature(attr, fields, dtype=np.float32):
        return np.hstack(
            [np.array(attr[field], dtype=dtype) for field in fields])

    input_node_fields = ("pos", "weight", "start", "end", "solution")
    input_edge_fields = ("distance", "solution")

    input_graph = graph.copy()

    solution_length = 0
    for node_index, node_feature in graph.nodes(data=True):
        input_graph.add_node(node_index,
                             features=create_feature(node_feature,
                                                     input_node_fields))

        solution_length += int(node_feature["solution"])
    solution_length /= graph.number_of_nodes()

    for receiver, sender, features in graph.edges(data=True):
        input_graph.add_edge(sender,
                             receiver,
                             features=create_feature(features,
                                                     input_edge_fields))
    input_graph.graph["features"] = np.array([solution_length], dtype=float)

    return input_graph


class ShortestPathDatasetConfig(tfds.core.BuilderConfig):
    """Config of the dataset."""

    def __init__(self, num_examples: int, min_num_nodes: int,
                 max_num_nodes: int, dimensions: int, theta: float, rate: float,
                 name: str, version: tfds.core.Version, description: str,
                 **kwargs):
        """Constructor."""
        super().__init__(name=name,
                         version=version,
                         description=description,
                         **kwargs)
        self.num_examples = num_examples
        self.min_num_nodes = min_num_nodes
        self.max_num_nodes = max_num_nodes
        self.dimensions = dimensions
        self.theta = theta
        self.rate = rate


class ShortestPath(tfds.core.GeneratorBasedBuilder):
    """The shortest path of graphs dataset."""

    BUILDER_CONFIGS = [
        ShortestPathDatasetConfig(num_examples=1000,
                                  min_num_nodes=15,
                                  max_num_nodes=17,
                                  dimensions=2,
                                  theta=40.,
                                  rate=1.0,
                                  name="default",
                                  version=tfds.core.Version("0.1.0"),
                                  description="The default config."),
        ShortestPathDatasetConfig(num_examples=10_000,
                                  min_num_nodes=15,
                                  max_num_nodes=17,
                                  dimensions=2,
                                  theta=40.,
                                  rate=1.0,
                                  name="medium",
                                  version=tfds.core.Version("0.1.0"),
                                  description="The default config."),
        ShortestPathDatasetConfig(num_examples=100_000,
                                  min_num_nodes=15,
                                  max_num_nodes=17,
                                  dimensions=2,
                                  theta=40.,
                                  rate=1.0,
                                  name="large",
                                  version=tfds.core.Version("0.1.0"),
                                  description="The default config.")
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Describe the metainformation of the dataset."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "graph":
                    tfds.features.FeaturesDict({
                        "nodes":
                            tfds.features.Tensor(shape=(None, 6),
                                                 dtype=tf.float32),
                        "edges":
                            tfds.features.Tensor(shape=(None, 2),
                                                 dtype=tf.float32),
                        "receivers":
                            tfds.features.Tensor(shape=(None,), dtype=tf.int32),
                        "senders":
                            tfds.features.Tensor(shape=(None,), dtype=tf.int32),
                        "globals":
                            tfds.features.Tensor(shape=(1,), dtype=tf.float64),
                        "n_node":
                            tfds.features.Tensor(shape=(), dtype=tf.int32),
                        "n_edge":
                            tfds.features.Tensor(shape=(), dtype=tf.int32)
                    }),
            }))

    def _split_generators(
        self, dl_manager: tfds.download.DownloadManager
    ) -> Dict[str, tfds.core.SplitGenerator]:
        del dl_manager
        return {
            k[0]: self._generate_examples(
                int(k[1] * self.builder_config.num_examples))
            for k in [("train", 0.8), ("validation", 0.1), ("test", 0.1)]
        }

    def _generate_examples(self, num_examples: int) -> tfds.core.SplitGenerator:
        rng = np.random.default_rng()
        for i in range(num_examples):
            graph, _, _ = generate_graph(
                rand=rng,
                min_num_nodes=self.builder_config.min_num_nodes,
                max_num_nodes=self.builder_config.max_num_nodes,
                dimensions=self.builder_config.dimensions,
                theta=self.builder_config.theta,
                rate=self.builder_config.rate)
            graph = add_shortest_path(rng, graph)

            yield i, {
                "graph":
                    utils_np.networkx_to_data_dict(graph_to_input_target(graph)
                                                  ),
            }
