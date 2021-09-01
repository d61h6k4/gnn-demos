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

from typing import Iterable, NamedTuple

import argparse
import pathlib

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import datasets.graphs.shortest_path as spg


class Color(NamedTuple):
    r: float
    g: float
    b: float


def draw_graph(graph: nx.Graph,
               ax: plt.Axes,
               node_size: int = 200,
               node_color: Color = Color(0.4, 0.8, 0.4),
               node_linewidth: float = 1.0,
               edge_width: float = 1.0):
    pos = {node_ix: graph.nodes[node_ix]["pos"] for node_ix in graph.nodes}

    start_color = "w",
    end_color = "k",
    solution_color = Color(191. / 255, 181. / 255, 215. / 255)
    solution_node_linewidth = 3.0,
    solution_edge_width = 3.0
    node_border_color = (0.0, 0.0, 0.0, 1.0)
    # Plot start nodes
    collection = nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=[n for n in graph.nodes if graph.nodes[n].get("start", False)],
        ax=ax,
        node_size=node_size,
        node_color=start_color,
        linewidths=solution_node_linewidth,
        edgecolors=node_border_color)
    collection.set_zorder(100)

    # Plot end nodes
    collection = nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=[n for n in graph.nodes if graph.nodes[n].get("end", False)],
        ax=ax,
        node_size=node_size,
        node_color=end_color,
        linewidths=solution_node_linewidth,
        edgecolors=node_border_color)
    collection.set_zorder(90)

    # Plot intermidiate solution nodes
    intermidiate_solution_nodes = [
        n for n in graph.nodes if graph.nodes[n].get("solution", False)
    ]
    collection = nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=intermidiate_solution_nodes,
        ax=ax,
        node_size=node_size,
        node_color=np.tile(
            np.array(solution_color)[None],
            [len(intermidiate_solution_nodes), 1]),
        linewidths=solution_node_linewidth,
        edgecolors=node_border_color)
    collection.set_zorder(80)

    # Plot solution edges.
    collection = nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=[
            e for e in graph.edges
            if graph.get_edge_data(e[0], e[1]).get("solution", False)
        ],
        ax=ax,
        width=solution_edge_width)

    # Plot non-solution nodes
    intermidiate_solution_nodes = [
        n for n in graph.nodes if not graph.nodes[n].get("solution", False)
    ]
    collection = nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=intermidiate_solution_nodes,
        ax=ax,
        node_size=node_size,
        node_color=np.tile(
            np.array(node_color)[None], [len(intermidiate_solution_nodes), 1]),
        linewidths=node_linewidth)
    collection.set_zorder(20)

    # Plot non-solution edges.
    collection = nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=[
            e for e in graph.edges
            if not graph.get_edge_data(e[0], e[1]).get("solution", False)
        ],
        ax=ax,
        width=edge_width)

    return collection


def show(ds: Iterable[object],
         rows: int = 1,
         cols: int = 1,
         plot_scale: float = 5,
         **kwargs):
    # Create subplots.
    fig, axs = plt.subplots(nrows=rows,
                            ncols=cols,
                            squeeze=False,
                            figsize=(plot_scale * cols, plot_scale * rows))

    plt.subplots_adjust(hspace=1 / plot_scale, wspace=1 / plot_scale)

    for nx_graph, ax in zip(ds, axs.reshape(-1)):
        # Draw with NetworkX.
        draw_graph(nx_graph, ax=ax)

    return fig


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output",
                        type=pathlib.Path,
                        required=True,
                        help="Folder to store generated images")

    return parser.parse_args()


def main():
    args = parse_args()

    rng = np.random.default_rng()

    graphs = []

    graphs_num = 10
    for _ in range(graphs_num):
        combined_graph, geo_graph, mst_graph = spg.generate_graph(rng,
                                                                  16,
                                                                  17,
                                                                  theta=40.)
        graphs.extend([
            geo_graph, mst_graph, combined_graph,
            spg.add_shortest_path(rng, combined_graph)
        ])

    fig = show(graphs, rows=graphs_num, cols=4)
    fig.savefig(args.output / "shortest_path_graphs.png")


if __name__ == "__main__":
    main()
