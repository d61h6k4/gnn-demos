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
"""Default hyperparameter configuration."""

import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.dataset = "shortest_path"
    config.n_node_per_graph = 17
    config.n_edge_per_graph = 2 * 17
    config.cache = True
    config.batch_size = 32
    config.num_epochs = 100
    config.num_train_steps = -1
    config.steps_per_eval = -1
    config.learning_rate = 32e-2
    config.warmup_epochs = 10
    config.half_precision = False
    config.momentum = 0.99
    config.log_every_steps = False

    return config
