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

import unittest

import numpy as np
import datasets.graphs.shortest_path as spg


class TestGenerateGraph(unittest.TestSuite):

    def setUp(self):
        self.rand = np.random.default_rng()

    def test_smoke(self):
        graph = spg.generate_graph(self.rand, min_num_nodes=5, max_num_nodes=10)


if __name__ == "__main__":
    unittest.main()
