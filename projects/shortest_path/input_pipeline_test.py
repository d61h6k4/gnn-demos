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

from absl.testing import absltest

import tensorflow_datasets as tfds

import projects.shortest_path.input_pipeline as input_pipeline


class InputPipelineTest(absltest.TestCase):

    def test_sanity_check(self):
        dataset_builder = tfds.builder("shortest_path")
        ds = input_pipeline.create_split(dataset_builder, True)
        self.assertTrue(ds is not None)

    def test_create_input_iter(self):
        dataset_builder = tfds.builder("shortest_path")
        it = input_pipeline.create_input_iter(dataset_builder,
                                              batch_size=2,
                                              n_node_per_graph=17,
                                              n_edge_per_graph=17,
                                              dtype=int,
                                              train=True,
                                              cache=True)
        for ex in it:
            print(ex)
            break


if __name__ == "__main__":
    absltest.main()
