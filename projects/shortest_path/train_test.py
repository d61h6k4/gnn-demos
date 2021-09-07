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

import tensorflow_datasets as tfds

import projects.shortest_path.input_pipeline as input_pipeline
import projects.shortest_path.train as train


class TrainTest(unittest.TestCase):

    def setUp(self):
        dataset_builder = tfds.builder("shortest_path")
        self.ds = input_pipeline.create_split(dataset_builder, True)


if __name__ == "__main__":
    unittest.main()
