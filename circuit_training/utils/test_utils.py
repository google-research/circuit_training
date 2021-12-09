# coding=utf-8
# Copyright 2021 The Circuit Training Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Common utility functions for testing."""

from absl import flags

import gin
import tensorflow as tf

flags.DEFINE_multi_string('test_gin_bindings', None, 'Gin bindings.')

FLAGS = flags.FLAGS


class TestCase(tf.test.TestCase):
  """Base class for TF-Agents unit tests."""

  def setUp(self):
    super(TestCase, self).setUp()
    tf.compat.v1.enable_resource_variables()
    gin.clear_config()
    gin.parse_config(FLAGS.test_gin_bindings)

  def tearDown(self):
    gin.clear_config()
    super(TestCase, self).tearDown()


# Main function so that users of `test_utils.TestCase` can also call
# `test_utils.main()`.
def main():
  tf.test.main()
