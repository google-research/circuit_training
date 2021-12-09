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
"""Pytest fixtures."""
import sys

from absl import flags
import pytest


@pytest.fixture(scope='session', autouse=True)
def parse_flags():
  """Triggers flags to be parsed before tests are executed.

  Without this fixture FLAGs are not parsed and an error that flags are being
  used before they are parsed is thrown.
  """
  # Only pass the first item, because pytest flags shouldn't be parsed as absl
  # flags.
  flags.FLAGS(sys.argv[:1])
