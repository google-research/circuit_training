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
"""Tests for split_proto_netlist."""

from circuit_training.grouping import split_proto_netlist
from absl.testing import absltest

_NETLIST_FILE_PATH = 'third_party/py/circuit_training/grouping/testdata/ariane_test.pb.txt'


class SplitProtoNetlistTest(absltest.TestCase):

  def _read_files(self, file_name):
    with open(file_name, 'r') as f:
      file_string = f.read()
    return file_string

  def test_split_proto_netlist(self):
    output_dir = self.create_tempdir()
    # Sets max file size to be 1MB
    max_file_size = 1024 * 1024
    print_pos_interval = max_file_size
    output_file_list = split_proto_netlist.split_proto_netlist(
        _NETLIST_FILE_PATH, output_dir, max_file_size, print_pos_interval)
    # The original file is around 2 MB, so there should be two files in the
    # outputs.
    self.assertLen(output_file_list, 2)

    org_file_string = self._read_files(_NETLIST_FILE_PATH)
    file_part_string_list = []
    file_part_string_list.append(self._read_files(output_file_list[0]))
    file_part_string_list.append(self._read_files(output_file_list[1]))

    # Compares the file string before and after split. They should be exactly
    # the same.
    self.assertEqual(org_file_string, ''.join(file_part_string_list))


if __name__ == '__main__':
  absltest.main()
