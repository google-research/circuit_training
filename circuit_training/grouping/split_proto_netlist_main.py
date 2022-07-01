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
r"""Splits Protobuf circuit graph when the file size is larger than 2GB.

Example command:

python circuittraining/grouping/split_proto_netlist_main \
--file_name=ariane.pb.txt \
--output_dir=/tmp/output_dir
"""
from collections.abc import Sequence

from absl import app
from absl import flags

from circuit_training.grouping import split_proto_netlist

_FILE_NAME = flags.DEFINE_string('file_name', None, 'input file name')
_OUTPUT_DIR = flags.DEFINE_string('output_dir', None, 'output directory')

FLAGS = flags.FLAGS

flags.mark_flags_as_required(['file_name'])


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  split_list = split_proto_netlist.split_proto_netlist(_FILE_NAME.value,
                                                       _OUTPUT_DIR.value)
  print('Split into {} parts.'.format(len(split_list)))


if __name__ == '__main__':
  app.run(main)
