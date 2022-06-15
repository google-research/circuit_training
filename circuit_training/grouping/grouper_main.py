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
r"""Main executable to call hmetis and do partitioning.

Optionally, it will call automatic grid col/row selection before partitioning.
It will print out cost statistics before, after partitioning, it will also
create a legalize placement file, and print cost statistics for that placement,
as well.

Example command:

python circuittraining/grouping/grouper_main \
--output_dir=/tmp/output_dir/ \
--netlist_file=ariane.pb.txt \
--block_name=ariane \
--alsologtostderr
"""

from collections.abc import Sequence

from absl import app
from absl import flags

from circuit_training.grouping import grouper

_NETLIST_FILE = flags.DEFINE_string('netlist_file', None,
                                    'Path to the input netlist file.')
_OUTPUT_DIR = flags.DEFINE_string('output_dir', None,
                                  'Base directory to output logs and results.')
_BLOCK_NAME = flags.DEFINE_string('block_name', None, 'Name of the block.')

flags.mark_flags_as_required(['output_dir', 'netlist_file', 'block_name'])

FLAGS = flags.FLAGS


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  grouped_plc, placement_file = grouper.group_stdcells(
      netlist_file=_NETLIST_FILE.value,
      output_dir=_OUTPUT_DIR.value,
      block_name=_BLOCK_NAME.value,
  )
  print('Saved legalized placement : {}'.format(placement_file))


if __name__ == '__main__':
  app.run(main)
