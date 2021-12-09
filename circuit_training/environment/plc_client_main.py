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
"""An example and simple binary to create and call plc client."""

from typing import Sequence

from absl import app
from absl import flags
from circuit_training.environment import plc_client

flags.DEFINE_string("netlist_file", None, "Path to the input netlist file.")

flags.mark_flags_as_required([
    "netlist_file",
])

FLAGS = flags.FLAGS


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  plc = plc_client.PlacementCost(netlist_file=FLAGS.netlist_file)

  print("get_cost:", plc.get_cost())
  print("get_congestion_cost:", plc.get_congestion_cost())
  print("get_density_cost:", plc.get_density_cost())

  hard_macro_indices = [
      m for m in plc.get_macro_indices() if not plc.is_node_soft_macro(m)
  ]
  print("hard_macro_indices:", hard_macro_indices)
  print("get_node_mask:", plc.get_node_mask(hard_macro_indices[0]))


if __name__ == "__main__":
  app.run(main)
