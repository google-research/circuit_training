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
import os
import subprocess
import tempfile

from absl import flags
from absl import logging

# Internal gfile dependencies

_HMETIS_DIR = flags.DEFINE_string(
    'hmetis_dir', 'third_party/hmetis/',
    'Path to executable and libraries for hmetis.')


def call_hmetis(graph_file: str,
                fix_file: str,
                n_parts: int,
                ub_factor: int = 5,
                n_runs: int = 10,
                c_type: int = 5,
                r_type: int = 3,
                v_cycle: int = 3,
                reconst: int = 1,
                dbglvl: int = 0) -> str:
  """Call hMETIS.

  Args:
    graph_file: name of the hypergraph file
    fix_file: name of the file containing pre-assignment of vertices to
      partitions
    n_parts: number of partitions desired
    ub_factor: balance between the partitions (e.g., use 5 for a 45-55 bisection
      balance)
    n_runs: Number of Iterations
    c_type: HFC(1), FC(2), GFC(3), HEDGE(4), EDGE(5)
    r_type: FM(1), 1WayFM(2), EEFM(3)
    v_cycle: No(0), @End(1), ForMin(2), All(3)
    reconst: NoReconstruct_HE(0), Reconstruct_HE (1)
    dbglvl: debug level

  Returns:
    Name of the metis output file, or None if no output is generated.
  """
  hmetis_exe = os.path.join(_HMETIS_DIR.value, 'hmetis')
  assert gfile.Exists(hmetis_exe), f"{hmetis_exe} doesn't exist."
  args = [
      hmetis_exe, graph_file, fix_file, n_parts, ub_factor, n_runs, c_type,
      r_type, v_cycle, reconst, dbglvl
  ]
  args = [str(arg) for arg in args]
  logging.info('Run: %s', ' '.join(args))
  subprocess.run(args=args, check=True)

  output_file = f'{graph_file}.part.{n_parts}'
  return output_file
