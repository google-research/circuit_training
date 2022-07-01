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
"""Splits Protobuf circuit graph."""

import os
from typing import List, Optional

from absl import logging

# Internal gfile dependencies


# Cut the file just below 2GB.
_MAX_FILE_SIZE = (2 * 1024 - 1) * 1024 * 1024

# Print progress every 100MB.
_PRINT_POS_INTERVAL = 100 * 1024 * 1024


def split_proto_netlist(
    file_name: str,
    output_dir: str,
    max_file_size: int = _MAX_FILE_SIZE,
    print_pos_interval: int = _PRINT_POS_INTERVAL) -> Optional[List[str]]:
  """Main function of split Protobuf circuit graph.

  Args:
    file_name: Input text protobuf file.
    output_dir: Location where the split files to be written.
    max_file_size: Max file size.
    print_pos_interval: Print position interval.
  Returns:
    List of names of split files. 'None' if there's a failure.
  """
  print('-------------------')
  print('split_proto_netlist')
  print('-------------------')
  print('Max file size = {}MB'.format(int(max_file_size / (1024*1024))))
  print('Print interval = {}MB'.format(int(print_pos_interval / (1024*1024))))

  print('Input file: ', file_name)
  basename = os.path.basename(file_name)
  if not basename.endswith('.pb.txt'):
    logging.error('The input file name doesn\'t end with .pb.txt')
    return None
  if output_dir is None:
    output_dir = os.path.dirname(file_name)

  file_name_base = os.path.join(output_dir, basename[0:-7])
  outfile_cnt = 1
  outfile_name = '{}.part{}.pb.txt'.format(file_name_base, outfile_cnt)
  print('Output file: ', outfile_name)
  try:
    outfile = open(outfile_name, 'w')
  except IOError:
    logging.error('Cannot open output file %s', outfile_name)
    return None

  split_list = [outfile_name]
  infile_pos = 0
  ready_to_close = False
  next_close_pos = max_file_size
  next_print_pos = print_pos_interval
  with open(file_name, 'rt') as infile:
    for line in infile:
      outfile.write(line)
      infile_pos += len(line) + 1
      if infile_pos >= next_print_pos:
        print('Reading input file at {}MB'.format(int(infile_pos/(1024*1024))))
        next_print_pos += print_pos_interval

      if infile_pos > next_close_pos:
        # Now search for the end of the clause.
        ready_to_close = True

      if ready_to_close and line[0] == '}':
        outfile.close()
        ready_to_close = False
        next_close_pos += max_file_size
        outfile_cnt += 1
        outfile_name = '{}.part{}.pb.txt'.format(file_name_base, outfile_cnt)
        print('Output file: {}'.format(outfile_name))
        split_list.append(outfile_name)
        try:
          outfile = open(outfile_name, 'w')
        except IOError:
          logging.error('Cannot open output file %s', outfile_name)
          return None

  outfile.close()
  return split_list
