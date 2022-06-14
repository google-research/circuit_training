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
"""Tests for circuit_training.grouping.grid_size_selection."""

import collections
import os

from absl import flags
from absl.testing import absltest

from circuit_training.environment import placement_util
from circuit_training.grouping import grid_size_selection
from circuit_training.environment import plc_client
from circuit_training.utils import test_utils

FLAGS = flags.FLAGS


class GridSizeSelectionTest(absltest.TestCase):

  def test_span_and_extra_piece(self):
    # Testing the routine that calcultaes the span of a segment in terms of
    # gcell width, and the extra piece that occupies the last gcell.
    data = collections.namedtuple(
        'data', 'seg_width, gcell_width, exp_span, exp_piece')
    test_vect = [
        data(1.0, 1.0, 1, 1.0),
        data(1.5, 1.0, 3, 0.25),
        data(1.5, 2.0, 1, 1.75),
        data(3.3, 1.0, 5, 0.15),
        data(3.0, 1.0, 3, 1.0),
        data(20.0, 4.0, 5, 4.0),
        data(20.1, 4.0, 7, 0.05)
    ]
    for d in test_vect:
      span, extra_piece = grid_size_selection.get_span_and_extra_piece(
          d.seg_width, d.gcell_width)
      self.assertEqual(span, d.exp_span)
      self.assertAlmostEqual(extra_piece, d.exp_piece)

  def test_waste_ratio_of_segments(self):
    # Tests the wasted (unoccupied) space when a list of segments
    # are placed next to each other in one dimension.
    data = collections.namedtuple('data', 'segment_widths, cell_width, exp_wr')
    test_vect = [
        data([1, 2], 1, 0.25),
        data([2, 3], 3, 1.0 / 6),
        data([1, 2, 3], 3, 1.0 / 3),
        data([1, 5, 1], 3, 2.0 / 9),
        data([3], 2, 0.5),
        data([6, 10], 4, 0.2),
        data([10, 6], 4, 0.2),
        data([1, 1, 1, 1], 2, 0.5),
        data([2, 2, 2, 2], 1, 1.0 / 9)
    ]
    for d in test_vect:
      wr = grid_size_selection.get_waste_ratio(d.segment_widths, d.cell_width)
      self.assertAlmostEqual(wr, d.exp_wr)

  def test_get_grid_choices(self):
    # Tests the number of columns/rows choice for a netlist with some
    # constraints.
    data = collections.namedtuple(
        'data', 'c_width, c_height, min_num, max_num,'
        ' max_grid_size, min_num_grid_cells, '
        ' max_num_grid_cells, max_aspect_ratio, '
        ' add_size, include_fixed_macros, exp_cols, exp_rows')
    test_vect = [
        data(3, 3, 1, 10, 128, 5, 100, 2, 0, False, 3, 3),
        data(4, 4, 1, 10, 128, 5, 100, 2, 0, False, 4, 4),
        data(4, 4, 1, 10, 128, 5, 100, 2, 0.2, False, 3, 3),
    ]
    test_netlist_dir = ('circuit_training/'
                        'grouping/testdata')
    filename = os.path.join(FLAGS.test_srcdir, test_netlist_dir,
                            'macro_tiles_3rows_3cols_1pins.pb.txt')
    plc = plc_client.PlacementCost(filename)
    for d in test_vect:
      plc.set_canvas_size(d.c_width, d.c_height)
      grid_choices = grid_size_selection.get_grid_choices(
          plc, d.min_num, d.max_num, d.max_grid_size, d.min_num_grid_cells,
          d.max_num_grid_cells, d.max_aspect_ratio, d.add_size,
          d.include_fixed_macros)
      self.assertNotEmpty(grid_choices)
      cols, rows = grid_size_selection.select_from_grid_choices(grid_choices)
      self.assertEqual(cols, d.exp_cols)
      self.assertEqual(rows, d.exp_rows)


if __name__ == '__main__':
  test_utils.main()
