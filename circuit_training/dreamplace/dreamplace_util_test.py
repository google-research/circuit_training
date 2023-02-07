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
"""Test for dreamplace_util."""

from circuit_training.dreamplace import dreamplace_util
from circuit_training.utils import test_utils


class DreamplaceUtilTest(test_utils.TestCase):

  def test_get_bin_size(self):
    num_bins_x, num_bins_y = dreamplace_util.get_bin_size(1, 4, num_bins=2 * 2)
    self.assertEqual(num_bins_x, 1)
    self.assertEqual(num_bins_y, 4)

    num_bins_x, num_bins_y = dreamplace_util.get_bin_size(4, 1, num_bins=2 * 2)
    self.assertEqual(num_bins_x, 4)
    self.assertEqual(num_bins_y, 1)

    num_bins_x, num_bins_y = dreamplace_util.get_bin_size(
        1, 100000, num_bins=128 * 128
    )
    self.assertEqual(num_bins_x, 1)
    self.assertEqual(num_bins_y, 128 * 128)

    num_bins_x, num_bins_y = dreamplace_util.get_bin_size(
        100000, 1, num_bins=128 * 128
    )
    self.assertEqual(num_bins_x, 128 * 128)
    self.assertEqual(num_bins_y, 1)

    num_bins_x, num_bins_y = dreamplace_util.get_bin_size(
        300, 5200, num_bins=128 * 128
    )
    true_ratio = 300.0 / 5200.0
    bin_ratio = 1.0 * num_bins_x / num_bins_y

    next_h_num_bins_x = num_bins_x / 2
    next_h_num_bins_y = num_bins_y * 2
    next_h_bin_ratio = 1.0 * next_h_num_bins_x / next_h_num_bins_y
    self.assertLess(
        abs(bin_ratio - true_ratio), abs(next_h_bin_ratio - true_ratio)
    )

    next_v_num_bins_x = num_bins_x * 2
    next_v_num_bins_y = num_bins_y / 2
    next_v_bin_ratio = 1.0 * next_v_num_bins_x / next_v_num_bins_y
    self.assertLess(
        abs(bin_ratio - true_ratio), abs(next_v_bin_ratio - true_ratio)
    )


if __name__ == "__main__":
  test_utils.main()
