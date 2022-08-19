"""Tests for placement_viewer."""
import os
import unittest
import matplotlib.pyplot as plt
from absl import app, flags
from typing import Sequence

from circuit_training.environment import plc_client
from placement_viewer import placement_viewer


class TestPlacementViewer(unittest.TestCase):
    def test_plc_viewer(self):

        test_dir = "circuit_training/placement_viewer/test_data/ariane/"
        netlist_file = os.path.join(test_dir, "netlist.pb.txt")
        init_file = os.path.join(test_dir, "rl_opt_placement.plc")
        img_name = os.path.join(test_dir, "ariane")
        os.system( "python3 -m circuit_training.placement_viewer.placement_viewer \
            --netlist_file "+  netlist_file+
                " --init_file " + init_file+
                " --img_name " + img_name )

        self.assertTrue((plt.imread(os.path.join(test_dir, "ariane_initial_placement.png")) ==
         plt.imread(os.path.join(test_dir, "test_initial_placement.png"))).all())

        self.assertTrue((plt.imread(os.path.join(test_dir, "ariane_final_placement.png")) ==
        plt.imread(os.path.join(test_dir, "test_final_placement.png"))).all())


if __name__ == '__main__':
  TestPlacementViewer().test_plc_viewer()