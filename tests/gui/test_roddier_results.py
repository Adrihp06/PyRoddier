# Copyright (c) 2025 Adrián Hernández Padrón
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt
import unittest
import sys

# Add the src directory to the Python path
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.gui.dialogs.roddiertestresults import RoddierTestResultsWindow

class TestRoddierResultsWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(sys.argv)

    def setUp(self):
        self.window = RoddierTestResultsWindow("Test Results")

    def test_initial_state(self):
        """Test the initial state of the window"""
        self.assertIsNotNone(self.window.wavefront_fig)
        self.assertIsNotNone(self.window.wavefront_ax)
        self.assertIsNone(self.window.zernike_coeffs)
        self.assertIsNone(self.window.zernike_base)

    def test_update_plots(self):
        """Test updating the plots with Zernike coefficients"""
        # Create test data
        coeffs = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        base = np.array([
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]]
        ], dtype=np.float64)

        # Update plots
        self.window.update_plots(coeffs, base)

        # Check that data was updated
        np.testing.assert_array_equal(self.window.zernike_coeffs, coeffs)
        np.testing.assert_array_equal(self.window.zernike_base, base)
        self.assertEqual(len(self.window.zernike_checks), len(coeffs))

    def test_update_wavefront_plot_internal(self):
        """Test updating the wavefront plot"""
        # Create test data
        coeffs = np.array([0.1, 0.2], dtype=np.float64)
        base = np.array([
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0]]
        ], dtype=np.float64)

        # Set up the data
        self.window.update_plots(coeffs, base)

        # Force an update of the wavefront plot
        self.window._update_wavefront_plot()

        # Check that the plot was updated
        self.assertTrue(len(self.window.wavefront_ax.images) > 0)

    def tearDown(self):
        self.window.close()

if __name__ == '__main__':
    unittest.main()