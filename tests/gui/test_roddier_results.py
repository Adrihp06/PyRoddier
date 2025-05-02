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
        self.assertIsNotNone(self.window.interferogram_fig)
        self.assertIsNotNone(self.window.interferogram_ax)
        self.assertIsNotNone(self.window.psf_fig)
        self.assertIsNotNone(self.window.psf_ax)
        self.assertIsNone(self.window.zernike_coeffs)
        self.assertIsNone(self.window.zernike_base)
        self.assertIsNone(self.window.annular_mask)
        self.assertIsNone(self.window.interferogram_params)
        self.assertIsNone(self.window.telescope_params)

    def test_update_plots(self):
        """Test updating the plots with Zernike coefficients"""
        # Create test data
        coeffs = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        base = np.array([
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]]
        ], dtype=np.float64)
        annular_mask = np.array([[1, 1], [1, 1]], dtype=bool)
        interferogram_params = {
            'fringes': 4,
            'reference_frequency': 1.0,
            'reference_intensity': 0.5
        }
        telescope_params = {
            'apertura': 200.0,
            'focal': 1000.0,
            'tamano_pixel': 5.5
        }

        # Update plots
        self.window.update_plots(
            zernike_coeffs=coeffs,
            zernike_base=base,
            annular_mask=annular_mask,
            interferogram_params=interferogram_params,
            telescope_params=telescope_params
        )

        # Check that data was updated
        np.testing.assert_array_equal(self.window.zernike_coeffs, coeffs)
        np.testing.assert_array_equal(self.window.zernike_base, base)
        np.testing.assert_array_equal(self.window.annular_mask, annular_mask)
        self.assertEqual(self.window.interferogram_params, interferogram_params)
        self.assertEqual(self.window.telescope_params, telescope_params)
        self.assertEqual(len(self.window.zernike_checks), len(coeffs))

    def test_update_wavefront_plot_internal(self):
        """Test updating the wavefront plot"""
        # Create test data
        coeffs = np.array([0.1, 0.2], dtype=np.float64)
        base = np.array([
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0]]
        ], dtype=np.float64)
        annular_mask = np.array([[1, 1], [1, 1]], dtype=bool)
        interferogram_params = {
            'fringes': 4,
            'reference_frequency': 1.0,
            'reference_intensity': 0.5
        }
        telescope_params = {
            'apertura': 200.0,
            'focal': 1000.0,
            'tamano_pixel': 5.5
        }

        # Set up the data
        self.window.update_plots(
            zernike_coeffs=coeffs,
            zernike_base=base,
            annular_mask=annular_mask,
            interferogram_params=interferogram_params,
            telescope_params=telescope_params
        )

        # Force an update of the wavefront plot
        self.window._update_wavefront_plot()

        # Check that the plot was updated
        self.assertTrue(len(self.window.wavefront_ax.images) > 0)
        self.assertTrue(len(self.window.interferogram_ax.images) > 0)
        self.assertTrue(len(self.window.psf_ax.images) > 0)

    def test_histogram_update(self):
        """Test updating the Zernike coefficients histogram"""
        # Create test data with varying magnitudes
        coeffs = np.array([0.15, 0.08, 0.02, 0.01], dtype=np.float64)
        base = np.array([
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]]
        ], dtype=np.float64)
        annular_mask = np.array([[1, 1], [1, 1]], dtype=bool)
        interferogram_params = {
            'fringes': 4,
            'reference_frequency': 1.0,
            'reference_intensity': 0.5
        }
        telescope_params = {
            'apertura': 200.0,
            'focal': 1000.0,
            'tamano_pixel': 5.5
        }

        # Update plots
        self.window.update_plots(
            zernike_coeffs=coeffs,
            zernike_base=base,
            annular_mask=annular_mask,
            interferogram_params=interferogram_params,
            telescope_params=telescope_params
        )

        # Check that the histogram was created
        self.assertTrue(len(self.window.histogram_ax.patches) > 0)

        # Check that the bars are colored correctly based on magnitude
        bars = self.window.histogram_ax.patches
        # Red for > 0.1
        self.assertAlmostEqual(bars[0].get_facecolor()[0], 1.0, places=2)
        # Orange for > 0.05
        self.assertAlmostEqual(bars[1].get_facecolor()[0], 1.0, places=2)
        # Light orange for > 0.01
        self.assertAlmostEqual(bars[2].get_facecolor()[0], 1.0, places=2)
        # Sky blue for <= 0.01
        self.assertAlmostEqual(bars[3].get_facecolor()[0], 0.53, places=2)

        # Check that the labels are set correctly
        labels = [t.get_text() for t in self.window.histogram_ax.get_xticklabels()]
        self.assertEqual(labels[0], "Piston")
        self.assertEqual(labels[1], "Tilt X")
        self.assertEqual(labels[2], "Tilt Y")
        self.assertEqual(labels[3], "Defocus")

    def tearDown(self):
        self.window.close()

if __name__ == '__main__':
    unittest.main()