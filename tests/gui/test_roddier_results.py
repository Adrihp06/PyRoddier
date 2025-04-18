import unittest
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt

# Add the src directory to the Python path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.gui.dialogs.roddiertestresults import RoddierTestResultsWindow

class TestRoddierTestResultsWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create QApplication instance for all tests
        cls.app = QApplication(sys.argv)

    def setUp(self):
        self.window = RoddierTestResultsWindow("Test Results")

    def test_initial_state(self):
        """Test the initial state of the results window"""
        self.assertIsNone(self.window.zernike_coeffs)
        self.assertIsNone(self.window.zernike_base)
        self.assertEqual(len(self.window.zernike_checks), 0)

    def test_update_plots(self):
        """Test updating plots with Zernike coefficients"""
        # Create test data
        zernike_coeffs = np.random.rand(10)
        zernike_base = np.random.rand(10, 100, 100)
        annular_mask = np.ones((100, 100), dtype=bool)

        # Update plots
        self.window.update_plots(zernike_coeffs, zernike_base, annular_mask)

        # Verify data was stored
        np.testing.assert_array_equal(self.window.zernike_coeffs, zernike_coeffs)
        np.testing.assert_array_equal(self.window.zernike_base, zernike_base)
        np.testing.assert_array_equal(self.window.annular_mask, annular_mask)

        # Verify checkboxes were created
        self.assertEqual(len(self.window.zernike_checks), len(zernike_coeffs))

    def test_update_wavefront_plot(self):
        """Test updating the wavefront plot"""
        # Create test data
        zernike_coeffs = np.random.rand(10)
        zernike_base = np.random.rand(10, 100, 100)
        annular_mask = np.ones((100, 100), dtype=bool)

        # Update plots
        self.window.update_plots(zernike_coeffs, zernike_base, annular_mask)

        # Test with all checkboxes checked
        for cb in self.window.zernike_checks:
            cb.setChecked(True)

        # This should not raise any exceptions
        self.window._update_wavefront_plot()

        # Test with some checkboxes unchecked
        for i, cb in enumerate(self.window.zernike_checks):
            if i % 2 == 0:
                cb.setChecked(False)

        # This should not raise any exceptions
        self.window._update_wavefront_plot()

if __name__ == '__main__':
    unittest.main()