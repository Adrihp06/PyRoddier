import unittest
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt

# Add the src directory to the Python path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.gui.dialogs.roddiertest import RoddierTestDialog

class TestRoddierTestDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create QApplication instance for all tests
        cls.app = QApplication(sys.argv)

    def setUp(self):
        # Create test images
        self.intra_image = np.random.rand(100, 100)
        self.extra_image = np.random.rand(100, 100)
        self.dialog = RoddierTestDialog(self.intra_image, self.extra_image)

    def test_initial_state(self):
        """Test the initial state of the dialog"""
        self.assertEqual(self.dialog.crop_size, 250)
        self.assertIsNone(self.dialog.crop_center)
        self.assertIsNone(self.dialog.cropped_intra)
        self.assertIsNone(self.dialog.cropped_extra)

        # Check default telescope parameters
        self.assertEqual(self.dialog.telescope_params['apertura'], 0.0)
        self.assertEqual(self.dialog.telescope_params['focal'], 0.0)
        self.assertEqual(self.dialog.telescope_params['pixel_scale'], 0.0)
        self.assertEqual(self.dialog.telescope_params['max_order'], 6)
        self.assertFalse(self.dialog.telescope_params['substract_tilt_and_defocus'])

    def test_get_telescope_params(self):
        """Test getting telescope parameters"""
        # Set some values
        self.dialog.apertura.setValue(900)
        self.dialog.focal.setValue(7200)
        self.dialog.pixel_scale_spin.setValue(15)
        self.dialog.binning_spin.setValue(2)
        self.dialog.substract_tilt_and_defocus.setChecked(True)

        params = self.dialog.get_telescope_params()

        self.assertEqual(params['apertura'], 900)
        self.assertEqual(params['focal'], 7200)
        self.assertEqual(params['pixel_scale'], 15)
        self.assertEqual(params['binning'], 2)
        self.assertTrue(params['substract_tilt_and_defocus'])

    def test_get_cropped_images(self):
        """Test getting cropped images"""
        # Set some cropped images
        self.dialog.cropped_intra = np.random.rand(50, 50)
        self.dialog.cropped_extra = np.random.rand(50, 50)

        intra, extra = self.dialog.get_cropped_images()

        np.testing.assert_array_equal(intra, self.dialog.cropped_intra)
        np.testing.assert_array_equal(extra, self.dialog.cropped_extra)

if __name__ == '__main__':
    unittest.main()