# Copyright (c) 2025 Adrián Hernández Padrón
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt
import unittest
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.gui.dialogs.roddiertest import RoddierTestDialog

class TestRoddierTestDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(sys.argv)

    def setUp(self):
        # Create test images
        self.size = 100
        x, y = np.meshgrid(np.linspace(-1, 1, self.size), np.linspace(-1, 1, self.size))
        r = np.sqrt(x**2 + y**2)
        self.intra_image = np.exp(-r**2 / 0.2**2)
        self.extra_image = 1.2 * np.exp(-r**2 / 0.2**2)

        self.dialog = RoddierTestDialog(self.intra_image, self.extra_image)

    def test_initial_state(self):
        """Test the initial state of the dialog"""
        self.assertIsNotNone(self.dialog.intra_label)
        self.assertIsNotNone(self.dialog.extra_label)
        self.assertIsNotNone(self.dialog.apertura_edit)
        self.assertIsNotNone(self.dialog.focal_edit)
        self.assertIsNotNone(self.dialog.tamano_pixel_edit)
        self.assertIsNotNone(self.dialog.max_order_edit)

    def test_get_telescope_params(self):
        """Test getting telescope parameters"""
        # Set all required values
        self.dialog.espejo_primario_edit.setText("100.0")
        self.dialog.espejo_secundario_edit.setText("50.0")
        self.dialog.focal_edit.setText("7200.0")
        self.dialog.apertura_edit.setText("900.0")
        self.dialog.tamano_pixel_edit.setText("15.0")
        self.dialog.binning_edit.setText("1x1")

        # Get parameters
        params = self.dialog.get_telescope_params()

        # Verify the values
        self.assertEqual(params['apertura'], 900.0)
        self.assertEqual(params['focal'], 7200.0)
        self.assertEqual(params['tamano_pixel'], 15.0)
        self.assertEqual(params['espejo_primario'], 100.0)
        self.assertEqual(params['espejo_secundario'], 50.0)
        self.assertEqual(params['binning'], "1x1")

    def test_get_cropped_images(self):
        """Test getting cropped images"""
        # Set all required values
        self.dialog.espejo_primario_edit.setText("100.0")
        self.dialog.espejo_secundario_edit.setText("50.0")
        self.dialog.focal_edit.setText("7200.0")
        self.dialog.apertura_edit.setText("900.0")
        self.dialog.tamano_pixel_edit.setText("15.0")
        self.dialog.binning_edit.setText("1x1")

        # Get images before cropping
        intra_crop, extra_crop = self.dialog.get_cropped_images()

        # Ya no comprobamos que sean None, solo que sean arrays válidos tras crop
        self.dialog.crop_images()
        intra_crop, extra_crop = self.dialog.get_cropped_images()

        # Verify the cropped images
        self.assertIsNotNone(intra_crop)
        self.assertIsNotNone(extra_crop)
        self.assertEqual(intra_crop.shape, (self.dialog.crop_size, self.dialog.crop_size))
        self.assertEqual(extra_crop.shape, (self.dialog.crop_size, self.dialog.crop_size))

    def tearDown(self):
        self.dialog.close()

if __name__ == '__main__':
    unittest.main()