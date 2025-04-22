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
        self.assertIsNotNone(self.dialog.apertura)
        self.assertIsNotNone(self.dialog.focal)
        self.assertIsNotNone(self.dialog.pixel_scale_spin)
        self.assertIsNotNone(self.dialog.numero_de_terminos)

    def test_get_telescope_params(self):
        """Test getting telescope parameters"""
        # Set some values
        self.dialog.apertura.setValue(900.0)
        self.dialog.focal.setValue(7200.0)
        self.dialog.pixel_scale_spin.setValue(15.0)
        self.dialog.numero_de_terminos.setValue(23)

        # Get parameters
        params = self.dialog.get_telescope_params()

        # Verify the values
        self.assertEqual(params['apertura'], 900.0)
        self.assertEqual(params['focal'], 7200.0)
        self.assertEqual(params['pixel_scale'], 15.0)
        self.assertEqual(params['max_order'], 23)

    def test_get_cropped_images(self):
        """Test getting cropped images"""
        # Get images before cropping
        intra_crop, extra_crop = self.dialog.get_cropped_images()

        # Verify the cropped images are None initially
        self.assertIsNone(intra_crop)
        self.assertIsNone(extra_crop)

        # Now crop the images
        self.dialog.crop_images()
        intra_crop, extra_crop = self.dialog.get_cropped_images()

        # Verify the cropped images
        self.assertEqual(intra_crop.shape, (self.dialog.crop_size, self.dialog.crop_size))
        self.assertEqual(extra_crop.shape, (self.dialog.crop_size, self.dialog.crop_size))

    def tearDown(self):
        self.dialog.close()

if __name__ == '__main__':
    unittest.main()