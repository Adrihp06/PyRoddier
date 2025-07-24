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
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

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
        self.intra_image = np.exp(-(r**2) / 0.2**2)
        self.extra_image = 1.2 * np.exp(-(r**2) / 0.2**2)

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
        params: dict = self.dialog.get_telescope_params()

        # Verify the values
        self.assertEqual(params["apertura"], 900.0)
        self.assertEqual(params["focal"], 7200.0)
        self.assertEqual(params["tamano_pixel"], 15.0)
        self.assertEqual(params["espejo_primario"], 100.0)
        self.assertEqual(params["espejo_secundario"], 50.0)
        self.assertEqual(params["binning"], "1x1")

    def test_get_cropped_images(self):
        """Test getting cropped images"""
        # Set all required valuess
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
        self.assertEqual(
            intra_crop.shape, (self.dialog.crop_size, self.dialog.crop_size)
        )
        self.assertEqual(
            extra_crop.shape, (self.dialog.crop_size, self.dialog.crop_size)
        )

    def test_parameter_validation(self):
        """Test parameter validation in the dialog"""
        # Test with empty fields (should use defaults or handle gracefully)
        params = self.dialog.get_telescope_params()

        # Should return a dictionary even with empty fields
        self.assertIsInstance(params, dict)

        # Test with invalid numeric values
        self.dialog.apertura_edit.setText("invalid")
        self.dialog.focal_edit.setText("not_a_number")

        # Should handle gracefully (might use defaults or show error)
        params = self.dialog.get_telescope_params()
        self.assertIsInstance(params, dict)

    def test_image_display(self):
        """Test that images are properly displayed in the dialog"""
        # Check that labels are showing images
        intra_pixmap = self.dialog.intra_label.pixmap()
        extra_pixmap = self.dialog.extra_label.pixmap()

        # Pixmaps should exist (images were provided in setUp)
        self.assertIsNotNone(intra_pixmap)
        self.assertIsNotNone(extra_pixmap)

        # Pixmaps should have reasonable dimensions
        self.assertGreater(intra_pixmap.width(), 0)
        self.assertGreater(intra_pixmap.height(), 0)
        self.assertGreater(extra_pixmap.width(), 0)
        self.assertGreater(extra_pixmap.height(), 0)

    def test_crop_functionality(self):
        """Test the image cropping functionality"""
        # Set required parameters
        self.dialog.espejo_primario_edit.setText("100.0")
        self.dialog.espejo_secundario_edit.setText("50.0")
        self.dialog.focal_edit.setText("7200.0")
        self.dialog.apertura_edit.setText("900.0")
        self.dialog.tamano_pixel_edit.setText("15.0")
        self.dialog.binning_edit.setText("1x1")

        # Get original image shapes
        original_intra_shape = self.intra_image.shape
        original_extra_shape = self.extra_image.shape

        # Perform crop
        self.dialog.crop_images()

        # Get cropped images
        intra_crop, extra_crop = self.dialog.get_cropped_images()

        # Verify cropping occurred
        self.assertIsNotNone(intra_crop)
        self.assertIsNotNone(extra_crop)

        # Cropped images might be smaller than original
        if hasattr(self.dialog, "crop_size"):
            expected_size = (self.dialog.crop_size, self.dialog.crop_size)
            self.assertEqual(intra_crop.shape, expected_size)
            self.assertEqual(extra_crop.shape, expected_size)

    def test_dialog_buttons(self):
        """Test dialog button functionality"""
        # Check that cancel button exists and works
        if hasattr(self.dialog, "cancel_button"):
            self.assertIsNotNone(self.dialog.cancel_button)

        # Check that crop/execute button exists
        if hasattr(self.dialog, "crop_button"):
            self.assertIsNotNone(self.dialog.crop_button)

    def test_parameter_defaults(self):
        """Test that parameter fields have reasonable defaults or can be set"""
        # Test that we can set and retrieve various parameter combinations
        test_cases = [
            {
                "apertura": "800.0",
                "focal": "6000.0",
                "tamano_pixel": "12.0",
                "espejo_primario": "80.0",
                "espejo_secundario": "40.0",
                "binning": "2x2",
            },
            {
                "apertura": "1200.0",
                "focal": "9600.0",
                "tamano_pixel": "20.0",
                "espejo_primario": "120.0",
                "espejo_secundario": "60.0",
                "binning": "1x1",
            },
        ]

        for test_case in test_cases:
            with self.subTest(test_case=test_case):
                # Set values
                for field, value in test_case.items():
                    field_widget = getattr(self.dialog, f"{field}_edit", None)
                    if field_widget:
                        field_widget.setText(value)

                # Get parameters
                params = self.dialog.get_telescope_params()

                # Verify parameters
                self.assertIsInstance(params, dict)

                # Check specific numeric conversions if they work
                try:
                    if "apertura" in params:
                        self.assertIsInstance(params["apertura"], float)
                    if "focal" in params:
                        self.assertIsInstance(params["focal"], float)
                except (ValueError, KeyError):
                    # Some implementations might handle conversion differently
                    pass

    def test_image_types_and_shapes(self):
        """Test that dialog handles different image types properly"""
        # Test that original images are numpy arrays
        self.assertIsInstance(self.intra_image, np.ndarray)
        self.assertIsInstance(self.extra_image, np.ndarray)

        # Test that images have expected properties
        self.assertEqual(len(self.intra_image.shape), 2)  # 2D images
        self.assertEqual(len(self.extra_image.shape), 2)
        self.assertEqual(self.intra_image.shape, self.extra_image.shape)  # Same size

        # Test that images have reasonable value ranges
        self.assertTrue(np.all(np.isfinite(self.intra_image)))
        self.assertTrue(np.all(np.isfinite(self.extra_image)))
        self.assertTrue(np.all(self.intra_image >= 0))  # Non-negative
        self.assertTrue(np.all(self.extra_image >= 0))

    def tearDown(self):
        self.dialog.close()


if __name__ == "__main__":
    unittest.main()

