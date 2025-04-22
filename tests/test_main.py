# Copyright (c) 2025 Adrián Hernández Padrón
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import sys
import os
import unittest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QWheelEvent
import numpy as np
from astropy.io import fits

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.main import main
from src.gui.main_window import FitsViewer

class TestFitsViewer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the QApplication for all tests"""
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(sys.argv)

    def setUp(self):
        """Set up a new FitsViewer instance for each test"""
        self.viewer = FitsViewer()

    def test_initial_state(self):
        """Test the initial state of the FitsViewer"""
        self.assertIsNone(self.viewer.intra_image_path)
        self.assertIsNone(self.viewer.extra_image_path)
        self.assertIsNone(self.viewer.intra_image_data)
        self.assertIsNone(self.viewer.extra_image_data)
        self.assertEqual(self.viewer.zoom_factor, 1.0)
        self.assertTrue(self.viewer.is_dark_theme)

    def test_theme_toggle(self):
        """Test theme toggle functionality"""
        initial_theme = self.viewer.is_dark_theme
        self.viewer.toggle_theme()
        self.assertNotEqual(initial_theme, self.viewer.is_dark_theme)
        self.viewer.toggle_theme()
        self.assertEqual(initial_theme, self.viewer.is_dark_theme)

    def test_reset_state(self):
        """Test the reset state functionality"""
        # Create dummy data
        self.viewer.intra_image_path = "dummy_path"
        self.viewer.extra_image_path = "dummy_path"
        self.viewer.intra_image_data = np.zeros((100, 100))
        self.viewer.extra_image_data = np.zeros((100, 100))
        self.viewer.zoom_factor = 2.0

        # Reset state
        self.viewer.reset_state()

        # Verify reset
        self.assertIsNone(self.viewer.intra_image_path)
        self.assertIsNone(self.viewer.extra_image_path)
        self.assertIsNone(self.viewer.intra_image_data)
        self.assertIsNone(self.viewer.extra_image_data)
        self.assertEqual(self.viewer.zoom_factor, 1.0)

    def test_center_of_mass_calculation(self):
        """Test center of mass calculation"""
        # Create a test image with a bright spot at (25, 25)
        test_image = np.zeros((50, 50))
        test_image[25, 25] = 1.0

        com_y, com_x = self.viewer.calculate_center_of_mass(test_image)
        self.assertEqual(com_y, 25)
        self.assertEqual(com_x, 25)

    def test_zoom_handling(self):
        """Test zoom handling functionality"""
        # Create a test image
        test_image = np.zeros((100, 100))
        test_image[50, 50] = 1.0

        # Create a wheel event with control modifier
        pos = QPoint(0, 0)
        global_pos = QPoint(0, 0)
        pixel_delta = QPoint(0, 0)
        angle_delta = QPoint(0, 120)  # Positive delta for zoom in
        buttons = Qt.NoButton
        modifiers = Qt.ControlModifier

        # Test zoom in with control modifier
        initial_zoom = self.viewer.zoom_factor
        event = QWheelEvent(pos, global_pos, pixel_delta, angle_delta, 0, Qt.Vertical, buttons, modifiers)
        self.viewer.handle_wheel_event(event, self.viewer.intra_label)
        self.assertGreater(self.viewer.zoom_factor, initial_zoom)

        # Test zoom out with control modifier
        initial_zoom = self.viewer.zoom_factor
        event = QWheelEvent(pos, global_pos, pixel_delta, QPoint(0, -120), 0, Qt.Vertical, buttons, modifiers)
        self.viewer.handle_wheel_event(event, self.viewer.intra_label)
        self.assertLess(self.viewer.zoom_factor, initial_zoom)

        # Test without control modifier (should not change zoom)
        initial_zoom = self.viewer.zoom_factor
        event = QWheelEvent(pos, global_pos, pixel_delta, angle_delta, 0, Qt.Vertical, buttons, Qt.NoModifier)
        self.viewer.handle_wheel_event(event, self.viewer.intra_label)
        self.assertEqual(self.viewer.zoom_factor, initial_zoom)

    def tearDown(self):
        """Clean up after each test"""
        self.viewer.close()

if __name__ == '__main__':
    unittest.main()