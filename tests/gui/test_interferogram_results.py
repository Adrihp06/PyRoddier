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

from src.gui.dialogs.interferogramresults import InterferogramResultsDialog

class TestInterferogramResultsDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(sys.argv)

    def setUp(self):
        self.dialog = InterferogramResultsDialog("Test Results")

    def test_initial_state(self):
        """Test the initial state of the dialog"""
        self.assertIsNotNone(self.dialog.figure)
        self.assertIsNotNone(self.dialog.canvas)
        self.assertIsNotNone(self.dialog.ax)

    def test_update_plot(self):
        """Test updating the plot with interferogram data"""
        # Create test interferogram data
        interferogram = np.random.rand(100, 100)

        # Update plot
        self.dialog.update_plot(interferogram)

        # Verify the plot was updated
        self.assertEqual(len(self.dialog.ax.images), 1)
        self.assertEqual(self.dialog.ax.images[0].get_array().shape, (100, 100))

    def test_update_plot_invalid(self):
        """Test updating the plot with invalid data"""
        # Test with None
        with self.assertRaises(ValueError) as context:
            self.dialog.update_plot(None)
        self.assertEqual(str(context.exception), "El interferograma no puede ser None")

        # Test with empty array
        with self.assertRaises(ValueError) as context:
            self.dialog.update_plot(np.array([]))
        self.assertEqual(str(context.exception), "El interferograma no puede estar vacío")

    def tearDown(self):
        self.dialog.close()

if __name__ == '__main__':
    unittest.main()