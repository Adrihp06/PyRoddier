# Copyright (c) 2025 Adrián Hernández Padrón
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QWheelEvent
import numpy as np
from astropy.io import fits
import pytest

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.main import main
from src.gui.main_window import FitsViewer

# Create QApplication instance for all tests
@pytest.fixture(scope="session")
def qapp():
    return QApplication(sys.argv)

@pytest.fixture
def viewer(qapp):
    return FitsViewer()

def test_initial_state(viewer):
    """Test the initial state of the FitsViewer"""
    assert viewer.intra_image_path is None
    assert viewer.extra_image_path is None
    assert viewer.intra_image_data is None
    assert viewer.extra_image_data is None
    assert viewer.zoom_factor == 1.0
    assert viewer.is_dark_theme is True

def test_theme_toggle(viewer):
    """Test theme toggle functionality"""
    initial_theme = viewer.is_dark_theme
    viewer.toggle_theme()
    assert initial_theme != viewer.is_dark_theme
    viewer.toggle_theme()
    assert initial_theme == viewer.is_dark_theme

def test_reset_state(viewer):
    """Test the reset state functionality"""
    # Create dummy data
    viewer.intra_image_path = "dummy_path"
    viewer.extra_image_path = "dummy_path"
    viewer.intra_image_data = np.zeros((100, 100))
    viewer.extra_image_data = np.zeros((100, 100))
    viewer.zoom_factor = 2.0

    # Reset state
    viewer.reset_state()

    # Verify reset
    assert viewer.intra_image_path is None
    assert viewer.extra_image_path is None
    assert viewer.intra_image_data is None
    assert viewer.extra_image_data is None
    assert viewer.zoom_factor == 1.0

def test_center_of_mass_calculation(viewer):
    """Test center of mass calculation"""
    # Create a test image with a bright spot at (25, 25)
    test_image = np.zeros((50, 50))
    test_image[25, 25] = 1.0

    com_y, com_x = viewer.calculate_center_of_mass(test_image)
    assert com_y == 25
    assert com_x == 25

def test_zoom_handling(viewer):
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
    initial_zoom = viewer.zoom_factor
    event = QWheelEvent(pos, global_pos, pixel_delta, angle_delta, 0, Qt.Vertical, buttons, modifiers)
    viewer.handle_wheel_event(event, viewer.intra_label)
    assert viewer.zoom_factor > initial_zoom

    # Test zoom out with control modifier
    initial_zoom = viewer.zoom_factor
    event = QWheelEvent(pos, global_pos, pixel_delta, QPoint(0, -120), 0, Qt.Vertical, buttons, modifiers)
    viewer.handle_wheel_event(event, viewer.intra_label)
    assert viewer.zoom_factor < initial_zoom

    # Test without control modifier (should not change zoom)
    initial_zoom = viewer.zoom_factor
    event = QWheelEvent(pos, global_pos, pixel_delta, angle_delta, 0, Qt.Vertical, buttons, Qt.NoModifier)
    viewer.handle_wheel_event(event, viewer.intra_label)
    assert viewer.zoom_factor == initial_zoom