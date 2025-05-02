# Copyright (c) 2025 Adrián Hernández Padrón
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import numpy as np
import os
import tempfile
import unittest
from astropy.io import fits
import shutil

# Add the src directory to the Python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.common.utils import load_fits_image

class TestImagePreprocessing(unittest.TestCase):
    def setUp(self):
        # Create a temporary FITS file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir,         'test.fits')

        # Create a test image
        test_data = np.random.rand(100, 100)
        hdu = fits.PrimaryHDU(test_data)
        hdu.writeto(self.test_file)

    def test_load_fits_image(self):
        """Test loading a valid FITS image"""
        # Load the image
        image = load_fits_image(self.test_file)

        # Verify the result
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.shape, (100, 100))
        self.assertTrue(np.all(np.isfinite(image)))

    def test_load_fits_image_invalid(self):
        """Test loading invalid FITS images"""
        # Test with non-existent file
        with self.assertRaises(FileNotFoundError):
            load_fits_image('non_existent.fits')

        # Test with invalid FITS file
        invalid_file = os.path.join(self.temp_dir, 'invalid.fits')
        with open(invalid_file, 'w') as f:
            f.write('This is not a FITS file')

        with self.assertRaises(Exception):
            load_fits_image(invalid_file)

    def test_calculate_center_of_mass(self):
        """Test center of mass calculation"""
        from src.common.utils import calculate_center_of_mass

        # Test case 1: Single bright pixel
        image1 = np.zeros((100, 100))
        image1[50, 50] = 1.0
        com_y, com_x = calculate_center_of_mass(image1)
        self.assertEqual(com_y, 50)
        self.assertEqual(com_x, 50)

        # Test case 2: Multiple bright pixels
        image2 = np.zeros((100, 100))
        image2[25:30, 25:30] = 1.0
        com_y, com_x = calculate_center_of_mass(image2)
        self.assertEqual(com_y, 27)  # Center of the 5x5 square
        self.assertEqual(com_x, 27)

        # Test case 3: No bright pixels (should return geometric center)
        image3 = np.zeros((100, 100))
        com_y, com_x = calculate_center_of_mass(image3)
        self.assertEqual(com_y, 50)
        self.assertEqual(com_x, 50)

        # Test case 4: Asymmetric distribution
        image4 = np.zeros((100, 100))
        image4[20:30, 20:30] = 0.5
        image4[70:80, 70:80] = 1.0
        com_y, com_x = calculate_center_of_mass(image4)
        # Should be closer to the brighter region
        self.assertTrue(com_y > 50)
        self.assertTrue(com_x > 50)

    def tearDown(self):
        # Clean up temporary files
        shutil.rmtree(self.temp_dir)

if __name__ == '__main__':
    unittest.main()