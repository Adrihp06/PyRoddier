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

from src.utils.image_preprocessing import load_fits_image

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
        with open(invalid_file,         'w') as f:
            f.write('This is not a FITS file')

        with self.assertRaises(Exception):
            load_fits_image(invalid_file)

    def tearDown(self):
        # Clean up temporary files
        shutil.rmtree(self.temp_dir)

if __name__ == '__main__':
    unittest.main()