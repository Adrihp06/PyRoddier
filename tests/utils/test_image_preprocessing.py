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

from src.common.utils import load_fits_image, find_center, apply_mask, calculate_center_of_mass

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

    def test_find_center(self):
        """Test find_center function (wrapper around scipy's center_of_mass)"""
        # Test with simple case
        image = np.zeros((100, 100))
        image[30, 40] = 1.0
        
        cx, cy = find_center(image)
        
        # Should return coordinates close to the bright pixel
        self.assertAlmostEqual(cx, 40, delta=1)
        self.assertAlmostEqual(cy, 30, delta=1)
        
        # Test with symmetric distribution
        image2 = np.zeros((50, 50))
        image2[20:30, 20:30] = 1.0
        
        cx2, cy2 = find_center(image2)
        
        # Should return center of the bright region
        self.assertAlmostEqual(cx2, 24.5, delta=0.5)
        self.assertAlmostEqual(cy2, 24.5, delta=0.5)

    def test_apply_mask(self):
        """Test apply_mask function"""
        # Create test image and mask
        image = np.ones((10, 10)) * 5.0
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:8, 2:8] = True  # Square mask in center
        
        # Apply mask
        masked_image = apply_mask(image, mask)
        
        # Check that only masked region has values
        self.assertTrue(np.all(masked_image[mask] == 5.0))
        self.assertTrue(np.all(masked_image[~mask] == 0.0))
        
        # Test with float mask
        float_mask = np.zeros((10, 10))
        float_mask[2:8, 2:8] = 0.5  # Half intensity in center
        
        masked_image2 = apply_mask(image, float_mask)
        
        # Check values
        self.assertTrue(np.all(masked_image2[2:8, 2:8] == 2.5))
        self.assertTrue(np.all(masked_image2[:2, :] == 0.0))
        
        # Test with complex image
        complex_image = np.random.random((20, 20)) * 10
        binary_mask = np.random.choice([0, 1], (20, 20), p=[0.3, 0.7]).astype(bool)
        
        masked_complex = apply_mask(complex_image, binary_mask)
        
        # Check that masked regions are preserved, others are zero
        np.testing.assert_array_equal(masked_complex[binary_mask], complex_image[binary_mask])
        np.testing.assert_array_equal(masked_complex[~binary_mask], 0.0)

    def test_function_consistency(self):
        """Test consistency between find_center and calculate_center_of_mass"""
        # Create test image with clear peak
        image = np.zeros((60, 60))
        y, x = np.ogrid[:60, :60]
        image = np.exp(-((x-30)**2 + (y-25)**2) / (2*5**2))  # Gaussian peak at (30, 25)
        
        # Use both functions
        cx_find, cy_find = find_center(image)
        com_y_calc, com_x_calc = calculate_center_of_mass(image)
        
        # Results should be reasonably close (allowing for different algorithms)
        self.assertAlmostEqual(cx_find, com_x_calc, delta=2)
        self.assertAlmostEqual(cy_find, com_y_calc, delta=2)
        
        # Both should be close to the true center
        self.assertAlmostEqual(cx_find, 30, delta=2)
        self.assertAlmostEqual(cy_find, 25, delta=2)

    def tearDown(self):
        # Clean up temporary files
        shutil.rmtree(self.temp_dir)

if __name__ == '__main__':
    unittest.main()