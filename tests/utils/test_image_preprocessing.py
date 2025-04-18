import unittest
import numpy as np
from astropy.io import fits
import os
import tempfile

# Add the src directory to the Python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.utils.image_preprocessing import load_fits_image

class TestImagePreprocessing(unittest.TestCase):
    def setUp(self):
        # Create a temporary FITS file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_fits_path = os.path.join(self.temp_dir, 'test.fits')

        # Create a test image
        test_data = np.random.rand(100, 100)
        hdu = fits.PrimaryHDU(test_data)
        hdu.writeto(self.test_fits_path)

    def tearDown(self):
        # Clean up temporary files
        if os.path.exists(self.test_fits_path):
            os.remove(self.test_fits_path)
        os.rmdir(self.temp_dir)

    def test_load_fits_image(self):
        """Test loading a FITS image"""
        # Test successful loading
        image_data = load_fits_image(self.test_fits_path)
        self.assertIsNotNone(image_data)
        self.assertEqual(image_data.shape, (100, 100))
        self.assertEqual(image_data.dtype, np.float64)

        # Test loading non-existent file
        with self.assertRaises(Exception):
            load_fits_image('non_existent.fits')

    def test_load_fits_image_invalid(self):
        """Test loading invalid FITS files"""
        # Create an invalid FITS file
        invalid_path = os.path.join(self.temp_dir, 'invalid.fits')
        with open(invalid_path, 'w') as f:
            f.write('This is not a FITS file')

        with self.assertRaises(Exception):
            load_fits_image(invalid_path)

if __name__ == '__main__':
    unittest.main()