import unittest
import numpy as np
from scipy.fft import fftshift, fft2, ifft2
from scipy.ndimage import center_of_mass, shift

# Add the src directory to the Python path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.core.roddier import calculate_wavefront
from src.core.zernike import fit_zernike
from src.core.interferometry import calculate_interferogram

class TestRoddierCalculations(unittest.TestCase):
    def setUp(self):
        # Create test images
        self.size = 100
        self.intra_image = np.zeros((self.size, self.size))
        self.extra_image = np.zeros((self.size, self.size))

        # Add a bright spot in the center
        center = self.size // 2
        self.intra_image[center-5:center+5, center-5:center+5] = 1.0
        self.extra_image[center-5:center+5, center-5:center+5] = 1.0

    def test_calculate_wavefront(self):
        """Test wavefront calculation"""
        # Create test data
        delta_I_norm = (self.extra_image - self.intra_image) / (self.extra_image + self.intra_image)
        annular_mask = np.ones_like(delta_I_norm, dtype=bool)
        dz_mm = 1.0

        # Calculate wavefront
        wavefront = calculate_wavefront(delta_I_norm, annular_mask, dz_mm)

        # Verify the result
        self.assertEqual(wavefront.shape, delta_I_norm.shape)
        self.assertTrue(np.all(np.isfinite(wavefront)))

    def test_fit_zernike(self):
        """Test Zernike polynomial fitting"""
        # Create test wavefront
        wavefront = np.random.rand(self.size, self.size)
        annular_mask = np.ones_like(wavefront, dtype=bool)
        R_out = self.size / 2
        center = (self.size // 2, self.size // 2)
        max_order = 6

        # Fit Zernike polynomials
        coeffs, base = fit_zernike(wavefront, annular_mask, R_out, center, max_order)

        # Verify the result
        self.assertEqual(len(coeffs), (max_order + 1) * (max_order + 2) // 2)
        self.assertEqual(base.shape[0], len(coeffs))
        self.assertEqual(base.shape[1:], wavefront.shape)

    def test_calculate_interferogram(self):
        """Test interferogram calculation"""
        # Create test data
        reference_intensity = 1.0
        reference_frequency = 0.1
        wavefront = np.random.rand(self.size, self.size)
        annular_mask = np.ones_like(wavefront, dtype=bool)

        # Calculate interferogram
        interferogram = calculate_interferogram(
            reference_intensity,
            reference_frequency,
            wavefront,
            annular_mask
        )

        # Verify the result
        self.assertEqual(interferogram.shape, wavefront.shape)
        self.assertTrue(np.all(np.isfinite(interferogram)))
        self.assertTrue(np.all(interferogram >= 0))  # Intensity should be non-negative

if __name__ == '__main__':
    unittest.main()