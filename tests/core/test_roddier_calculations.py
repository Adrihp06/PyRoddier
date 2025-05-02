# Copyright (c) 2025 Adrián Hernández Padrón
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import numpy as np
from scipy.fft import fftshift, fft2, ifft2
from scipy.ndimage import center_of_mass, shift
import unittest

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
        size = 100
        x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
        r = np.sqrt(x**2 + y**2)

        # Create Gaussian spots with different intensities
        self.intra_image = np.exp(-r**2 / 0.2**2)
        self.extra_image = 1.2 * np.exp(-r**2 / 0.2**2)  # Slightly brighter
        self.size = size

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
        # Create test wavefront with known properties
        x, y = np.meshgrid(np.linspace(-1, 1, self.size), np.linspace(-1, 1, self.size))
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        # Create a simple wavefront (defocus term)
        wavefront = 2 * r**2 - 1  # Simple defocus term
        annular_mask = r <= 1  # Circular mask

        R_out = self.size / 2
        center = (self.size // 2, self.size // 2)
        max_order = 6

        # Fit Zernike polynomials
        coeffs, base = fit_zernike(wavefront, annular_mask, R_out, center, max_order)

        # Verify the result
        expected_terms = 6  # Number of Zernike terms for max_order=6
        self.assertEqual(len(coeffs), expected_terms)
        self.assertEqual(base.shape[0], expected_terms)
        self.assertEqual(base.shape[1:], wavefront.shape)

    def test_fit_zernike_complex(self):
        """Test Zernike polynomial fitting with complex wavefront"""
        # Create test wavefront with multiple Zernike terms
        x, y = np.meshgrid(np.linspace(-1, 1, self.size), np.linspace(-1, 1, self.size))
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        # Create a complex wavefront with known Zernike terms
        # Usando términos más simples y más grandes para el test
        wavefront = np.zeros_like(r)
        mask = r <= 1
        wavefront[mask] = (
            2.0 * r[mask]**2 +  # Término cuadrático (relacionado con defocus)
            1.0 * r[mask] * np.cos(theta[mask])  # Tilt X
        )

        # Fit Zernike polynomials
        R_out = self.size / 2
        center = (self.size // 2, self.size // 2)
        max_order = 6

        coeffs, base = fit_zernike(wavefront, mask, R_out, center, max_order)

        # Verify the result
        self.assertEqual(len(coeffs), 6)  # Number of Zernike terms for max_order=6

        # Check that we have significant coefficients
        # Los coeficientes deberían ser más pequeños debido a la normalización
        coeff_magnitudes = np.abs(coeffs)
        self.assertTrue(np.any(coeff_magnitudes > 0.1),
                       f"No hay coeficientes mayores que 0.1. Máximo: {np.max(coeff_magnitudes)}")

        # Verificar que hay al menos dos coeficientes significativos
        significant_coeffs = coeff_magnitudes[coeff_magnitudes > 0.05]
        self.assertGreaterEqual(len(significant_coeffs), 2,
                              f"Solo hay {len(significant_coeffs)} coeficientes significativos")

    def test_calculate_wavefront_edge_cases(self):
        """Test wavefront calculation with edge cases"""
        # Test case 1: All zeros
        delta_I_norm = np.zeros((self.size, self.size))
        annular_mask = np.ones_like(delta_I_norm, dtype=bool)
        dz_mm = 1.0

        wavefront = calculate_wavefront(delta_I_norm, annular_mask, dz_mm)
        self.assertEqual(wavefront.shape, delta_I_norm.shape)
        self.assertTrue(np.all(np.isfinite(wavefront)))
        self.assertTrue(np.allclose(wavefront, 0))

        # Test case 2: All ones
        delta_I_norm = np.ones((self.size, self.size))
        wavefront = calculate_wavefront(delta_I_norm, annular_mask, dz_mm)
        self.assertEqual(wavefront.shape, delta_I_norm.shape)
        self.assertTrue(np.all(np.isfinite(wavefront)))

        # Test case 3: Invalid mask (all zeros)
        delta_I_norm = np.random.rand(self.size, self.size)
        annular_mask = np.zeros_like(delta_I_norm, dtype=bool)
        wavefront = calculate_wavefront(delta_I_norm, annular_mask, dz_mm)
        # The function should return zeros for invalid mask
        self.assertTrue(np.allclose(wavefront, 0))

    def test_calculate_interferogram_edge_cases(self):
        """Test interferogram calculation with edge cases"""
        # Test case 1: Zero wavefront
        wavefront = np.zeros((self.size, self.size))
        annular_mask = np.ones_like(wavefront, dtype=bool)
        reference_frequency = 0.1
        reference_intensity = 1.0

        interferogram = calculate_interferogram(
            wavefront,
            reference_frequency,
            reference_intensity,
            annular_mask
        )

        self.assertEqual(interferogram.shape, wavefront.shape)
        self.assertTrue(np.all(np.isfinite(interferogram)))
        self.assertTrue(np.all(interferogram >= 0))

        # Test case 2: Invalid mask (all zeros)
        wavefront = np.random.rand(self.size, self.size)
        annular_mask = np.zeros_like(wavefront, dtype=bool)
        interferogram = calculate_interferogram(
            wavefront,
            reference_frequency,
            reference_intensity,
            annular_mask
        )
        self.assertTrue(np.all(np.isnan(interferogram)))

    def test_calculate_interferogram(self):
        """Test interferogram calculation"""
        # Create test data
        wavefront = np.random.rand(self.size, self.size)
        annular_mask = np.ones_like(wavefront, dtype=bool)
        reference_frequency = 0.1
        reference_intensity = 1.0

        # Calculate interferogram
        interferogram = calculate_interferogram(
            wavefront,
            reference_frequency,
            reference_intensity,
            annular_mask
        )

        # Verify the result
        self.assertEqual(interferogram.shape, wavefront.shape)
        self.assertTrue(np.all(np.isfinite(interferogram)))
        self.assertTrue(np.all(interferogram >= 0))  # Intensity should be non-negative

if __name__ == '__main__':
    unittest.main()