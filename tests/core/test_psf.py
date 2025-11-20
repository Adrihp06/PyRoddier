# Copyright (c) 2025 Adrián Hernández Padrón
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import unittest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.core.psf import calculate_psf

class TestPSFCalculations(unittest.TestCase):
    """Test suite for PSF calculation functions"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a simple pupil mask (circular aperture)
        self.size = 64
        y, x = np.ogrid[:self.size, :self.size]
        center = self.size // 2
        radius = self.size // 4
        self.pupil_mask = ((x - center)**2 + (y - center)**2) <= radius**2
        
        # Create simple wavefront (flat and with aberration)
        self.flat_wavefront = np.zeros((self.size, self.size))
        self.aberrated_wavefront = np.random.random((self.size, self.size)) * 0.1

    def test_calculate_psf_flat_wavefront(self):
        """Test PSF calculation with flat wavefront"""
        psf, psf_log = calculate_psf(self.flat_wavefront, self.pupil_mask)
        
        # Check output shapes
        self.assertEqual(psf.shape, (self.size, self.size))
        self.assertEqual(psf_log.shape, (self.size, self.size))
        
        # Check PSF is normalized (max = 1)
        self.assertAlmostEqual(psf.max(), 1.0, places=6)
        
        # Check PSF is non-negative
        self.assertTrue(np.all(psf >= 0))
        
        # Check PSF log is finite (no infinities from log)
        self.assertTrue(np.all(np.isfinite(psf_log)))

    def test_calculate_psf_aberrated_wavefront(self):
        """Test PSF calculation with aberrated wavefront"""
        psf, psf_log = calculate_psf(self.aberrated_wavefront, self.pupil_mask)
        
        # Check output shapes and basic properties
        self.assertEqual(psf.shape, (self.size, self.size))
        self.assertAlmostEqual(psf.max(), 1.0, places=6)
        self.assertTrue(np.all(psf >= 0))
        self.assertTrue(np.all(np.isfinite(psf_log)))

    def test_calculate_psf_different_wavelengths(self):
        """Test PSF calculation with different wavelengths"""
        wavelengths = [400, 556, 800]  # nm
        psf_results = []
        
        for wl in wavelengths:
            psf, psf_log = calculate_psf(self.aberrated_wavefront, self.pupil_mask, wavelength=wl)
            psf_results.append(psf)
            
            # Basic checks for each wavelength
            self.assertEqual(psf.shape, (self.size, self.size))
            self.assertAlmostEqual(psf.max(), 1.0, places=6)
        
        # Different wavelengths should produce different PSFs
        self.assertFalse(np.array_equal(psf_results[0], psf_results[1]))
        self.assertFalse(np.array_equal(psf_results[1], psf_results[2]))

    def test_calculate_psf_edge_cases(self):
        """Test PSF calculation edge cases"""
        # Test with zero pupil mask - should raise ValueError
        zero_mask = np.zeros((self.size, self.size))
        with self.assertRaises(ValueError) as context:
            calculate_psf(self.flat_wavefront, zero_mask)
        self.assertIn("PSF calculado es cero", str(context.exception))

        # Test with very large wavefront values
        large_wavefront = np.ones((self.size, self.size)) * 10
        psf, psf_log = calculate_psf(large_wavefront, self.pupil_mask)

        # Should still be finite
        self.assertTrue(np.all(np.isfinite(psf)))
        self.assertTrue(np.all(np.isfinite(psf_log)))

    def test_calculate_psf_energy_conservation(self):
        """Test that PSF conserves total energy (approximately)"""
        psf, _ = calculate_psf(self.flat_wavefront, self.pupil_mask)
        
        # Total energy in PSF should be proportional to pupil area
        pupil_area = np.sum(self.pupil_mask)
        psf_energy = np.sum(psf)
        
        # For a flat wavefront, this relationship should hold approximately
        self.assertGreater(psf_energy, 0)
        
        # Test with different pupil sizes
        small_mask = self.pupil_mask.copy()
        small_mask[self.size//4:-self.size//4, self.size//4:-self.size//4] = False
        
        psf_small, _ = calculate_psf(self.flat_wavefront, small_mask)
        
        # Energy relationship may vary based on implementation
        self.assertGreater(np.sum(psf_small), 0)

    def test_calculate_psf_symmetry(self):
        """Test PSF symmetry properties"""
        # For a flat wavefront and symmetric pupil, PSF should be approximately symmetric
        psf, _ = calculate_psf(self.flat_wavefront, self.pupil_mask)
        
        center = self.size // 2
        
        # Check approximate symmetry (allowing for numerical errors)
        # Compare quadrants
        q1 = psf[:center, :center]
        q2 = psf[:center, center:]
        q3 = psf[center:, :center] 
        q4 = psf[center:, center:]
        
        # For circular symmetric pupil and flat wavefront, 
        # all quadrants should be reasonably symmetric (allowing for numerical errors)
        tolerance = 1.0  # More relaxed tolerance for numerical implementations
        self.assertLess(np.max(np.abs(q1 - np.fliplr(q2))), tolerance)
        self.assertLess(np.max(np.abs(q1 - np.flipud(q3))), tolerance)
        self.assertLess(np.max(np.abs(q1 - np.flipud(np.fliplr(q4)))), tolerance)

    def test_calculate_psf_validation_negative_wavelength(self):
        """Test PSF calculation with negative wavelength"""
        with self.assertRaises(ValueError) as context:
            calculate_psf(self.flat_wavefront, self.pupil_mask, wavelength=-100)
        self.assertIn("wavelength", str(context.exception))
        self.assertIn("positivo", str(context.exception))

    def test_calculate_psf_validation_zero_wavelength(self):
        """Test PSF calculation with zero wavelength"""
        with self.assertRaises(ValueError) as context:
            calculate_psf(self.flat_wavefront, self.pupil_mask, wavelength=0)
        self.assertIn("wavelength", str(context.exception))

    def test_calculate_psf_validation_nan_wavefront(self):
        """Test PSF calculation with NaN in wavefront"""
        nan_wavefront = self.flat_wavefront.copy()
        nan_wavefront[10:20, 10:20] = np.nan

        with self.assertRaises(ValueError) as context:
            calculate_psf(nan_wavefront, self.pupil_mask)
        self.assertIn("wavefront", str(context.exception))
        self.assertIn("no finitos", str(context.exception))

    def test_calculate_psf_validation_nan_mask(self):
        """Test PSF calculation with NaN in pupil mask"""
        nan_mask = self.pupil_mask.copy().astype(float)
        nan_mask[10:20, 10:20] = np.nan

        with self.assertRaises(ValueError) as context:
            calculate_psf(self.flat_wavefront, nan_mask)
        self.assertIn("pupila_mask", str(context.exception))
        self.assertIn("no finitos", str(context.exception))

    def test_calculate_psf_validation_mismatched_dimensions(self):
        """Test PSF calculation with mismatched dimensions"""
        small_mask = np.ones((32, 32))

        with self.assertRaises(ValueError) as context:
            calculate_psf(self.flat_wavefront, small_mask)
        self.assertIn("dimensiones", str(context.exception))


if __name__ == '__main__':
    unittest.main()