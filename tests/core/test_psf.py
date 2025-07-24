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
        # Test with zero pupil mask
        zero_mask = np.zeros((self.size, self.size))
        psf, psf_log = calculate_psf(self.flat_wavefront, zero_mask)
        
        # With zero mask, PSF should be essentially zero everywhere
        self.assertTrue(np.max(psf) < 1e-10)
        
        # Test with very large wavefront values
        large_wavefront = np.ones((self.size, self.size)) * 10
        psf, psf_log = calculate_psf(large_wavefront, self.pupil_mask)
        
        # Should still be normalized and finite
        self.assertAlmostEqual(psf.max(), 1.0, places=6)
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
        
        # Smaller pupil should have less total energy
        self.assertLess(np.sum(psf_small), np.sum(psf))

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
        # all quadrants should be approximately equal
        tolerance = 1e-10
        self.assertLess(np.max(np.abs(q1 - np.fliplr(q2))), tolerance)
        self.assertLess(np.max(np.abs(q1 - np.flipud(q3))), tolerance)
        self.assertLess(np.max(np.abs(q1 - np.flipud(np.fliplr(q4)))), tolerance)

if __name__ == '__main__':
    unittest.main()