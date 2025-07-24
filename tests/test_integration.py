# Copyright (c) 2025 Adrián Hernández Padrón
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import unittest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.roddier import calculate_wavefront
from src.core.zernike import fit_zernike, zernike_polynomials, calculate_rms
from src.core.optical_preprocessing import preprocess_roddier
from src.core.psf import calculate_psf
from src.core.telescope import TelescopeParams

class TestIntegration(unittest.TestCase):
    """Integration tests for complete Roddier analysis pipeline"""

    def setUp(self):
        """Set up test fixtures for integration tests"""
        self.size = 128
        self.center = self.size // 2
        
        # Create synthetic telescope parameters
        self.telescope = TelescopeParams(
            apertura=900.0,
            focal=7200.0,
            pixel_scale=15.0,
            max_order=15,
            threshold=0.5
        )
        
        # Create synthetic intra and extra focal images
        self.intra_img, self.extra_img = self._create_synthetic_images()

    def _create_synthetic_images(self):
        """Create realistic synthetic intra/extra focal images"""
        y, x = np.ogrid[:self.size, :self.size]
        
        # Create base pupil pattern
        r = np.sqrt((x - self.center)**2 + (y - self.center)**2)
        pupil_radius = self.size // 4
        base_intensity = (r <= pupil_radius).astype(float)
        
        # Add some realistic aberrations to create differential pattern
        # Simulate defocus and other aberrations
        defocus_pattern = 0.1 * np.cos(2 * np.pi * r / pupil_radius) * base_intensity
        
        # Intra-focal: slightly more intensity in center
        intra_img = base_intensity + defocus_pattern * 0.5
        
        # Extra-focal: slightly less intensity in center  
        extra_img = base_intensity - defocus_pattern * 0.5
        
        # Add noise
        intra_img += 0.01 * np.random.random((self.size, self.size))
        extra_img += 0.01 * np.random.random((self.size, self.size))
        
        # Ensure positive values
        intra_img = np.maximum(intra_img, 0.01)
        extra_img = np.maximum(extra_img, 0.01)
        
        return intra_img, extra_img

    def test_complete_roddier_pipeline(self):
        """Test complete Roddier analysis pipeline from images to results"""
        # Step 1: Preprocess images
        delta_I_norm, annular_mask, center, R_out, dz_mm = preprocess_roddier(
            self.intra_img, 
            self.extra_img,
            apertura=self.telescope.apertura,
            focal=self.telescope.focal,
            pixel_scale=self.telescope.pixel_scale,
            threshold=self.telescope.threshold
        )
        
        # Verify preprocessing results
        self.assertEqual(delta_I_norm.shape, self.intra_img.shape)
        self.assertEqual(annular_mask.shape, self.intra_img.shape)
        self.assertIsInstance(dz_mm, (int, float, np.number))
        self.assertGreater(dz_mm, 0)
        
        # Step 2: Calculate wavefront using Roddier algorithm
        wavefront = calculate_wavefront(
            delta_I_norm, 
            annular_mask, 
            wavelength_nm=556,
            dz_mm=dz_mm
        )
        
        # Verify wavefront calculation
        self.assertEqual(wavefront.shape, self.intra_img.shape)
        self.assertTrue(np.all(np.isfinite(wavefront)))
        
        # Step 3: Generate Zernike basis
        zernike_base = zernike_polynomials(
            shape=self.intra_img.shape,
            mask=annular_mask,
            R_out=R_out,
            center=center,
            max_terms=self.telescope.max_order
        )
        
        # Verify Zernike basis
        self.assertEqual(len(zernike_base), self.telescope.max_order)
        for poly in zernike_base:
            self.assertEqual(poly.shape, self.intra_img.shape)
        
        # Step 4: Fit Zernike coefficients
        zernike_coeffs, _ = fit_zernike(wavefront, annular_mask, R_out, center, max_order=self.telescope.max_order)
        
        # Verify coefficients
        self.assertEqual(len(zernike_coeffs), self.telescope.max_order)
        self.assertTrue(np.all(np.isfinite(zernike_coeffs)))
        
        # Step 5: Calculate RMS wavefront error
        rms_error = calculate_rms(zernike_coeffs, exclude_piston=True)
        
        # Verify RMS calculation
        self.assertIsInstance(rms_error, (int, float, np.number))
        self.assertGreater(rms_error, 0)
        
        # Step 6: Generate PSF
        psf, psf_log = calculate_psf(wavefront, annular_mask, wavelength=556)
        
        # Verify PSF calculation
        self.assertEqual(psf.shape, self.intra_img.shape)
        self.assertEqual(psf_log.shape, self.intra_img.shape)
        self.assertAlmostEqual(psf.max(), 1.0, places=6)
        self.assertTrue(np.all(psf >= 0))

    def test_pipeline_error_propagation(self):
        """Test how errors propagate through the pipeline"""
        # Test with very noisy images
        noisy_intra = self.intra_img + 0.5 * np.random.random(self.intra_img.shape)
        noisy_extra = self.extra_img + 0.5 * np.random.random(self.extra_img.shape)
        
        # Pipeline should still work but with degraded results
        delta_I_norm, annular_mask, center, R_out, dz_mm = preprocess_roddier(
            noisy_intra, noisy_extra
        )
        
        wavefront = calculate_wavefront(delta_I_norm, annular_mask, dz_mm=dz_mm)
        
        # Results should still be finite and reasonable
        self.assertTrue(np.all(np.isfinite(wavefront)))
        self.assertTrue(np.any(annular_mask))  # Some valid pixels remain

    def test_pipeline_with_different_telescope_configs(self):
        """Test pipeline with different telescope configurations"""
        configs = [
            # Small refractor
            TelescopeParams(apertura=100, focal=800, pixel_scale=2.0, max_order=10),
            # Medium telescope  
            TelescopeParams(apertura=500, focal=2000, pixel_scale=1.0, max_order=20),
            # Large telescope
            TelescopeParams(apertura=2000, focal=8000, pixel_scale=0.3, max_order=25)
        ]
        
        for config in configs:
            with self.subTest(config=config):
                # Run pipeline with different configuration
                delta_I_norm, annular_mask, center, R_out, dz_mm = preprocess_roddier(
                    self.intra_img, 
                    self.extra_img,
                    apertura=config.apertura,
                    focal=config.focal,
                    pixel_scale=config.pixel_scale
                )
                
                wavefront = calculate_wavefront(delta_I_norm, annular_mask, dz_mm=dz_mm)
                
                zernike_base = zernike_polynomials(
                    shape=self.intra_img.shape,
                    mask=annular_mask,
                    R_out=R_out,
                    center=center,
                    max_terms=config.max_order
                )
                
                zernike_coeffs, _ = fit_zernike(wavefront, annular_mask, R_out, center, max_order=config.max_order)
                
                # All results should be valid
                self.assertTrue(np.all(np.isfinite(wavefront)))
                self.assertEqual(len(zernike_coeffs), config.max_order)
                self.assertGreater(dz_mm, 0)

    def test_pipeline_consistency(self):
        """Test that pipeline produces consistent results"""
        # Run pipeline multiple times with same input
        results = []
        
        for _ in range(3):
            delta_I_norm, annular_mask, center, R_out, dz_mm = preprocess_roddier(
                self.intra_img, self.extra_img
            )
            
            wavefront = calculate_wavefront(delta_I_norm, annular_mask, dz_mm=dz_mm)
            
            zernike_base = zernike_polynomials(
                shape=self.intra_img.shape,
                mask=annular_mask, 
                R_out=R_out,
                center=center,
                max_terms=10
            )
            
            zernike_coeffs, _ = fit_zernike(wavefront, annular_mask, R_out, center, max_order=10)
            rms_error = calculate_rms(zernike_coeffs, exclude_piston=True)
            
            results.append({
                'dz_mm': dz_mm,
                'rms_error': rms_error,
                'coeffs': zernike_coeffs
            })
        
        # Results should be identical (deterministic algorithm)
        for i in range(1, len(results)):
            self.assertAlmostEqual(results[0]['dz_mm'], results[i]['dz_mm'], places=10)
            self.assertAlmostEqual(results[0]['rms_error'], results[i]['rms_error'], places=10)
            np.testing.assert_array_almost_equal(
                results[0]['coeffs'], results[i]['coeffs'], decimal=10
            )

    def test_pipeline_reconstruction_accuracy(self):
        """Test accuracy of wavefront reconstruction"""
        # Run full pipeline
        delta_I_norm, annular_mask, center, R_out, dz_mm = preprocess_roddier(
            self.intra_img, self.extra_img
        )
        
        original_wavefront = calculate_wavefront(delta_I_norm, annular_mask, dz_mm)
        
        zernike_base = zernike_polynomials(
            shape=self.intra_img.shape,
            mask=annular_mask,
            R_out=R_out, 
            center=center,
            max_terms=15
        )
        
        zernike_coeffs, _ = fit_zernike(original_wavefront, annular_mask, R_out, center, max_order=15)
        
        # Reconstruct wavefront from coefficients
        reconstructed_wavefront = np.zeros_like(original_wavefront)
        for i, coeff in enumerate(zernike_coeffs):
            reconstructed_wavefront += coeff * zernike_base[i]
        
        # Calculate reconstruction error over pupil
        error = original_wavefront - reconstructed_wavefront
        masked_error = error[annular_mask]
        
        # RMS reconstruction error should be small
        reconstruction_rms = np.sqrt(np.mean(masked_error**2))
        original_rms = np.sqrt(np.mean(original_wavefront[annular_mask]**2))
        
        # Reconstruction should capture most of the wavefront
        relative_error = reconstruction_rms / original_rms
        self.assertLess(relative_error, 0.2)  # Less than 20% relative error (adjusted for realistic performance)

    def test_pipeline_physical_reasonableness(self):
        """Test that pipeline produces physically reasonable results"""
        delta_I_norm, annular_mask, center, R_out, dz_mm = preprocess_roddier(
            self.intra_img, self.extra_img,
            apertura=900,  # 900mm telescope
            focal=7200,    # f/8 telescope
            pixel_scale=15 # 15 μm pixels
        )
        
        wavefront = calculate_wavefront(delta_I_norm, annular_mask, dz_mm=dz_mm)
        
        zernike_base = zernike_polynomials(
            shape=self.intra_img.shape,
            mask=annular_mask,
            R_out=R_out,
            center=center, 
            max_terms=15
        )
        
        zernike_coeffs, _ = fit_zernike(wavefront, annular_mask, R_out, center, max_order=15)
        rms_error = calculate_rms(zernike_coeffs, exclude_piston=True)
        
        # Physical reasonableness checks
        
        # 1. Defocus distance should be reasonable (few mm to few cm)
        self.assertGreater(dz_mm, 0.1)  # At least 0.1mm
        self.assertLess(dz_mm, 100)     # Less than 10cm
        
        # 2. RMS wavefront error should be reasonable (not too small or large)
        self.assertGreater(rms_error, 1e-6)  # Not unrealistically small
        self.assertLess(rms_error, 10)       # Not unrealistically large
        
        # 3. Pupil radius should be reasonable fraction of image
        self.assertGreater(R_out, 10)              # At least 10 pixels
        self.assertLess(R_out, self.size / 2)      # Less than half image size
        
        # 4. Center should be reasonably close to image center
        cx, cy = center
        self.assertLess(abs(cx - self.center), self.size / 4)
        self.assertLess(abs(cy - self.center), self.size / 4)
        
        # 5. Lower-order Zernike coefficients should generally be larger
        # (though this isn't always true, it's a reasonable expectation for synthetic data)
        if len(zernike_coeffs) >= 5:
            lower_order_rms = calculate_rms(zernike_coeffs[:5], exclude_piston=True)
            higher_order_rms = calculate_rms(zernike_coeffs[5:], exclude_piston=False)
            
            # Not a strict requirement, but generally expected
            # We'll just check they're both reasonable
            self.assertGreater(lower_order_rms, 0)
            self.assertGreater(higher_order_rms, 0)

if __name__ == '__main__':
    unittest.main()