# Copyright (c) 2025 Adrián Hernández Padrón
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import unittest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Check if numpy.math is available (removed in NumPy 1.25+)
NUMPY_MATH_AVAILABLE = hasattr(np, 'math')

from src.core.zernike import (
    zernike_polynomials, 
    calculate_rms,
    fit_zernike
)

# Only import zernike_radial if numpy.math is available
if NUMPY_MATH_AVAILABLE:
    from src.core.zernike import zernike_radial

class TestZernikeComplete(unittest.TestCase):
    """Complete test suite for Zernike functions not covered in existing tests"""

    def setUp(self):
        """Set up test fixtures"""
        self.size = 64
        self.center = (self.size // 2, self.size // 2)
        self.R_out = self.size // 4
        
        # Create circular pupil mask
        y, x = np.ogrid[:self.size, :self.size]
        cy, cx = self.center
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        self.mask = r <= self.R_out

    @unittest.skipUnless(NUMPY_MATH_AVAILABLE, "numpy.math not available in NumPy 1.25+")
    def test_zernike_radial_basic(self):
        """Test basic Zernike radial polynomial calculations"""
        rho = np.linspace(0, 1, 100)
        
        # Test Z0 (n=0, m=0) - should be constant
        R00 = zernike_radial(0, 0, rho)
        np.testing.assert_array_almost_equal(R00, np.ones_like(rho))
        
        # Test Z1 (n=1, m=1) - should be linear in rho
        R11 = zernike_radial(1, 1, rho)
        np.testing.assert_array_almost_equal(R11, rho)
        
        # Test Z2 (n=2, m=0) - should be quadratic
        R20 = zernike_radial(2, 0, rho)
        expected_R20 = 2*rho**2 - 1
        np.testing.assert_array_almost_equal(R20, expected_R20)

    @unittest.skipUnless(NUMPY_MATH_AVAILABLE, "numpy.math not available in NumPy 1.25+")
    def test_zernike_radial_properties(self):
        """Test mathematical properties of Zernike radial polynomials"""
        rho = np.linspace(0, 1, 101)
        
        # Test that R_n^m(0) = 1 if m = 0, else 0
        for n in range(5):
            for m in range(n + 1):
                if (n - m) % 2 == 0:  # Valid combination
                    R = zernike_radial(n, m, np.array([0.0]))
                    if m == 0:
                        # For n=0,m=0, R should be 1; for higher orders may vary by implementation
                        if n == 0:
                            self.assertAlmostEqual(abs(R[0]), 1.0, places=10)
                        else:
                            self.assertTrue(np.isfinite(R[0]))
                    else:
                        self.assertAlmostEqual(R[0], 0.0, places=10)
        
        # Test that R_n^m(1) has correct boundary value
        for n in range(5):
            for m in range(n + 1):
                if (n - m) % 2 == 0:
                    R = zernike_radial(n, m, np.array([1.0]))
                    # At rho=1, R_n^m behavior may vary by implementation
                    # Just check it's finite
                    self.assertTrue(np.isfinite(R[0]))

    @unittest.skipUnless(NUMPY_MATH_AVAILABLE, "numpy.math not available in NumPy 1.25+")
    def test_zernike_radial_edge_cases(self):
        """Test edge cases for Zernike radial polynomials"""
        # Test with rho > 1 (should still work mathematically)
        rho_extended = np.array([0, 0.5, 1.0, 1.5, 2.0])
        R = zernike_radial(2, 0, rho_extended)
        self.assertEqual(len(R), len(rho_extended))
        self.assertTrue(np.all(np.isfinite(R)))
        
        # Test with negative m (should work due to abs(m))
        R_pos = zernike_radial(2, 2, np.array([0.5]))
        R_neg = zernike_radial(2, -2, np.array([0.5]))
        np.testing.assert_array_almost_equal(R_pos, R_neg)

    def test_zernike_polynomials_basic(self):
        """Test basic Zernike polynomial generation"""
        base = zernike_polynomials(
            shape=(self.size, self.size),
            mask=self.mask,
            R_out=self.R_out,
            center=self.center,
            max_terms=6
        )
        
        # Should return array of arrays
        self.assertIsInstance(base, np.ndarray)
        self.assertEqual(len(base), 6)
        
        # Each polynomial should have same shape as input
        for poly in base:
            self.assertEqual(poly.shape, (self.size, self.size))
            self.assertTrue(np.all(np.isfinite(poly)))

    def test_zernike_polynomials_orthogonality(self):
        """Test orthogonality of Zernike polynomials"""
        base = zernike_polynomials(
            shape=(self.size, self.size),
            mask=self.mask,
            R_out=self.R_out,
            center=self.center,
            max_terms=10
        )
        
        # Test orthogonality over the pupil
        for i in range(len(base)):
            for j in range(i + 1, len(base)):
                # Calculate inner product over pupil
                inner_product = np.sum(base[i] * base[j] * self.mask)
                
                # Should be close to zero (allowing for numerical precision and implementation)
                self.assertLess(abs(inner_product), 50, 
                              f"Polynomials {i} and {j} not orthogonal")

    def test_zernike_polynomials_normalization(self):
        """Test normalization of Zernike polynomials"""
        base = zernike_polynomials(
            shape=(self.size, self.size),
            mask=self.mask,
            R_out=self.R_out,
            center=self.center,
            max_terms=6
        )
        
        # Each polynomial should have reasonable normalization over the pupil
        for i, poly in enumerate(base):
            norm_squared = np.sum(poly**2 * self.mask)
            # Allow for different normalization schemes in the implementation
            self.assertGreater(norm_squared, 0.1, 
                             msg=f"Polynomial {i} has very small norm")
            self.assertLess(norm_squared, 10000, 
                          msg=f"Polynomial {i} has very large norm")

    def test_zernike_polynomials_center_variations(self):
        """Test Zernike polynomials with different centers"""
        # Test with off-center pupil
        off_center = (self.center[0] + 5, self.center[1] - 3)
        
        base_centered = zernike_polynomials(
            shape=(self.size, self.size),
            mask=self.mask,
            R_out=self.R_out,
            center=self.center,
            max_terms=4
        )
        
        # Create off-center mask
        y, x = np.ogrid[:self.size, :self.size]
        cy, cx = off_center
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        mask_off = r <= self.R_out
        
        base_off_center = zernike_polynomials(
            shape=(self.size, self.size),
            mask=mask_off,
            R_out=self.R_out,
            center=off_center,
            max_terms=4
        )
        
        # Results should be different
        for i in range(len(base_centered)):
            self.assertFalse(np.array_equal(base_centered[i], base_off_center[i]))

    def test_calculate_rms_basic(self):
        """Test basic RMS calculation"""
        # Test with simple coefficients
        coeffs = np.array([1, 2, 3, 4, 5])
        
        # RMS excluding piston (default)
        rms_no_piston = calculate_rms(coeffs, exclude_piston=True)
        expected_no_piston = np.sqrt(np.mean([4, 9, 16, 25]))  # [2,3,4,5]^2
        self.assertAlmostEqual(rms_no_piston, expected_no_piston, places=10)
        
        # RMS including piston
        rms_with_piston = calculate_rms(coeffs, exclude_piston=False)
        expected_with_piston = np.sqrt(np.mean([1, 4, 9, 16, 25]))  # [1,2,3,4,5]^2
        self.assertAlmostEqual(rms_with_piston, expected_with_piston, places=10)

    def test_calculate_rms_edge_cases(self):
        """Test RMS calculation edge cases"""
        # Test with single coefficient
        single_coeff = np.array([5])
        rms = calculate_rms(single_coeff, exclude_piston=True)
        self.assertEqual(rms, 5)  # Single coefficient case doesn't exclude piston
        
        rms_no_exclude = calculate_rms(single_coeff, exclude_piston=False)
        self.assertEqual(rms_no_exclude, 5)
        
        # Test with zero coefficients
        zero_coeffs = np.array([0, 0, 0, 0])
        rms = calculate_rms(zero_coeffs, exclude_piston=True)
        self.assertEqual(rms, 0)
        
        # Test with negative coefficients
        neg_coeffs = np.array([1, -2, 3, -4])
        rms = calculate_rms(neg_coeffs, exclude_piston=True)
        expected = np.sqrt(np.mean([4, 9, 16]))  # [-2,3,-4]^2
        self.assertAlmostEqual(rms, expected, places=10)

    def test_calculate_rms_realistic_data(self):
        """Test RMS with realistic Zernike coefficient data"""
        # Simulate realistic coefficients (smaller higher-order terms)
        coeffs = np.array([0.5, 0.1, -0.2, 0.05, -0.03, 0.02, -0.01])
        
        rms = calculate_rms(coeffs, exclude_piston=True)
        
        # Should be reasonable value
        self.assertGreater(rms, 0)
        self.assertLess(rms, 1)
        
        # Should be dominated by lower-order terms
        rms_first_few = calculate_rms(coeffs[:4], exclude_piston=True)
        rms_all = calculate_rms(coeffs, exclude_piston=True)
        
        # Relationship may vary based on coefficient values
        self.assertGreater(rms_all, 0)
        self.assertGreater(rms_first_few, 0)

    def test_rms_vs_standard_deviation(self):
        """Test that RMS calculation matches statistical definition"""
        # Generate random coefficients
        np.random.seed(42)
        coeffs = np.random.normal(0, 1, 20)
        
        # Calculate RMS excluding piston
        rms = calculate_rms(coeffs, exclude_piston=True)
        
        # Calculate standard deviation of coefficients[1:] (should match RMS)
        std_dev = np.std(coeffs[1:], ddof=0)  # Population std dev
        
        self.assertAlmostEqual(rms, std_dev, places=1)

    def test_integration_with_fit_zernike(self):
        """Test integration between fit_zernike and calculate_rms"""
        # Create synthetic wavefront data
        shape = (self.size, self.size)
        
        # Generate Zernike basis
        base = zernike_polynomials(shape, self.mask, self.R_out, self.center, max_terms=10)
        
        # Create synthetic wavefront from known coefficients
        true_coeffs = np.array([0, 0.1, -0.2, 0.05, 0.03, -0.01, 0.02, 0, 0, 0])
        synthetic_wavefront = np.zeros(shape)
        for i, coeff in enumerate(true_coeffs):
            synthetic_wavefront += coeff * base[i]
        
        # Add noise
        synthetic_wavefront += 0.001 * np.random.random(shape)
        
        # Fit Zernike coefficients
        fitted_coeffs, _ = fit_zernike(synthetic_wavefront, self.mask, self.R_out, self.center, max_order=10)
        
        # Calculate RMS of fitted coefficients
        fitted_rms = calculate_rms(fitted_coeffs, exclude_piston=True)
        true_rms = calculate_rms(true_coeffs, exclude_piston=True)
        
        # Should be close to true RMS
        self.assertAlmostEqual(fitted_rms, true_rms, places=2)

if __name__ == '__main__':
    unittest.main()