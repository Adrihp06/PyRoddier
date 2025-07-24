# Copyright (c) 2025 Adrián Hernández Padrón
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import unittest
import json
import os
import tempfile
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.core.telescope import TelescopeParams

class TestTelescopeParams(unittest.TestCase):
    """Test suite for TelescopeParams class"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_params = {
            'apertura': 900.0,
            'focal': 7200.0,
            'pixel_scale': 15.0,
            'max_order': 23,
            'threshold': 0.5,
            'binning': 1
        }
        
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test files"""
        # Remove all files in test directory
        for filename in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, filename))
        os.rmdir(self.test_dir)

    def test_telescope_params_creation(self):
        """Test basic TelescopeParams creation"""
        params = TelescopeParams(
            apertura=900.0,
            focal=7200.0,
            pixel_scale=15.0
        )
        
        # Check required parameters
        self.assertEqual(params.apertura, 900.0)
        self.assertEqual(params.focal, 7200.0)
        self.assertEqual(params.pixel_scale, 15.0)
        
        # Check default parameters
        self.assertEqual(params.max_order, 23)
        self.assertEqual(params.threshold, 0.5)
        self.assertEqual(params.binning, 1)

    def test_telescope_params_with_custom_defaults(self):
        """Test TelescopeParams creation with custom default values"""
        params = TelescopeParams(
            apertura=1200.0,
            focal=8000.0,
            pixel_scale=10.0,
            max_order=30,
            threshold=0.3,
            binning=2
        )
        
        self.assertEqual(params.apertura, 1200.0)
        self.assertEqual(params.focal, 8000.0)
        self.assertEqual(params.pixel_scale, 10.0)
        self.assertEqual(params.max_order, 30)
        self.assertEqual(params.threshold, 0.3)
        self.assertEqual(params.binning, 2)

    def test_from_dict_complete(self):
        """Test creating TelescopeParams from complete dictionary"""
        params = TelescopeParams.from_dict(self.test_params)
        
        self.assertEqual(params.apertura, 900.0)
        self.assertEqual(params.focal, 7200.0)
        self.assertEqual(params.pixel_scale, 15.0)
        self.assertEqual(params.max_order, 23)
        self.assertEqual(params.threshold, 0.5)
        self.assertEqual(params.binning, 1)

    def test_from_dict_partial(self):
        """Test creating TelescopeParams from partial dictionary"""
        partial_params = {
            'apertura': 800.0,
            'focal': 6000.0,
            'pixel_scale': 20.0
            # Missing max_order, threshold, binning
        }
        
        params = TelescopeParams.from_dict(partial_params)
        
        # Should use provided values
        self.assertEqual(params.apertura, 800.0)
        self.assertEqual(params.focal, 6000.0)
        self.assertEqual(params.pixel_scale, 20.0)
        
        # Should use defaults for missing values
        self.assertEqual(params.max_order, 23)
        self.assertEqual(params.threshold, 0.5)
        self.assertEqual(params.binning, 1)

    def test_from_dict_empty(self):
        """Test creating TelescopeParams from empty dictionary"""
        params = TelescopeParams.from_dict({})
        
        # Should use defaults for all values
        self.assertEqual(params.apertura, 0.0)
        self.assertEqual(params.focal, 0.0)
        self.assertEqual(params.pixel_scale, 0.0)
        self.assertEqual(params.max_order, 23)
        self.assertEqual(params.threshold, 0.5)
        self.assertEqual(params.binning, 1)

    def test_to_dict(self):
        """Test converting TelescopeParams to dictionary"""
        params = TelescopeParams(
            apertura=900.0,
            focal=7200.0,
            pixel_scale=15.0,
            max_order=25,
            threshold=0.4,
            binning=2
        )
        
        result_dict = params.to_dict()
        
        expected_dict = {
            'apertura': 900.0,
            'focal': 7200.0,
            'pixel_scale': 15.0,
            'max_order': 25,
            'threshold': 0.4,
            'binning': 2
        }
        
        self.assertEqual(result_dict, expected_dict)

    def test_roundtrip_dict_conversion(self):
        """Test that from_dict and to_dict are inverses"""
        original_params = TelescopeParams(
            apertura=1000.0,
            focal=8000.0,
            pixel_scale=12.0,
            max_order=20,
            threshold=0.6,
            binning=3
        )
        
        # Convert to dict and back
        params_dict = original_params.to_dict()
        reconstructed_params = TelescopeParams.from_dict(params_dict)
        
        # Should be identical
        self.assertEqual(original_params.apertura, reconstructed_params.apertura)
        self.assertEqual(original_params.focal, reconstructed_params.focal)
        self.assertEqual(original_params.pixel_scale, reconstructed_params.pixel_scale)
        self.assertEqual(original_params.max_order, reconstructed_params.max_order)
        self.assertEqual(original_params.threshold, reconstructed_params.threshold)
        self.assertEqual(original_params.binning, reconstructed_params.binning)

    def test_save_to_json_valid(self):
        """Test saving TelescopeParams to JSON file"""
        params = TelescopeParams(
            apertura=900.0,
            focal=7200.0,
            pixel_scale=15.0
        )
        
        test_file = os.path.join(self.test_dir, 'test_params.json')
        
        # Save should succeed
        result = params.save_to_json(test_file)
        self.assertTrue(result)
        
        # File should exist
        self.assertTrue(os.path.exists(test_file))
        
        # File should contain correct JSON
        with open(test_file, 'r') as f:
            saved_data = json.load(f)
        
        expected_data = {
            'apertura': 900.0,
            'focal': 7200.0,
            'pixel_scale': 15.0,
            'max_order': 23,
            'threshold': 0.5,
            'binning': 1
        }
        
        self.assertEqual(saved_data, expected_data)

    def test_save_to_json_invalid_path(self):
        """Test saving TelescopeParams to invalid path"""
        params = TelescopeParams(apertura=900.0, focal=7200.0, pixel_scale=15.0)
        
        # Try to save to non-existent directory
        invalid_path = '/nonexistent/directory/test.json'
        
        # Save should fail
        result = params.save_to_json(invalid_path)
        self.assertFalse(result)

    def test_from_json_valid_file(self):
        """Test loading TelescopeParams from valid JSON file"""
        # Create test JSON file
        test_file = os.path.join(self.test_dir, 'valid_params.json')
        with open(test_file, 'w') as f:
            json.dump(self.test_params, f)
        
        # Load should succeed
        params = TelescopeParams.from_json(test_file)
        
        self.assertIsNotNone(params)
        self.assertEqual(params.apertura, 900.0)
        self.assertEqual(params.focal, 7200.0)
        self.assertEqual(params.pixel_scale, 15.0)
        self.assertEqual(params.max_order, 23)
        self.assertEqual(params.threshold, 0.5)
        self.assertEqual(params.binning, 1)

    def test_from_json_nonexistent_file(self):
        """Test loading TelescopeParams from non-existent file"""
        nonexistent_file = os.path.join(self.test_dir, 'nonexistent.json')
        
        # Should return None
        params = TelescopeParams.from_json(nonexistent_file)
        self.assertIsNone(params)

    def test_from_json_invalid_json(self):
        """Test loading TelescopeParams from invalid JSON file"""
        # Create invalid JSON file
        test_file = os.path.join(self.test_dir, 'invalid.json')
        with open(test_file, 'w') as f:
            f.write('{ invalid json content')
        
        # Should return None
        params = TelescopeParams.from_json(test_file)
        self.assertIsNone(params)

    def test_from_json_partial_data(self):
        """Test loading TelescopeParams from JSON with partial data"""
        partial_data = {
            'apertura': 1200.0,
            'focal': 9000.0
            # Missing pixel_scale and others
        }
        
        test_file = os.path.join(self.test_dir, 'partial.json')
        with open(test_file, 'w') as f:
            json.dump(partial_data, f)
        
        params = TelescopeParams.from_json(test_file)
        
        self.assertIsNotNone(params)
        self.assertEqual(params.apertura, 1200.0)
        self.assertEqual(params.focal, 9000.0)
        self.assertEqual(params.pixel_scale, 0.0)  # Default value
        self.assertEqual(params.max_order, 23)  # Default value

    def test_roundtrip_json_conversion(self):
        """Test that save_to_json and from_json are inverses"""
        original_params = TelescopeParams(
            apertura=1100.0,
            focal=8500.0,
            pixel_scale=18.0,
            max_order=26,
            threshold=0.35,
            binning=4
        )
        
        test_file = os.path.join(self.test_dir, 'roundtrip.json')
        
        # Save and load
        save_result = original_params.save_to_json(test_file)
        self.assertTrue(save_result)
        
        loaded_params = TelescopeParams.from_json(test_file)
        
        # Should be identical
        self.assertIsNotNone(loaded_params)
        self.assertEqual(original_params.apertura, loaded_params.apertura)
        self.assertEqual(original_params.focal, loaded_params.focal)
        self.assertEqual(original_params.pixel_scale, loaded_params.pixel_scale)
        self.assertEqual(original_params.max_order, loaded_params.max_order)
        self.assertEqual(original_params.threshold, loaded_params.threshold)
        self.assertEqual(original_params.binning, loaded_params.binning)

    def test_telescope_params_realistic_values(self):
        """Test TelescopeParams with realistic telescope configurations"""
        # Test small refractor
        small_refractor = TelescopeParams(
            apertura=100.0,  # 100mm aperture
            focal=1000.0,    # f/10
            pixel_scale=1.0,  # 1 arcsec/pixel
            binning=1
        )
        
        self.assertEqual(small_refractor.apertura, 100.0)
        self.assertEqual(small_refractor.focal, 1000.0)
        
        # Test large observatory telescope  
        large_telescope = TelescopeParams(
            apertura=8000.0,  # 8m aperture
            focal=24000.0,    # f/3
            pixel_scale=0.1,  # 0.1 arcsec/pixel
            binning=2
        )
        
        self.assertEqual(large_telescope.apertura, 8000.0)
        self.assertEqual(large_telescope.focal, 24000.0)
        self.assertEqual(large_telescope.pixel_scale, 0.1)

    def test_telescope_params_validation_implicit(self):
        """Test that TelescopeParams accepts various parameter ranges"""
        # Test zero values (should be allowed)
        zero_params = TelescopeParams(apertura=0.0, focal=0.0, pixel_scale=0.0)
        self.assertEqual(zero_params.apertura, 0.0)
        
        # Test negative values (should be allowed - validation is external)
        negative_params = TelescopeParams(apertura=-100.0, focal=-1000.0, pixel_scale=-1.0)
        self.assertEqual(negative_params.apertura, -100.0)
        
        # Test very large values
        large_params = TelescopeParams(apertura=100000.0, focal=1000000.0, pixel_scale=1000.0)
        self.assertEqual(large_params.apertura, 100000.0)

if __name__ == '__main__':
    unittest.main()