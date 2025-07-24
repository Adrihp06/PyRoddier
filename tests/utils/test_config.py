# Copyright (c) 2025 Adrián Hernández Padrón
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import unittest
import tempfile
import shutil
import os
import sys
from pathlib import Path
from unittest.mock import patch

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.common.config import ensure_config_dirs, get_config_paths

class TestConfig(unittest.TestCase):
    """Test suite for configuration management functions"""

    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory for testing
        self.temp_home = tempfile.mkdtemp()
        self.original_home = os.environ.get('HOME')

    def tearDown(self):
        """Clean up test fixtures"""
        # Restore original home directory
        if self.original_home:
            os.environ['HOME'] = self.original_home
        elif 'HOME' in os.environ:
            del os.environ['HOME']
        
        # Clean up temporary directory
        if os.path.exists(self.temp_home):
            shutil.rmtree(self.temp_home)

    @patch('pathlib.Path.home')
    def test_ensure_config_dirs_basic(self, mock_home):
        """Test basic functionality of ensure_config_dirs"""
        # Mock home directory
        mock_home.return_value = Path(self.temp_home)
        
        # Call function
        result = ensure_config_dirs()
        
        # Check that directories were created
        expected_config_dir = Path(self.temp_home) / '.pyroddier'
        expected_telescope_dir = expected_config_dir / 'telescopes'
        
        self.assertTrue(expected_config_dir.exists())
        self.assertTrue(expected_telescope_dir.exists())
        
        # Check return values
        self.assertEqual(result['config_dir'], str(expected_config_dir))
        self.assertEqual(result['telescope_dir'], str(expected_telescope_dir))

    @patch('pathlib.Path.home')
    def test_ensure_config_dirs_already_exist(self, mock_home):
        """Test ensure_config_dirs when directories already exist"""
        # Mock home directory
        mock_home.return_value = Path(self.temp_home)
        
        # Pre-create directories
        config_dir = Path(self.temp_home) / '.pyroddier'
        telescope_dir = config_dir / 'telescopes'
        config_dir.mkdir(parents=True, exist_ok=True)
        telescope_dir.mkdir(parents=True, exist_ok=True)
        
        # Add a test file to verify directories aren't overwritten
        test_file = config_dir / 'test.txt'
        test_file.write_text('test content')
        
        # Call function
        result = ensure_config_dirs()
        
        # Check that directories still exist and test file is preserved
        self.assertTrue(config_dir.exists())
        self.assertTrue(telescope_dir.exists())
        self.assertTrue(test_file.exists())
        self.assertEqual(test_file.read_text(), 'test content')
        
        # Check return values
        self.assertEqual(result['config_dir'], str(config_dir))
        self.assertEqual(result['telescope_dir'], str(telescope_dir))

    @patch('pathlib.Path.home')
    def test_ensure_config_dirs_nested_creation(self, mock_home):
        """Test that ensure_config_dirs creates nested directories properly"""
        # Mock home directory to a non-existent path
        mock_home.return_value = Path(self.temp_home) / 'non_existent' / 'user'
        
        # Call function
        result = ensure_config_dirs()
        
        # Check that nested directories were created
        expected_config_dir = Path(self.temp_home) / 'non_existent' / 'user' / '.pyroddier'
        expected_telescope_dir = expected_config_dir / 'telescopes'
        
        self.assertTrue(expected_config_dir.exists())
        self.assertTrue(expected_telescope_dir.exists())
        
        # Check return values
        self.assertEqual(result['config_dir'], str(expected_config_dir))
        self.assertEqual(result['telescope_dir'], str(expected_telescope_dir))

    @patch('pathlib.Path.home')
    def test_get_config_paths(self, mock_home):
        """Test get_config_paths function"""
        # Mock home directory
        mock_home.return_value = Path(self.temp_home)
        
        # Call function
        result = get_config_paths()
        
        # Should return same result as ensure_config_dirs
        expected_result = ensure_config_dirs()
        self.assertEqual(result, expected_result)
        
        # Check that directories were created
        config_dir = Path(self.temp_home) / '.pyroddier'
        telescope_dir = config_dir / 'telescopes'
        
        self.assertTrue(config_dir.exists())
        self.assertTrue(telescope_dir.exists())

    @patch('pathlib.Path.home')
    def test_config_dirs_are_absolute_paths(self, mock_home):
        """Test that returned paths are absolute"""
        # Mock home directory
        mock_home.return_value = Path(self.temp_home)
        
        # Call function
        result = ensure_config_dirs()
        
        # Check that paths are absolute
        self.assertTrue(Path(result['config_dir']).is_absolute())
        self.assertTrue(Path(result['telescope_dir']).is_absolute())

    @patch('pathlib.Path.home')
    def test_config_dirs_permissions(self, mock_home):
        """Test that created directories have correct permissions"""
        # Mock home directory
        mock_home.return_value = Path(self.temp_home)
        
        # Call function
        result = ensure_config_dirs()
        
        # Check that directories are readable and writable
        config_dir = Path(result['config_dir'])
        telescope_dir = Path(result['telescope_dir'])
        
        self.assertTrue(os.access(config_dir, os.R_OK))
        self.assertTrue(os.access(config_dir, os.W_OK))
        self.assertTrue(os.access(telescope_dir, os.R_OK))
        self.assertTrue(os.access(telescope_dir, os.W_OK))

    @patch('pathlib.Path.home')
    def test_multiple_calls_consistency(self, mock_home):
        """Test that multiple calls return consistent results"""
        # Mock home directory
        mock_home.return_value = Path(self.temp_home)
        
        # Call function multiple times
        result1 = ensure_config_dirs()
        result2 = ensure_config_dirs()
        result3 = get_config_paths()
        
        # All results should be identical
        self.assertEqual(result1, result2)
        self.assertEqual(result2, result3)
        
        # Directories should still exist and be the same
        self.assertTrue(Path(result1['config_dir']).exists())
        self.assertTrue(Path(result1['telescope_dir']).exists())

    @patch('pathlib.Path.home')
    def test_config_directory_structure(self, mock_home):
        """Test the complete directory structure created"""
        # Mock home directory
        mock_home.return_value = Path(self.temp_home)
        
        # Call function
        result = ensure_config_dirs()
        
        # Verify complete structure
        config_dir = Path(result['config_dir'])
        telescope_dir = Path(result['telescope_dir'])
        
        # Check that config_dir is a subdirectory of home
        self.assertTrue(str(config_dir).startswith(self.temp_home))
        self.assertEqual(config_dir.name, '.pyroddier')
        
        # Check that telescope_dir is a subdirectory of config_dir
        self.assertEqual(telescope_dir.parent, config_dir)
        self.assertEqual(telescope_dir.name, 'telescopes')

    @patch('pathlib.Path.home')
    def test_config_dirs_with_file_conflict(self, mock_home):
        """Test behavior when a file exists where directory should be created"""
        # Mock home directory
        mock_home.return_value = Path(self.temp_home)
        
        # Create a file where config directory should be
        conflict_file = Path(self.temp_home) / '.pyroddier'
        conflict_file.write_text('blocking file')
        
        # This should raise an exception due to the conflict
        with self.assertRaises(FileExistsError):
            ensure_config_dirs()

    @patch('pathlib.Path.home')
    def test_return_value_format(self, mock_home):
        """Test that return value has correct format and types"""
        # Mock home directory
        mock_home.return_value = Path(self.temp_home)
        
        # Call function
        result = ensure_config_dirs()
        
        # Check return value structure
        self.assertIsInstance(result, dict)
        self.assertIn('config_dir', result)
        self.assertIn('telescope_dir', result)
        self.assertEqual(len(result), 2)
        
        # Check that values are strings (not Path objects)
        self.assertIsInstance(result['config_dir'], str)
        self.assertIsInstance(result['telescope_dir'], str)
        
        # Check that both values are valid directory paths
        self.assertTrue(os.path.isdir(result['config_dir']))
        self.assertTrue(os.path.isdir(result['telescope_dir']))

if __name__ == '__main__':
    unittest.main()