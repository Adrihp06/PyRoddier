import os
from pathlib import Path

def ensure_config_dirs():
    """Ensure that the configuration directories exist."""
    # Get the home directory
    home_dir = Path.home()

    # Define the base config directory
    config_dir = home_dir / '.pyroddier'

    config_file = config_dir / 'config.json'
    # Define subdirectories
    telescope_dir = config_dir / 'telescopes'

    # Create directories if they don't exist
    config_dir.mkdir(parents=True, exist_ok=True)
    telescope_dir.mkdir(parents=True, exist_ok=True)

    return {
        'config_dir': str(config_dir),
        'telescope_dir': str(telescope_dir)
    }

def get_config_paths():
    """Get the paths to the configuration directories."""
    return ensure_config_dirs()