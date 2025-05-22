
#!/usr/bin/env python3
"""
Videous Project Reorganizer

This script automates the reorganization of the Videous & Friends project codebase
according to best practices for Python project structure.
"""

import os
import shutil
import re
import glob
import configparser
import sys
import traceback
from datetime import datetime

# Check for required dependencies
try:
    import yaml
    from pathlib import Path
    print("All required dependencies are installed.")
except ImportError as e:
    print(f"Error: Missing dependency - {e}")
    print("Please install required dependencies: pip install pyyaml")
    sys.exit(1)

def create_directory_structure(base_dir):
    """Create the new directory structure."""
    print("Creating new directory structure...")
    
    # Define the directories to create
    directories = [
        "src/core",
        "src/pegasus",
        "src/chef",
        "src/starleague",
        "src/utils",
        "models",
        "tests",
        "scripts",
        "configs",
        "docs",
        "output",
        "temp"
    ]
    
    # Create each directory
    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"  Created: {dir_path}")
    
    # Create __init__.py files for Python packages
    for root, dirs, files in os.walk(os.path.join(base_dir, "src")):
        init_file = os.path.join(root, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write("# Videous package\n")
            print(f"  Created: {os.path.relpath(init_file, base_dir)}")

def move_core_files(base_dir):
    """Move core files to their appropriate locations."""
    print("\nMoving core files to new structure...")
    
    # Define file mappings: source -> destination
    file_mappings = {
        # Core modules
        "./main/pegasusAlgos.py": "src/core/algorithms.py",
        
        # Pegasus files
        "./main/main.py": "src/pegasus/app.py",
        "./main/VideousSport.py": "src/pegasus/sport.py",
        
        # CHEF files
        "./Videous CHEF/videousfinale.py": "src/chef/main.py",
        "./Videous CHEF/video_effects.py": "src/chef/effects.py",
        "./Videous CHEF/midascontroller.py": "src/chef/depth_controller.py",
        "./Videous CHEF/BGREPLACEVIDEOS.py": "src/chef/background_replacement.py",
        
        # StarLeague files
        "./main/StarLeague.py": "src/starleague/main.py",
        
        # Utility scripts
        "./upload_to_s3.sh": "scripts/upload_to_s3.sh",
    }
    
    # Move each file
    for source, dest in file_mappings.items():
        source_path = os.path.join(base_dir, source)
        dest_path = os.path.join(base_dir, dest)
        
        if os.path.exists(source_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(source_path, dest_path)
            print(f"  Moved: {source} -> {dest}")
        else:
            print(f"  Warning: Source file not found: {source}")

def move_models(base_dir):
    """Move model files to the models directory."""
    print("\nMoving model files...")
    
    # Find all model files
    model_extensions = ['.pt', '.pth', '.onnx', '.pb', '.h5']
    model_files = []
    
    for root, _, files in os.walk(base_dir):
        for file in files:
            if any(file.endswith(ext) for ext in model_extensions):
                model_files.append(os.path.join(root, file))
    
    # Move each model file
    for model_file in model_files:
        rel_path = os.path.relpath(model_file, base_dir)
        dest_path = os.path.join(base_dir, "models", os.path.basename(model_file))
        
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(model_file, dest_path)
        print(f"  Moved model: {rel_path} -> models/{os.path.basename(model_file)}")

def convert_config_to_yaml(base_dir):
    """Convert config.ini to YAML format."""
    print("\nConverting configuration files to YAML...")
    
    config_files = glob.glob(os.path.join(base_dir, "**/config.ini"), recursive=True)
    
    for config_file in config_files:
        rel_path = os.path.relpath(config_file, base_dir)
        
        try:
            # Parse the INI file
            config = configparser.ConfigParser()
            config.read(config_file)
            
            # Convert to dictionary
            config_dict = {section: dict(config[section]) for section in config.sections()}
            
            # Write as YAML
            yaml_path = os.path.join(base_dir, "configs", f"{os.path.basename(os.path.dirname(config_file))}.yaml")
            os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
            
            with open(yaml_path, 'w') as yaml_file:
                yaml.dump(config_dict, yaml_file, default_flow_style=False)
            
            print(f"  Converted: {rel_path} -> configs/{os.path.basename(os.path.dirname(config_file))}.yaml")
        except Exception as e:
            print(f"  Error converting {rel_path}: {e}")

def create_package_structure(base_dir):
    """Create proper Python package structure with __init__.py files."""
    print("\nCreating Python package structure...")
    
    # Walk through src directory and create __init__.py files
    for root, dirs, files in os.walk(os.path.join(base_dir, "src")):
        init_file = os.path.join(root, "__init__.py")
        
        # Skip if __init__.py already exists
        if os.path.exists(init_file):
            continue
        
        # Create the __init__.py file
        with open(init_file, 'w') as f:
            package_name = os.path.basename(root)
            f.write(f'"""\n{package_name.capitalize()} package for Videous project.\n"""\n\n')
        
        print(f"  Created: {os.path.relpath(init_file, base_dir)}")

def clean_up_redundant_files(base_dir):
    """Clean up redundant and backup files."""
    print("\nCleaning up redundant files...")
    
    # Patterns for files to clean up
    patterns = [
        "**/*.py.bak",
        "**/*.py.backup",
        "**/highlight_*_*TEMP_MPY_wvf_snd.mp4"
    ]
    
    # Find and list redundant files
    redundant_files = []
    for pattern in patterns:
        redundant_files.extend(glob.glob(os.path.join(base_dir, pattern), recursive=True))
    
    if not redundant_files:
        print("  No redundant files found.")
        return
        
    # Move redundant files to a backup directory
    backup_dir = os.path.join(base_dir, "backup_redundant_files")
    os.makedirs(backup_dir, exist_ok=True)
    
    for file_path in redundant_files:
        try:
            rel_path = os.path.relpath(file_path, base_dir)
            backup_path = os.path.join(backup_dir, rel_path)
            
            # Create directory structure in backup
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            # Move the file
            shutil.move(file_path, backup_path)
            print(f"  Moved to backup: {rel_path}")
        except Exception as e:
            print(f"  Error moving {os.path.basename(file_path)}: {e}")

def create_documentation(base_dir):
    """Create basic documentation files."""
    print("\nCreating documentation files...")
    
    # Create main README.md
    readme_path = os.path.join(base_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write("""# Videous & Friends

A comprehensive video editing and processing platform.

## Components

- **Pegasus Editor**: Video highlight detection and editing
- **Videous CHEF**: Advanced video processing and effects
- **StarLeague**: Video analysis and enhancement

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python -m src.pegasus.app`

## Media Files

Media files are stored in Amazon S3:
- Bucket: `videous-media-files`
- Path: `s3://videous-media-files/videos/`

## Documentation

See the `docs` directory for detailed documentation.
""")
    print(f"  Created: README.md")
    
    # Create component-specific READMEs
    components = {
        "src/pegasus": "Pegasus Editor",
        "src/chef": "Videous CHEF",
        "src/starleague": "StarLeague"
    }
    
    for path, name in components.items():
        component_readme = os.path.join(base_dir, path, "README.md")
        with open(component_readme, 'w') as f:
            f.write(f"""# {name}

## Overview

This component is part of the Videous & Friends project.

## Features

- Feature 1
- Feature 2
- Feature 3

## Usage

python
# Example usage code
""")
        print(f"  Created: {path}/README.md")

def create_requirements_file(base_dir):
    """Create a consolidated requirements.txt file."""
    print("\nCreating consolidated requirements.txt...")
    
    # Find all requirements files
    req_files = glob.glob(os.path.join(base_dir, "**/requirements.txt"), recursive=True)
    
    if not req_files:
        print("  No requirements.txt files found.")
        return
        
    # Collect all requirements
    all_requirements = set()
    for req_file in req_files:
        try:
            with open(req_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        all_requirements.add(line)
        except Exception as e:
            print(f"  Error reading {os.path.basename(req_file)}: {e}")
    
    # Write consolidated requirements
    with open(os.path.join(base_dir, "requirements.txt"), 'w') as f:
        f.write("# Consolidated requirements for Videous & Friends project\n\n")
        for req in sorted(all_requirements):
            f.write(f"{req}\n")
    
    print(f"  Created: requirements.txt with {len(all_requirements)} packages")

def create_gitignore(base_dir):
    """Create a comprehensive .gitignore file."""
    print("\nCreating .gitignore file...")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/
pegasus_env_py310/

# IDE files
.idea/
.vscode/
*.swp
*.swo

# OS specific files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Media files
*.mp4
*.mov
*.wav
*.mp3
temp_*.wav
highlight_*_*TEMP_MPY_wvf_snd.mp4

# Output directories
output*/
logs/

# Temporary files
temp/
.s3_uploads/

# Backup files
*.bak
*.backup
backup_redundant_files/
"""
    
    with open(os.path.join(base_dir, ".gitignore"), 'w') as f:
        f.write(gitignore_content)
    
    print(f"  Created: .gitignore")

def update_import_statements(base_dir):
    """Update import statements in Python files."""
    print("\nUpdating import statements...")
    
    # Map of old imports to new imports
    import_mappings = {
        "import pegasusAlgos": "from src.core import algorithms",
        "from pegasusAlgos import": "from src.core.algorithms import",
        "import VideousSport": "from src.pegasus import sport",
        "from VideousSport import": "from src.pegasus.sport import",
        "import StarLeague": "from src.starleague import main",
        "from StarLeague import": "from src.starleague.main import",
        "import video_effects": "from src.chef import effects",
        "from video_effects import": "from src.chef.effects import",
        "import midascontroller": "from src.chef import depth_controller",
        "from midascontroller import": "from src.chef.depth_controller import",
        "import BGREPLACEVIDEOS": "from src.chef import background_replacement",
        "from BGREPLACEVIDEOS import": "from src.chef.background_replacement import",
    }
    
    # Find all Python files in src directory
    python_files = glob.glob(os.path.join(base_dir, "src/**/*.py"), recursive=True)
    
    if not python_files:
        print("  No Python files found in src directory.")
        return
        
    updated_count = 0
    for py_file in python_files:
        try:
            with open(py_file, 'r') as f:
                content = f.read()
            
            # Apply import replacements
            modified = False
            for old_import, new_import in import_mappings.items():
                if old_import in content:
                    content = content.replace(old_import, new_import)
                    modified = True
            
            if modified:
                with open(py_file, 'w') as f:
                    f.write(content)
                print(f"  Updated imports in: {os.path.relpath(py_file, base_dir)}")
                updated_count += 1
        except Exception as e:
            print(f"  Error updating imports in {os.path.basename(py_file)}: {e}")
    
    if updated_count == 0:
        print("  No import statements needed updating.")

def create_s3_integration_module(base_dir):
    """Create a dedicated S3 integration module."""
    print("\nCreating S3 integration module...")
    
    s3_module_path = os.path.join(base_dir, "src/utils/s3_manager.py")
    os.makedirs(os.path.dirname(s3_module_path), exist_ok=True)
    
    with open(s3_module_path, 'w') as f:
        f.write('''"""
S3 Media Manager for Videous & Friends

This module provides utilities for managing media files in Amazon S3.
"""

import os
import boto3
from botocore.exceptions import ClientError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3MediaManager:
    """Manages media files in Amazon S3."""
    
    def __init__(self, bucket_name="videous-media-files", prefix="videos/"):
        """Initialize the S3 media manager.
        
        Args:
            bucket_name (str): The S3 bucket name
            prefix (str): The prefix (folder) within the bucket
        """
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.s3_client = boto3.client('s3')
    
    def list_media_files(self):
        """List all media files in the S3 bucket.
        
        Returns:
            list: List of media file names
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.prefix
            )
            
            if 'Contents' not in response:
                return []
                
            return [item['Key'].replace(self.prefix, '') for item in response['Contents']]
        except ClientError as e:
            logger.error(f"Error listing S3 files: {e}")
            return []
    
    def download_file(self, filename, local_path=None):
        """Download a file from S3.
        
        Args:
            filename (str): The name of the file in S3
            local_path (str, optional): Local path to save the file.
                                       If None, saves to current directory.
        
        Returns:
            str: Path to the downloaded file or None if failed
        """
        if local_path is None:
            local_path = os.path.basename(filename)
            
        s3_key = f"{self.prefix}{filename}"
        
        try:
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"Downloaded {filename} to {local_path}")
            return local_path
        except ClientError as e:
            logger.error(f"Error downloading {filename}: {e}")
            return None
    
    def upload_file(self, local_path, s3_filename=None):
        """Upload a file to S3.
        
        Args:
            local_path (str): Path to the local file
            s3_filename (str, optional): Name to use in S3.
                                        If None, uses the basename of local_path.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(local_path):
            logger.error(f"File not found: {local_path}")
            return False
            
        if s3_filename is None:
            s3_filename = os.path.basename(local_path)
            
        s3_key = f"{self.prefix}{s3_filename}"
        
        try:
            # Determine content type based on file extension
            content_type = "video/mp4"  # Default
            if local_path.lower().endswith('.mov'):
                content_type = "video/quicktime"
            elif local_path.lower().endswith('.wav'):
                content_type = "audio/wav"
                
            self.s3_client.upload_file(
                local_path, 
                self.bucket_name, 
                s3_key,
                ExtraArgs={
                    'ContentType': content_type,
                    'StorageClass': 'STANDARD'
                }
            )
            logger.info(f"Uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Error uploading {local_path}: {e}")
            return False
    
    def delete_file(self, filename):
        """Delete a file from S3.
        
        Args:
            filename (str): Name of the file to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        s3_key = f"{self.prefix}{filename}"
        
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            logger.info(f"Deleted s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Error deleting {filename}: {e}")
            return False


# Example usage
if __name__ == "__main__":
    s3_manager = S3MediaManager()
    
    # List files
    files = s3_manager.list_media_files()
    print(f"Found {len(files)} files in S3")
''')
    
    print(f"  Created: src/utils/s3_manager.py")

def create_config_manager(base_dir):
    """Create a configuration manager module."""
    print("\nCreating configuration manager...")
    
    config_manager_path = os.path.join(base_dir, "src/utils/config_manager.py")
    os.makedirs(os.path.dirname(config_manager_path), exist_ok=True)
    
    with open(config_manager_path, 'w') as f:
        f.write('''"""
Configuration Manager for Videous & Friends

This module provides utilities for loading and managing configuration.
"""

import os
import yaml
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self, config_dir=None):
        """Initialize the configuration manager.
        
        Args:
            config_dir (str, optional): Directory containing config files.
                                       If None, uses the default configs directory.
        """
        if config_dir is None:
            # Default to the configs directory in the project root
            self.config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs")
        else:
            self.config_dir = config_dir
            
        self.configs = {}
    
    def load_config(self, component_name):
        """Load configuration for a specific component.
        
        Args:
            component_name (str): Name of the component (e.g., 'pegasus', 'chef')
            
        Returns:
            dict: Configuration dictionary or empty dict if not found
        """
        config_path = os.path.join(self.config_dir, f"{component_name}.yaml")
        
        if not os.path.exists(config_path):
            logger.warning(f"Configuration file not found: {config_path}")
            return {}
            
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.configs[component_name] = config
                logger.info(f"Loaded configuration for {component_name}")
                return config
        except Exception as e:
            logger.error(f"Error loading configuration for {component_name}: {e}")
            return {}
    
    def get_config(self, component_name, section=None, key=None, default=None):
        """Get configuration value.
        
        Args:
            component_name (str): Name of the component
            section (str, optional): Section name within the config
            key (str, optional): Key within the section
            default: Default value if not found
            
        Returns:
            The configuration value or default if not found
        """
        if component_name not in self.configs:
            self.load_config(component_name)
            
        config = self.configs.get(component_name, {})
        
        if section is None:
            return config
            
        section_data = config.get(section, {})
        
        if key is None:
            return section_data
            
        return section_data.get(key, default)
    
    def save_config(self, component_name, config_data):
        """Save configuration for a component.
        
        Args:
            component_name (str): Name of the component
            config_data (dict): Configuration data to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        config_path = os.path.join(self.config_dir, f"{component_name}.yaml")
        
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
                
            self.configs[component_name] = config_data
            logger.info(f"Saved configuration for {component_name}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration for {component_name}: {e}")
            return False


# Example usage
if __name__ == "__main__":
    config_manager = ConfigManager()
    
    # Load configuration
    pegasus_config = config_manager.load_config("pegasus")
    
    # Get specific values
    audio_weight = config_manager.get_config("pegasus", "Weights", "audio_rms", 0.5)
    print(f"Audio weight: {audio_weight}")
''')
    
    print(f"  Created: src/utils/config_manager.py")

def create_setup_script(base_dir):
    """Create a setup.py script."""
    print("\nCreating setup.py...")
    
    setup_path = os.path.join(base_dir, "setup.py")
    
    with open(setup_path, 'w') as f:
        f.write('''"""
Setup script for Videous & Friends project.
"""

from setuptools import setup, find_packages

setup(
    name="videous",
    version="0.1.0",
    description="A comprehensive video editing and processing platform",
    author="Videous Team",
    author_email="info@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Core dependencies will be read from requirements.txt
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "pegasus=pegasus.app:main",
            "chef=chef.main:main",
            "starleague=starleague.main:main",
        ],
    },
)
''')
    
    print(f"  Created: setup.py")

def create_test_structure(base_dir):
    """Create basic test structure."""
    print("\nCreating test structure...")
    
    test_components = ["core", "pegasus", "chef", "starleague", "utils"]
    
    for component in test_components:
        test_dir = os.path.join(base_dir, "tests", component)
        os.makedirs(test_dir, exist_ok=True)
        
        # Create __init__.py
        with open(os.path.join(test_dir, "__init__.py"), "w") as f:
            f.write(f'"""\nTests for {component} module.\n"""\n')
        
        # Create a basic test file
        with open(os.path.join(test_dir, f"test_{component}.py"), "w") as f:
            f.write(f'''"""
Basic tests for {component} module.
"""

import unittest

class Test{component.capitalize()}(unittest.TestCase):
    """Test cases for {component} module."""
    
    def setUp(self):
        """Set up test fixtures."""
        pass
    
    def tearDown(self):
        """Tear down test fixtures."""
        pass
    
    def test_example(self):
        """Example test case."""
        self.assertEqual(1 + 1, 2)


if __name__ == "__main__":
    unittest.main()
''')
        
        print(f"  Created: tests/{component}/test_{component}.py")
    
    # Create main test runner
    with open(os.path.join(base_dir, "tests", "run_tests.py"), "w") as f:
        f.write('''"""
Test runner for Videous & Friends project.
"""

import unittest
import sys
import os

def run_all_tests():
    """Run all tests in the tests directory."""
    # Add the src directory to the path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(os.path.dirname(__file__))
    
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
''')
    
    print(f"  Created: tests/run_tests.py")

def main():
    """Main function to reorganize the project."""
    try:
        # Get the base directory (current directory)
        base_dir = os.getcwd()
        
        print(f"Script starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current working directory: {base_dir}")
        print(f"Reorganizing Videous project in: {base_dir}")
        print("=" * 50)
        
        # Confirm with user
        response = input("This will reorganize your project. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return
        
        # Create backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{base_dir}_backup_{timestamp}"
        print(f"\nCreating backup at: {backup_dir}")
        shutil.copytree(base_dir, backup_dir, ignore=shutil.ignore_patterns('.git'))
        print(f"Backup created successfully at: {backup_dir}")
        
        # Execute reorganization steps
        create_directory_structure(base_dir)
        move_core_files(base_dir)
        move_models(base_dir)
        convert_config_to_yaml(base_dir)
        create_package_structure(base_dir)
        clean_up_redundant_files(base_dir)
        create_documentation(base_dir)
        create_requirements_file(base_dir)
        create_gitignore(base_dir)
        
        # Additional functionality
        update_import_statements(base_dir)
        create_s3_integration_module(base_dir)
        create_config_manager(base_dir)
        create_setup_script(base_dir)
        create_test_structure(base_dir)
        
        print("\n" + "=" * 50)
        print("Project reorganization complete!")
        print(f"A backup of your original project structure is available at: {backup_dir}")
        print("\nNext steps:")
        print("1. Review the new structure and make any necessary adjustments")
        print("2. Update import statements in Python files")
        print("3. Test the reorganized codebase")
        print("4. Commit the changes to version control")
        print(f"\nScript completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\nError during reorganization: {e}")
        print("\nStack trace:")
        traceback.print_exc()
        print("\nReorganization failed. Please check the error message above.")
        sys.exit(1)

if __name__ == "__main__":
    main()