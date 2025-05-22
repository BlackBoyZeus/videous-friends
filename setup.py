"""
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
