"""
Setup Script
============

This module provides the setup configuration for the `lps_sp` package.

"""

import os
import setuptools

package_dir = os.path.dirname(os.path.abspath(__file__))
packages = setuptools.find_packages()

if not packages:
    raise RuntimeError("No Python package found")

package_name = packages[0]

# Setup configuration for the package
setuptools.setup(
    name=package_name,
    version='0.0.1',
    packages=setuptools.find_packages(),
    description='A Signal processing basic library.',
    author='Fabio Oliveira',
    license='CC BY-NC-SA 4.0',
    install_requires=[
        # List your project dependencies here
    ],
    # Uncomment and specify package data if needed
    # package_data={package_name: ['info/*.csv']},
)
