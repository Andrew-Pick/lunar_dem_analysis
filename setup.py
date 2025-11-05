from setuptools import setup, find_packages

setup(
    name='lunar_crater_analysis',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'rasterio',
        'scipy',
        'matplotlib',
        'pyproj'
    ],
    description='A toolkit for analyzing lunar crater DEMs.',
    author='Andrew-Pick',
)