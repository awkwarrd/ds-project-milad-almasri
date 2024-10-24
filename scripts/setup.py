from setuptools import setup, find_packages

setup(
    name='transformers',
    version='0.2',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
    ],
)