from setuptools import setup, find_packages

setup(
    name='price_predictions',
    version="1.0",
    author="Milad",
    author_email="heregoesnothingowo@gmail.com",
    url="https://github.com/awkwarrd/ds-project-milad-almasri.git",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn==1.5.1',
        'category_encoders',
        'Boruta',
        'hyperopt',
    ],
    python_requires=">=3.6",
)