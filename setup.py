from setuptools import setup, find_packages

setup(
    name="harmony",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'sentence-transformers',
        'torch',
        'jellyfish',
        'tqdm',
        'pytest',
        'psutil'
    ],
) 