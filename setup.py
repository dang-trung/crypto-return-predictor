"""Setuptools-based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

import pathlib

from setuptools import setup, find_packages

wd = pathlib.Path(__file__).parent.resolve()
long_description = (wd / 'README.md').read_text(encoding='utf-8')
with open(wd / 'requirements.txt') as f:
    requirements = f.readlines()

setup(
    name='crypto-return-predictor',
    version='0.0.0',
    description='Predicts cryptocurrency returns with sentiment-based '
                'features',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/dang-trung/crypto-return-predictor/',
    author='Trung Dang',
    author_email='dangtrung96@gmail.com',
    package_dir={'': 'crypto_return_predictor'},
    packages=find_packages(include=['crypto_return_predictor']),
    package_data={'': ['data/final_dataset.csv']},
    python_requires='>=3.5, <4',
    install_requires=requirements,
)
