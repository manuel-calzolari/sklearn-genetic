import re
from io import open
from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the version
with open(path.join(here, 'genetic_selection', '__init__.py'), encoding='utf8') as f:
    version = re.search(r"__version__ = '(.*?)'", f.read()).group(1)

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='sklearn-genetic',
    version=version,
    description='Genetic feature selection module for scikit-learn',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/manuel-calzolari/sklearn-genetic',
    download_url='https://github.com/manuel-calzolari/sklearn-genetic/releases',
    author='Manuel Calzolari',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=['scikit-learn>=0.23', 'deap>=1.0.2', 'numpy', 'multiprocess', 'tqdm'],
)
