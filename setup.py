from setuptools import setup, find_packages

setup(
    name='sklearn-genetic',
    version='0.1',
    description='Genetic feature selection module for scikit-learn',
    url='https://github.com/manuel-calzolari/sklearn-genetic',
    download_url='https://github.com/manuel-calzolari/sklearn-genetic/archive/0.1.tar.gz',
    author='Manuel Calzolari',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    packages=find_packages(),
    python_requires='>=2.7',
    install_requires=['scikit-learn>=0.18', 'deap>=1.0.2'],
)
