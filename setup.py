from setuptools import setup
from setuptools import find_packages

setup(name='sklearn-genetic',
      version='0.1',
      description='Genetic feature selection module for scikit-learn',
      url='http://github.com/manuel-calzolari/sklearn-genetic',
      author='Manuel Calzolari',
      author_email='',
      license='GPLv3',
      install_requires=['scikit-learn>=0.18', 'deap>=1.0.2'],
      packages=find_packages())
