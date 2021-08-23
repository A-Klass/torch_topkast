# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

# kudos: https://stackoverflow.com/a/16084844
exec(open('_version.py').read())

requirements = [
    'matplotlib',
    'numpy',
    'sklearn',
    'torch_scatter',
    'torch_sparse']


tests_require = ['unittest']

setup(
    name='torch_topkast',
    version=__version__,
    description='Pytorch Implementation for Top-KAST',
    long_description=readme,
    python_requires='>=3.6',
    install_requires=requirements,
    tests_rquire=tests_require,
    author='Lisa Wimmer, Sven Lorenz, Andreas Klaß',
    # author_email='was soll hier rein?',
    url='https://github.com/A-Klass/appl_deepl',
#     license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

# upload to PyPI by: 
# python setup.py register sdist upload
