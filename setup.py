# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

# kudos: https://stackoverflow.com/a/16084844
exec(open('appl_deepl/_version.py').read())
 
setup(
    name='appl_deepl_topkast',
    version=__version__,
    description='Pytorch Implementation for Top-KAST',
    long_description=readme,
    author='Lisa Wimmer, Sven Lorenz, Andreas Klaß',
    # author_email='was soll hier rein?',
    url='https://github.com/A-Klass/appl_deepl',
#     license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

# upload to PyPI by: 
# python setup.py register sdist upload
