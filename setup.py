# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py
import os
from setuptools import setup, find_packages

NAME = 'biome'

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md')) as f:
    readme = f.read()

with open(os.path.join(here, 'LICENSE')) as f:
    license = f.read()

about = {}
with open(os.path.join(here, NAME, '__version__.py')) as f:
    exec(f.read(), about)

with open(os.path.join(here, 'requirements.txt')) as f:
    required = f.read().splitlines()

setup(
    name=NAME,
    version=about['__version__'],
    install_requires=required,
    description='Biome plaftform for deep learning models',
    long_description=readme,
    author='recognai',
    author_email='contact@recogn.ai',
    url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
