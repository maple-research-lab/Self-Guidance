#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os
import pkg_resources
from setuptools import setup, find_packages

# Package meta-data
NAME = 'self_guidance'
DESCRIPTION = 'Self Guidance: Boosting Flow and Diffusion Generation on Their Own'
URL = 'https://github.com/maple-research-lab/Self-Guidance'
EMAIL = 'litiancheng913@gmail.com'
AUTHOR = 'Tiancheng Li et al.'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.1.0'


here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(include=['models', 'models.*','dnnlib']),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ]
)