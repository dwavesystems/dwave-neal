from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os

install_requires = ['dwave-samplers>=1.0.0.dev0,<2.0.0']

classifiers = [
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
]

python_requires = '>=3.7'

# add __version__, __author__, __authoremail__, __description__ to this namespace
exec(open("./neal/package_info.py").read())


setup(
    name='dwave-neal',
    version=__version__,
    author=__author__,
    author_email=__authoremail__,
    description=__description__,
    long_description=open('README.rst').read(),
    url='https://github.com/dwavesystems/dwave-neal',
    license='Apache 2.0',
    classifiers=classifiers,
    packages=['neal'],
    install_requires=install_requires,
    python_requires=python_requires,
)
