from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os

cwd = os.path.abspath(os.path.dirname(__file__))
if not os.path.exists(os.path.join(cwd, 'PKG-INFO')):
    try:
        from Cython.Build import cythonize
        USE_CYTHON = True
    except ImportError:
        USE_CYTHON = False
else:
    USE_CYTHON = False

extra_compile_args = {
    'msvc': ['/std:c++14'],
    'unix': ['-std=c++11'],
}

extra_link_args = {
    'msvc': [],
    'unix': [],
}


class build_ext_compiler_check(build_ext):
    def build_extensions(self):
        compiler = self.compiler.compiler_type

        compile_args = extra_compile_args[compiler]
        for ext in self.extensions:
            ext.extra_compile_args = compile_args

        link_args = extra_compile_args[compiler]
        for ext in self.extensions:
            ext.extra_compile_args = link_args

        build_ext.build_extensions(self)

    def run(self):
        import numpy

        self.include_dirs.append(numpy.get_include())

        build_ext.run(self)


if USE_CYTHON:
    ext = '.pyx'
else:
    ext = '.cpp'


extensions = [Extension(
    name='neal.simulated_annealing',
    sources=['./neal/simulated_annealing' + ext,
             './neal/src/cpu_sa.cpp'],
    include_dirs=['./neal/src/'],
    language='c++',
)]

if USE_CYTHON:
    extensions = cythonize(extensions, language='c++')

packages = ['neal']

install_requires = ['dimod>=0.7.0,<0.8.0',
                    'numpy>=1.14.0,<1.16.0',
                    'six>=1.11.0,<2.0.0']

setup_requires = ['numpy>=1.14.0,<1.16.0']

classifiers = [
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7'
]

python_requires = '>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*'

_PY2 = sys.version_info.major == 2

# add __version__, __author__, __authoremail__, __description__ to this namespace
if _PY2:
    execfile("./neal/package_info.py")
else:
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
    packages=packages,
    install_requires=install_requires,
    ext_modules=extensions,
    cmdclass={'build_ext': build_ext_compiler_check},
    setup_requires=setup_requires,
    python_requires=python_requires,
    zip_safe=False
)
