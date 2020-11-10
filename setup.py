from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
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

install_requires = ['dimod>=0.9.2',
                    'numpy>=1.16.0',
                    ]

setup_requires = ['numpy>=1.16.0']

classifiers = [
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
]

python_requires = '>=3.5'

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
    packages=packages,
    install_requires=install_requires,
    ext_modules=extensions,
    cmdclass={'build_ext': build_ext_compiler_check},
    setup_requires=setup_requires,
    python_requires=python_requires,
    zip_safe=False
)
