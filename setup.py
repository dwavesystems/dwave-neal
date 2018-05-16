from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os

import numpy as np

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


ext = '.pyx' if USE_CYTHON else '.cpp'


extensions = [Extension(
    name="dwave_neal_sampler",
    sources=["./dwave_neal/sampler/general_simulated_annealing" + ext],
    include_dirs=[np.get_include()],
    language='c++',
)]

if USE_CYTHON:
    extensions = cythonize(extensions, language='c++')

packages = ['dwave_neal',
            'dwave_neal_dimod',
            ]

install_requires = ['dimod==0.3.1',
                    'numpy==1.13.0',
                    ]

_PY2 = sys.version_info.major == 2

# add __version__, __author__, __authoremail__, __description__ to this namespace
if _PY2:
    execfile("./dwave_neal/package_info.py")
else:
    exec(open("./dwave_neal/package_info.py").read())


setup(
    name='dwave_neal',
    version=__version__,
    author=__author__,
    author_email=__authoremail__,
    description=__description__,
    long_description=open('README.md').read(),
    url='https://github.com/dwavesystems/dwavebinarycsp',
    license='Apache 2.0',
    packages=packages,
    install_requires=install_requires,
    ext_modules=extensions,
    cmdclass={'build_ext': build_ext_compiler_check}
)
