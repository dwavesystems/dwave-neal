from setuptools import setup, Extension
from Cython.Distutils import build_ext
import platform
import numpy

src_folder = "dwave_neal/"

cpp_version_arg = "-std=c++11" # default, works on linux

if platform.system().lower() == "windows":
    cpp_version_arg = "/std:c++14"

ext_module = Extension("dwave_neal_sampler", 
        [src_folder + "sampler/" + "general_simulated_annealing.pyx", 
            src_folder + "sampler/" + "cpu_sa.cpp"], 
        language="c++",
        extra_compile_args=[cpp_version_arg],
        extra_link_args=[cpp_version_arg],
        include_dirs=[numpy.get_include()]
        )

setup(name="dwave_neal",
      version="0.2.0",
      description="General Ising graph simulated annealing solver",
      author="William Bernoudy",
      author_email="wbernoudy@dwavesys.com",
      license="Apache 2.0",
      cmdclass = {"build_ext": build_ext}, 
      ext_modules = [ext_module],
      packages=["dwave_neal", "dwave_neal.dwave_neal_dimod"],
      install_requires=[
          "numpy",
          "dimod",
      ])

