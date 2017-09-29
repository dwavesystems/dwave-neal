from setuptools import setup, Extension
from Cython.Distutils import build_ext
import platform
import numpy

src_folder = "dw_sa_chi/"

cpp_version_arg = "-std=c++11" # default, works on linux

if platform.system().lower() == "windows":
    cpp_version_arg = "/std:c++14"

ext_module = Extension("dwave_sage_sampler", 
        [src_folder + "sampler/" + "general_simulated_annealing.pyx", 
            src_folder + "sampler/" + "cpu_sa.cpp"], 
        language="c++",
        extra_compile_args=[cpp_version_arg],
        extra_link_args=[cpp_version_arg],
        include_dirs=[numpy.get_include()]
        )

setup(name="dwave_sage",
      version="0.1.1",
      description="General Ising graph simulated annealing solver",
      author="William Bernoudy",
      author_email="wbernoudy@dwavesys.com",
      license="Apache 2.0",
      cmdclass = {"build_ext": build_ext}, 
      ext_modules = [ext_module],
      packages=["dwave_sage"],
      package_dir={"dwave_sage": src_folder + "dwave_sage_dimod"},
      install_requires=[
          "numpy",
          "dimod",
      ])

