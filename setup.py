from setuptools import setup,find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

exts=[Extension(name="Optx.BlackScholes",
                sources=["Optx/BlackScholes.pyx"]),
      Extension(name="Optx.Support",
                sources=["Optx/Support.pyx"]),
      Extension(name="Optx.Strategy",
                sources=["Optx/Strategy.pyx"])]

setup(name='Optx',version='0.1.3',
      requires=['scipy','numpy','datetime','cython'],
      author='Roberto Gomes',
      packages=find_packages(),
      ext_modules=cythonize(exts,language_level="3"),
      include_dirs=[numpy.get_include()])


