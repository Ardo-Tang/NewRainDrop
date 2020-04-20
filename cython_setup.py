from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("other_functions.pyx")
)
# python cython_setup.py build_ext --inplace