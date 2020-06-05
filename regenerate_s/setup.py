from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
setup(name = 'func_stokes', ext_modules = cythonize('func_stokes.pyx'), include_dirs = [np.get_include()])
