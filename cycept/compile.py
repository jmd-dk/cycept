import contextlib
import logging
import os
import pathlib
import sys
import warnings

import Cython.Build
import Cython.Distutils
import setuptools


# Method for handling Cythonization and C compilation
def compile(module_path, optimizations, c_lang, html):
    module_path = pathlib.Path(module_path)
    optimization_options = OptimizationOptions(optimizations)
    macros = get_macros()
    # Context manager to redirect all logging to stdout
    @contextlib.contextmanager
    def print_logging():
        root = logging.getLogger()
        level = root.level
        root.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        root.addHandler(handler)
        try:
            yield
        except Exception:
            raise
        finally:
            root.removeHandler(handler)
    # Cythonize and compile extension module
    with (
        contextlib.chdir(module_path.parent),
        print_logging(),
    ):
        print(f'Changed working directory to {os.getcwd()}')
        extension_kwargs = {}
        if c_lang:
            extension_kwargs |= {'language': c_lang}
        extension = Cython.Distutils.Extension(
            name=module_path.stem,
            sources=[module_path.with_suffix('.pyx').name],
            define_macros=[(macro, None) for macro in macros],
            extra_compile_args=optimization_options.all,
            extra_link_args=optimization_options.base,
            **extension_kwargs,
        )
        build_ext = get_build_ext()
        build_ext.build_temp = ''
        build_ext.build_lib  = ''
        build_ext.extensions = Cython.Build.cythonize(
            [extension],
            compiler_directives={'language_level': 3},
            annotate=html,
        )
        build_ext.run()


# Function for obtaining a newly instantiated build_ext
def get_build_ext():
    dist = setuptools.dist.Distribution()
    dist.parse_config_files(dist.find_config_files())
    build_ext = Cython.Distutils.build_ext(dist)
    build_ext.finalize_options()
    return build_ext


# Function for detecting the type of C compiler to be used by setuptools
def get_compiler():
    # We adopt compiler names from the compiler_type attributes
    # on the distutils compiler classes. In particular:
    # * 'unix': GCC or Clang
    # * 'msvc': Microsoft Visual C++
    compiler_default = 'unix'
    build_ext = get_build_ext()
    build_ext.extensions = [None]
    try:
        build_ext.run()
    except setuptools.distutils.errors.DistutilsError:
        pass
    if build_ext.compiler is None:
        return compiler_default
    return build_ext.compiler.compiler_type


# Class for handling compiler optimization options
class OptimizationOptions:

    defaults = {
        'base': True,
        'fastmath': True,
        'native': True,
    }
    compilers = {'unix', 'msvc'}
    mappings = {
        'base': {
            'unix': ['-O3', '-funroll-loops'],
            'msvc': ['/O2'],
        },
        'fastmath': {
            'unix': ['-ffast-math'],
            'msvc': ['/fp:fast'],
        },
        'native': {
            'unix': ['-march=native', '-mtune=native'],
            'msvc': [],
        },
    }

    def __init__(self, optimizations=None):
        if optimizations is None:
            optimizations = {}
        # Merge user and default optimizations
        self.optimizations = self.defaults | optimizations
        if unrecognized := set(optimizations) - set(self.defaults):
            raise AttributeError(
                'Unrecognized optimization(s): {}'.format(', '.join(unrecognized))
            )
        # Set optimization options
        self.compiler = get_compiler()
        if self.compiler not in self.compilers and any(self.optimizations.values()):
            warnings.warn(
                (
                    f'Will apply compiler optimizations for unknown '
                    f'compiler \'{self.compiler}\' in Unix style'
                ),
                RuntimeWarning,
            )
            self.compiler = 'unix'
        for optimization, enable in self.optimizations.items():
            attr = []
            if enable:
                attr = self.mappings[optimization][self.compiler]
            setattr(self, optimization, attr)

    @property
    def all(self):
        return self.base + self.fastmath + self.native


# Function for setting up macros
def get_macros(macros=None):
    if macros is None:
        macros = []
    return macros_default + macros
macros_default = ['CYTHON_WITHOUT_ASSERTIONS', 'NDEBUG']

