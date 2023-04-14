import contextlib
import os
import pathlib
import re
import sys

import Cython.Build
import distutils.core
import distutils.command.build_ext

# Implement contextlib.chdir if missing
if sys.version_info < (3, 11):
    @contextlib.contextmanager
    def _chdir(path):
        old_cwd = os.getcwd()
        os.chdir(path)
        yield
        os.chdir(old_cwd)
    contextlib.chdir = _chdir

# Method for handling Cythonization and C compilation
def compile(module_path, optimizations, c_lang, html):
    module_path = pathlib.Path(module_path)
    # Extract define macros from passed optimizations
    macros = []
    for optimization in optimizations:
        if match := re.match(r'-D(\S+)', optimization):
            macros.append(match.group(1))
    # Context manager forcing distutils to print out its logged messages
    @contextlib.contextmanager
    def hack_distutils_log():
        class Printer:
            def print(self, msg, *args, **kwargs):
                print(msg)
            def __getattr__(self, attr):
                return self.print
        printer = Printer()
        module_names = ['spawn', 'dist']
        loggers = {}
        for module_name in module_names:
            module = getattr(distutils, module_name, None)
            if module_name is None:
                continue
            logger = getattr(module, 'log', None)
            if logger is None:
                continue
            loggers[module_name] = logger
            module.log = printer
        yield
        for module_name, logger in loggers.items():
            module = getattr(distutils, module_name)
            module.log = logger
    # Cythonize and compile extension module
    with (
        contextlib.chdir(module_path.parent),
        hack_distutils_log(),
    ):
        print(f'Changed working directory to {os.getcwd()}')
        extension_kwargs = {}
        if c_lang:
            extension_kwargs |= {'language': c_lang}
        extension = distutils.core.Extension(
            name=module_path.stem,
            sources=[module_path.with_suffix('.pyx').name],
            define_macros=[(macro, None) for macro in macros],
            extra_compile_args=optimizations,
            extra_link_args=optimizations,
            **extension_kwargs,
        )
        dist = distutils.core.Distribution()
        dist.parse_config_files(dist.find_config_files())
        build_extension = distutils.command.build_ext.build_ext(dist)
        build_extension.finalize_options()
        build_extension.build_temp = ''
        build_extension.build_lib  = ''
        build_extension.extensions = Cython.Build.cythonize(
            [extension],
            compiler_directives={'language_level': 3},
            annotate=html,
        )
        build_extension.run()

