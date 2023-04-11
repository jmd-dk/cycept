import contextlib
import distutils
import itertools
import os
import pathlib
import sys


import Cython.Build.Cythonize


# Method for handling Cythonization and C compilation
def compile(module_path, optimizations, html, silent):
    module_path = pathlib.Path(module_path)
    # Define context managers for temporarily hack into various
    # objects during Cythonization and compilation.
    @contextlib.contextmanager
    def hack_os_environ():
        cflags_in_environ = ('CFLAGS' in os.environ)
        cflags = os.environ.get('CFLAGS', '')
        os.environ['CFLAGS'] = ' '.join(optimizations)
        yield
        if cflags_in_environ:
            os.environ['CFLAGS'] = cflags
        else:
            os.environ.pop('CFLAGS')
    @contextlib.contextmanager
    def hack_sys_argv():
        sys_argv = sys.argv
        sys.argv = list(
            itertools.chain(
                [''],
                ['-i', '-3'],
                ['-q'] * silent,
                ['-a'] * html,
                [str(module_path.with_suffix('.pyx'))],
            )
        )
        yield
        sys.argv = sys_argv
    @contextlib.contextmanager
    def hack_distutils_log(silent):
        class Printer:
            def __init__(self, silent):
                self.silent = silent
            def print(self, msg, *args, **kwargs):
                if not self.silent:
                    print(msg)
            def __getattr__(self, attr):
                return self.print
        printer = Printer(silent)
        module_names = ['spawn', 'dist']
        loggers = {}
        for module_name in module_names:
            module = getattr(distutils, module_name)
            if module_name is None:
                continue
            logger = getattr(module, 'log')
            if logger is None:
                continue
            loggers[module_name] = logger
            module.log = printer
        yield
        for module_name, logger in loggers.items():
            module = getattr(distutils, module_name)
            module.log = logger
    # Cythonize and compile extension module with arguments to Cython
    # and the C compiler provided through hacking of os.environ and sys.argv.
    # We also hack into the logging of distutils, either silencing it
    # or making it print out the logged information.
    with (
        hack_os_environ(),
        hack_sys_argv(),
        hack_distutils_log(silent),
    ):
        Cython.Build.Cythonize.main()
