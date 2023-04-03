import ast
import collections
import contextlib
import dataclasses
import functools
import inspect
import itertools
import os
import pathlib
import re
import sys
import tempfile
import traceback
import typing
import warnings
import webbrowser

import cython
import Cython.Build.Cythonize
import numpy as np


# Main JIT decorator function
def jit(func=None, **options):
    """This decorator compiles a Python function to machine code using Cython

    Basic usage
    -----------
        @jit
        def func(x):
            return 2*x

        func(5)

    Arguments
    ---------
        options:
            silent: bool
                Set to False to display compilation commands.
                Default value is True.


            silent_print: bool
                The function body is executed as part of the compilation.
                Side effects produced by running the function are then
                generally also produced by compiling it, which is typically
                unwanted. Thus, by default, the print() function is disabled
                during compilation. Set this argument to False to enable
                print() during compilation, meaning that any print() call will
                be run twice on the first call to the function.
                Default value is True.

            html: bool
                Set to True to produce HTML annotations of the compiled code.
                View the HTMl using func.__cycept__().
                Default value is False.

            checks: bool
                Set to False to disable the following checks within Cython:
                    * boundscheck
                    * initializedcheck
                Disabling these checks speeds up array code.
                Note that using

                    @jit(checks=False)

                is equivalent to

                    @jit(directives={'boundscheck': False, 'initializedcheck', False})

                Default value is True.

            clike: bool
                Set to True to substitute Python behaviour for
                non-equivalent C behaviour for the following operations:
                    * Indexing with negative integers.
                    * Integer division involving negative numbers.
                    * Exponentiation involving negative numbers.
                If your code does not rely on the above Python specifics,
                explicitly adopting the C behaviour improves performance
                for these operations.

                Note that using

                    @jit(clike=True)

                is equivalent to

                    @jit(directives={'wraparound': False, 'cdivision', True, 'cpow': True})

                Default value is False.

            directives: None | dict
                Allows for adding Cython directives to the transpiled
                function, typically in order to achieve further speedup.
                To e.g. disable boundschecking of indexing operations:

                    @jit(directives={'boundscheck': False})

                You can read about the different directives here:
                https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#compiler-directives

            optimizations: None | str | list |  dict | int | bool
                Specifies which optimizations to use with the C compiler
                (given as CFLAGS).
                If not provided or None, default (aggressive) optimizations
                are used. One or more optimizations can be given as a str
                or list of strs, overwriting the defaults, e.g.

                    @jit(optimizations='-O2')

                If not supplied, default optimizations include
                    * -O3
                    * -ffast-math
                    * -march=native
                To remove e.g. -ffast-math while keeping the remaining
                default optimizations, use

                    @jit(optimizations={'-ffast-math': False})

                If the value within the dict is True rather than False,
                the given optimization is added on top of the default
                optimizations rather than being removed.

                Specifying the optimizations argument as an int 'n'
                is equivalent to passing just the overall optimization level
                (which disables all defaults):

                    @jit(optimizations=n)  # equivalent to the below
                    @jit(optimizations=f'-O{n}')

                If the optimizations argument is given as a bool,
                all optimizations are enabled/disabled. That is,

                    @jit(optimizations=True)

                is equivalent to not passing the optimizations argument
                at all, making use of the defaults. Using

                    @jit(optimizations=False)

                replaces all of the default optimizations with -O0.

    Annotations
    -----------
        Types of variables are automatically inferred at runtime.
        Type annotations can be specified to manually set the types
        of specific variables, e.g.

            @jit
            def func(a: int, b: list) -> float:
                c: object
                c = b[a]
                d: float = (c**1000 % 42)**0.5
                return d

            func(2, list(range(0, 10, 3)))
    """
    if func is None:
        return functools.partial(jit, **options)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get and call transpiled function
        result = _transpile(func, wrapper, args, kwargs, **options)(*args, **kwargs)
        # If a Cython memoryview is returned, convert to NumPy array
        if 'memoryview' in str(type(result)).lower():
            result = np.asarray(result)
        return result

    # Add storage for function call objects on wrapper
    wrapper.__cycept__ = _CyceptStorage()
    return wrapper


# Dict-like class for storing function call objects on the wrapper
class _CyceptStorage(dict):
    # When called, dispatch to all function call objects
    def __call__(self):
        if not self:
            print('No compilation has taken place')
        else:
            dir_name = None
            for call in self.values():
                dir_name = call(dir_name)
                if dir_name is None:
                    break


# Function for carrying out the transpilation
# Python → Cython → C → machine code.
def _transpile(
    func,
    wrapper,
    args,
    kwargs,
    *,
    silent=True,
    silent_print=True,
    html=False,
    checks=True,
    clike=False,
    directives=None,
    optimizations=None,
):
    # Fetch function call object, implementing dynamic evaluation
    # of various attributes.
    call = _fetch_function_call(func, args, kwargs)
    # If the call has already been transpiled and cached,
    # return immediately.
    if call.compiled is not None:
        return call.compiled.func
    # The function call object was not found in cache
    if not silent:
        print(f'Jitting {call!r}')
    # Record types of passed arguments and locals
    _record_types(call)
    _record_locals(call, silent_print)
    # Convert Python function source into Cython module source
    directives = _get_directives(directives, checks, clike)
    call.to_cython(directives)
    # Cythonize and compile
    optimizations = _get_optimizations(optimizations)
    call.compile(optimizations, html, silent)
    # Store the function call object on the wrapper function
    wrapper.__cycept__[call.arguments_types] = call
    return call.compiled.func


# Function used to fetch _FunctionCall instance from the global
# cache or instantiating a new one if not found in the cache.
def _fetch_function_call(func, args=None, kwargs=None):
    if isinstance(func, int):
        # Called with function call hash
        return _cache[func]
    # We need to instantiate a fresh instance to obtain the hash
    call = _FunctionCall(func, args, kwargs)
    call_cached = _cache.get(call.hash)
    if call_cached is not None:
        # Found in cache. Disregard the freshly created instance.
        return call_cached
    # Not found in cache. Cache and return the freshly created instance.
    _cache[call.hash] = call
    return call
_cache = {}


# Class for storing and dynamically handling
# various aspects of the function call to be jitted.
class _FunctionCall:

    class Compiled(typing.NamedTuple):
        func: object
        module: object
        source: str
        html: str

    def __init__(self, func, args, kwargs):
        # Function and call arguments
        self.func = func
        self.args = args
        self.kwargs = kwargs
        # Properties to be lazily constructed
        self._func_name = None
        self._signature = None
        self._arguments = None
        self._arguments_types = None
        self._hash = None
        self._source = None
        self._ast = None
        self._annotations = None
        self._types = None
        self._module = None
        self._module_dict = None
        self._globals = None
        # Record of inferred types of local variables
        self.locals_types = {}
        # Compilation products
        self.compiled = None

    # Name of function (with lambda functions renamed)
    @property
    def func_name(self):
        if self._func_name is not None:
            return self._func_name
        self._func_name = self.func.__name__
        if self._func_name == '<lambda>':
            self._func_name = f'_cycept_lambda_{self.hash}'
        return self._func_name

    # Function signature (no type information)
    @property
    def signature(self):
        if self._signature is not None:
            return self._signature
        self._signature = inspect.signature(self.func)
        return self._signature

    # Argument values (dict)
    @property
    def arguments(self):
        if self._arguments is not None:
            return self._arguments
        bound = self.signature.bind(*self.args, **self.kwargs)
        bound.apply_defaults()
        self._arguments = bound.arguments
        return self._arguments

    # Argument types (tuple)
    @property
    def arguments_types(self):
        if self._arguments_types is not None:
            return self._arguments_types
        self._arguments_types = tuple(
            _get_type(val) for val in self.arguments.values()
        )
        return self._arguments_types

    # Hash of the function call
    @property
    def hash(self):
        if self._hash is not None:
            return self._hash
        self._hash = abs(hash((self.func, self.arguments_types)))
        return self._hash

    # Source code of the function
    @property
    def source(self):
        if self._source is not None:
            return self._source
        self._source = self.get_source()
        # Separate annotations from source
        self._annotations, self._source = self.extract_annotations()
        # Invalidate AST
        self._ast = None
        return self._source

    @property
    def ast(self):
        if self._ast is not None:
            return self._ast
        self._ast = ast.parse(self.source)
        return self._ast

    # User-defined annotations within the function
    @property
    def annotations(self):
        if self._annotations is not None:
            return self._annotations
        self._annotations, _ = self.extract_annotations()
        return self._annotations

    # Types to be used during compilation
    @property
    def types(self):
        if self._types is not None:
            return self._types
        # Transform recorded types of locals to str
        # representations of corresponding Cython types.
        locals_types = {
            name: _convert_to_cython_type(tp)
            for name, tp in self.locals_types.items()
        }
        # Transform annotations to str representations
        # of Cython types.  Keep unrecognized types as is.
        annotations = {
            name: _convert_to_cython_type(tp, tp)
            for name, tp in self.annotations.items()
        }
        # Merge recorded types and annotations.
        # On conflict we keep the annotation.
        self._types = locals_types | annotations
        return self._types

    # Module within which function was defined
    @property
    def module(self):
        if self._module is not None:
            return self._module
        self._module = inspect.getmodule(self.func)
        return self._module

    # Module dictionary associated with function module
    @property
    def module_dict(self):
        if self._module_dict is not None:
            return self._module_dict
        self._module_dict = getattr(self.module, '__dict__', None)
        if self._module_dict is None:
            raise ModuleNotFoundError(
                f'Failed to get __dict__ of module {self.module} '
                f'within which function {self.func} is defined'
            )
        return self._module_dict

    # Global names referenced within the function
    @property
    def globals(self):
        if self._globals is not None:
            return self._globals
        self._globals = inspect.getclosurevars(self.func).globals
        return self._globals

    # Method for obtaining the function source
    def get_source(self):
        try:
            source = inspect.getsource(self.func)
        except Exception:
            # inspect.getsource() fails in the interactive REPL.
            # The dill package can hack around this limitation.
            try:
                import dill
            except ModuleNotFoundError:
                print(
                    f'To use cycept.jit interactively, '
                    f'please install the dill Python package:\n'
                    f'    {sys.executable} -m pip install dill\n',
                    file=sys.stderr,
                )
                raise
            source = dill.source.getsource(self.func)
        # Dedent source lines
        lines = source.split('\n')
        indentation = ' ' * (len(lines[0]) - len(lines[0].lstrip()))
        if indentation:
            for i, line in enumerate(lines):
                lines[i] = line.removeprefix(indentation)
            source = '\n'.join(lines)
        # Convert lambda function to def function
        if self.func.__name__ == '<lambda>':
            # Extract just the lambda expression
            # (both inspect.getsource() and dill.source.getsource()
            # returns the surrounding code as well in the case of unassigned
            # lambda expressions).
            class LambdaVisitor(ast.NodeVisitor):
                def __init__(self, root):
                    self.root = root
                    self._lambda = None
                def get_lambda(self):
                    self._lambda = None
                    self.visit(self.root)
                    return self._lambda
                def visit_Lambda(self, node):
                    if self._lambda is None:
                        self._lambda = node
            source = ast.unparse(LambdaVisitor(ast.parse(source)).get_lambda())
            # Transform to def function
            indentation = ' ' * 4
            source = re.subn(
                r'.*(^|\W)lambda (.+?): ?',
                rf'def {self.func_name}(\g<2>):\n{indentation}return ',
                source,
                1,
            )[0]
        else:
            # Do AST round-trip, ensuring canonical code style
            # (e.g. 4 space indentation).
            source = ast.unparse(ast.parse(source))
        # Remove jit decorator and decorators above it
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if match := re.search(r'^@(.+?)(\(|$)', line):
                deco_name = match.group(1).strip()
                try:
                    deco_func = eval(deco_name, self.module_dict)
                except Exception:
                    continue
                if deco_func is jit:
                    lines = lines[i+1:]
                    break
            elif line.startswith('def '):
                # Arrived at definition line without finding jit decorator.
                # This happens if the "decoration" is done as
                # func = jit(func) rather than using the @jit syntax.
                break
        source = '\n'.join(lines)
        return source

    # Method for extracting user-defined type annotations.
    # The return value is the annotations as well as
    # the modified source without the annotations.
    def extract_annotations(self):
        class AnnotationExtractor(ast.NodeTransformer):
            def __init__(self, call):
                self.call = call
                self.annotations = {}
            def extract(self):
                self.annotations.clear()
                source = ast.unparse(self.visit(self.call.ast))
                return self.annotations, source
            def add_annotation(self, name, tp):
                if not isinstance(name, str):
                    name = ast.unparse(name)
                tp = eval(ast.unparse(tp), self.call.module_dict)
                tp_old = self.annotations.setdefault(name, tp)
                if tp_old == tp:
                    return
                warnings.warn(
                    (
                        f'Name {name!r} annotated with different types '
                        f'in jitted {self.call}: {tp}, {tp_old}. '
                        f'The first will be used'
                    ),
                    RuntimeWarning,
                )
            def visit_arg(self, node):
                if not node.annotation:
                    return node
                self.add_annotation(node.arg, node.annotation)
                return type(node)(**(vars(node) | {'annotation': None}))
            def visit_FunctionDef(self, node):
                self.generic_visit(node)
                if not node.returns:
                    return node
                self.add_annotation('return', node.returns)
                return type(node)(**(vars(node) | {'returns': None}))
            def visit_AnnAssign(self, node):
                if node.simple:
                    self.add_annotation(node.target, node.annotation)
                else:
                    warnings.warn(
                        (
                            f'Ignoring complex type annotation '
                            f'{ast.unparse(node)!r} in jitted {self.call}'
                        ),
                        RuntimeWarning,
                    )
                if not node.value:
                    # Annotation-only statement. Remove completely.
                    return None
                # Transform to assignment statement without annotation
                return ast.Assign(**(
                    vars(node) | {'targets': [node.target], 'value': node.value}
                ))
        annotations, source = AnnotationExtractor(self).extract()
        return annotations, source

    # Method for updating the source from being a Python function
    # to being a Cython extension module.
    def to_cython(self, directives):
        preamble = [
            '#',  # first comment line can have C code attached to it
            f'# Cycept version of {self} with type signature',
            f'# {self!r}',
        ]
        excludes = (self.func_name, 'cython')
        fake_globals = [
            f'{name} = object()'
            for name in self.globals
            if name not in excludes
        ]
        if fake_globals:
            preamble += [
                '\n# Declare fake globals',
                *fake_globals,
            ]
        preamble += [
            '\n# Import Cython',
            'import cython',
            'cimport cython',
        ]
        header = ['\n# Function to be jitted']
        if self.ccall_allowed():
            header.append('@cython.ccall')
        for directive, val in directives.items():
            header.append(f'@cython.{directive}({val!r})')
        excludes = ('return', 'cycept')
        declaration_locals = ', '.join(
            f'{name}={tp}'
            for name, tp in self.types.items()
            if name not in excludes
        )
        declaration_return = self.types.get('return', 'object')
        if declaration_locals:
            header.append(f'@cython.locals({declaration_locals})')
        header.append(f'@cython.returns({declaration_return})')
        self._source = '\n'.join(
            itertools.chain(
                preamble,
                header,
                self.source.split('\n'),
            )
        )

    # Method for determining whether the function to be Cythonized
    # may be done so using @cython.ccall (cpdef).
    def ccall_allowed(self):
        # *args and **kwargs not allowed with ccall
        for param in self.signature.parameters.values():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                return False
        # Closures not allowed with ccall
        class ClosureVisitor(ast.NodeVisitor):
            def __init__(self, call):
                self.call = call
                self._contains_closure = False
            def contains_closure(self):
                self._contains_closure = False
                self.visit(self.call.ast)
                return self._contains_closure
            def visit_FunctionDef(self, node):
                self.generic_visit(node)
                if node.name != self.call.func_name:
                    self._contains_closure = True
            def visit_Lambda(self, node):
                self._contains_closure = True
        if ClosureVisitor(self).contains_closure():
            return False
        # No violations found
        return True

    # Method for handling Cythonization and C compilation
    def compile(self, optimizations, html, silent):
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
            module_path = get_module_path(dir_name)
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
        def hack_sys_path():
            sys.path.append(dir_name)
            yield
            sys.path.remove(dir_name)
        @contextlib.contextmanager
        def hack_distutils_spawn_log(silent):
            if silent:
                # Operating silently is the default behaviour of distutils
                yield
                return
            import distutils
            class StdoutPrinter:
                @staticmethod
                def print(msg, *args, **kwargs):
                    print(msg)
                def __getattr__(self, attr):
                    return self.print
            distutils_spawn_log = distutils.spawn.log
            distutils.spawn.log = StdoutPrinter()
            yield
            distutils.spawn.log = distutils_spawn_log
        # Cythonize and compile extension module within temporary directory,
        # with arguments to Cython and the C compiler provided through hacking
        # of sys.argv and os.environ. The compiled module is then imported
        # (through hacking of sys.path) before it is removed from disk.
        # If not compiling silently, we further hack distutils_spawn_log to
        # print output to stdout.
        module_name = f'_cycept_module_{self.hash}'
        get_module_path = lambda dir_name: pathlib.Path(dir_name) / module_name
        namespace = {}
        html_annotation = None
        with (
            tempfile.TemporaryDirectory() as dir_name,
            hack_os_environ(),
            hack_sys_argv(),
            hack_sys_path(),
            hack_distutils_spawn_log(silent),
        ):
            module_path = get_module_path(dir_name)
            # Write Cython source to file
            module_path.with_suffix('.pyx').write_text(self.source, 'utf-8')
            # Call Cython.Build.Cythonize.main(), which is equivalent
            # to calling the cythonize script.
            ok = True
            try:
                Cython.Build.Cythonize.main()
            except BaseException:
                ok = False
                traceback.print_exc()
            if not ok:
                if sys.flags.interactive:
                    # Do not remove the compilation files immediately when
                    # running interactively, allowing the user to inspect them.
                    input(f'Press Enter to clean up temporary build directory {dir_name} ')
                raise OSError('Cythonization failed')
            # Import function from compiled module into temporary namespace
            exec(f'import {module_name}', namespace)
            # Read in C source and annotated HTML
            source_c = module_path.with_suffix('.c').read_text('utf-8')
            path_html = module_path.with_suffix('.html')
            if path_html.is_file():
                html_annotation = path_html.read_text('utf-8')
        # Extract compiled function
        module_compiled = namespace[module_name]
        module_compiled_dict = module_compiled.__dict__
        func_compiled = module_compiled_dict[self.func_name]
        # Replace fake globals with actual globals within extension module
        for name, val in self.globals.items():
            if name == self.func_name:
                continue
            module_compiled_dict[name] = val
        # Store compilation products
        self.compiled = self.Compiled(
            func_compiled,
            module_compiled,
            source_c,
            html_annotation,
        )

    # Method for viewing the annotated HTML
    def __call__(self, dir_name=None):
        if self.compiled is None or self.compiled.html is None:
            print(f'No Cython HTML annotation generated for {self.func_name}()')
            return
        if dir_name is None:
            dir_name = tempfile.mkdtemp()  # not cleaned up
        module_path = pathlib.Path(dir_name) / self.compiled.module.__name__
        module_path.with_suffix('.c').write_text(self.compiled.source, 'utf-8')
        path_html = module_path.with_suffix('.html')
        path_html.write_text(self.compiled.html, 'utf-8')
        webbrowser.open_new_tab(str(path_html.as_uri()))
        return dir_name

    # Method for pretty printing (showing types)
    def __repr__(self):
        return f'{self.func_name}({{}})'.format(
            ', '.join(
                f'{name}: {_prettify_type(tp)}'
                for name, tp in zip(self.arguments, self.arguments_types)
            )
        )

    # Method for pretty printing (showing module)
    def __str__(self):
        return f'function {self.func_name}() defined in {self.module}'


# Function for setting up Cython directives
def _get_directives(directives, checks, clike):
    def set_directive(directives, directive, val):
        if not hasattr(cython, directive):
            return
        directives.setdefault(directive, val)
    if directives is None:
        directives = {}
    if not checks:
        checks_disable = {
            'boundscheck': False,
            'initializedcheck': False,
        }
        for directive, val in checks_disable.items():
            set_directive(directives, directive, val)
    if clike:
        clike_enable = {
            'wraparound': False,
            'cdivision': True,
            'cpow': True,
        }
        for directive, val in clike_enable.items():
            set_directive(directives, directive, val)
    for directive, val in directives.items():
        if not hasattr(cython, directive):
            warnings.warn(
                f'Cython directive {directive!r} does not seem to exist',
                RuntimeWarning,
            )
    return directives


# Function for setting up compiler optimizations (CFLAGS)
def _get_optimizations(optimizations):
    get_defaults = lambda: _optimizations_default.copy()
    if optimizations is None:
        # No optimizations provided. Use defaults.
        optimizations = get_defaults()
    elif isinstance(optimizations, (bool, int)):
        # Enable or disable all optimizations
        if optimizations is True:
            optimizations = get_defaults()
        else:
            level = abs(int(optimizations))
            optimizations = [f'-O{level}']
    elif isinstance(optimizations, str):
        # Transform str of optimizations to list
        optimizations = optimizations.split(' ')
    elif isinstance(optimizations, dict):
        # Use defaults with amendments
        optimizations_ammended = get_defaults()
        for optimization, enable in optimizations.items():
            if enable and optimization not in optimizations_ammended:
                optimizations_ammended.append(optimization)
            elif not enable and optimization in optimizations_ammended:
                optimizations_ammended.remove(optimization)
        optimizations = optimizations_ammended
    else:
        # Use supplied optimizations as is
        optimizations = list(optimizations)
    return optimizations
_optimizations_default = [
    '-DNDEBUG',
    '-O3',
    '-funroll-loops',
    '-ffast-math',
    '-march=native',
]


# Function for extracting the Python/NumPy type off of a value
def _get_type(val):
    tp = _construct_ndarray_type_info(val)
    if tp is not None:
        return tp
    return type(val)


# Function responsible for constructing NdarrayTypeInfo instances,
# storing type information of arrays/memoryviews.
def _construct_ndarray_type_info(a):
    def construct(dtype, ndim, c_contig, f_contig):
        if c_contig:
            # Disregard F-contiguousness if C-contiguous
            f_contig = False
        return NdarrayTypeInfo(dtype, ndim, c_contig, f_contig)
    if isinstance(a, NdarrayTypeInfo):
        return a
    if isinstance(a, _numpy_array_type):
        return construct(
            a.dtype.type,
            a.ndim,
            a.flags['C_CONTIGUOUS'],
            a.flags['F_CONTIGUOUS'],
        )
    if isinstance(a, _cython_array_type):
        return construct(
            _cython_types_reverse[_cython_types[a.dtype]],
            a.ndim,
            a.is_c_contig,
            a.is_f_contig,
        )
    # Passed value is not an array
    return None
_numpy_array_type = np.ndarray
_cython_array_type = type(cython.int[:])


# Type used to discern different kinds of arrays (memoryviews)
@dataclasses.dataclass(frozen=True)
class NdarrayTypeInfo:
    dtype: np.dtype.type
    ndim: int
    c_contig: bool
    f_contig: bool

    @property
    def dtype_cython(self):
        return _convert_to_cython_type(self.dtype)

    @property
    def __name__(self):
        name = '{}[{}]'.format(self.dtype_cython, ', '.join(':' * self.ndim))
        if self.c_contig:
            name = name.replace(':]', '::1]')
        elif self.f_contig:
            name = name.replace('[:', '[::1')
        return name

    def __str__(self):
        return self.__name__.removeprefix('cython.')


# Function for recording types
def _record_types(call, variables=None):
    if isinstance(call, int):
        # Called with function call hash
        call = _fetch_function_call(call)
    if variables is None:
        # Use function call arguments
        variables = call.arguments
    for name, val in variables.items():
        tp = _get_type(val)
        tp_old = call.locals_types.setdefault(name, tp)
        if tp_old == tp:
            continue
        if (
            (tp_old in (float, int, bool) and tp in (int, bool))
            or (tp_old == complex and tp in (float, int, bool))
        ):
            # Allow implicit upcasting of Python scalars
            # as well as casting back and forth between bools and ints.
            call.locals_types.record[name] = tp_old
        else:
            warnings.warn(
                (
                    f'Name {name!r} assigned different types '
                    f'in jitted {call}: {tp}, {tp_old}. '
                    f'The first will be used'
                ),
                RuntimeWarning,
            )


# Function for obtaining the types of locals within a function
def _record_locals(call, silent_print):
    # Create copy of function source with the function renamed
    class FunctionRenamer(ast.NodeTransformer):
        def __init__(self, name_old, name_new):
            self.name_old = name_old
            self.name_new = name_new
            self.visit_Name = self.make_visitor('id')
            self.visit_FunctionDef = self.make_visitor('name')
        def make_visitor(self, attr):
            def rename(node):
                self.generic_visit(node)
                if getattr(node, attr) == self.name_old:
                    node = type(node)(**(vars(node) | {attr: self.name_new}))
                return node
            return rename
    func_name_tmp = f'_cycept_func_{call.hash}'
    source = ast.unparse(
        FunctionRenamer(call.func_name, func_name_tmp).visit(call.ast)
    )
    # Add type recording calls to source
    cycept_module_refname = '__cycept__'
    record_str = f'{cycept_module_refname}._record_types({call.hash}, locals())'
    source = re.sub(
        r'( |;)return($|\W)',
        rf'\g<1>{record_str}; return\g<2>',
        source,
    )
    indentation = ' ' * 4
    source += f'\n{indentation}{record_str}'
    # Define and call modified function within definition module,
    # with recording of the types added as a side effect.
    # Temporarily add a reference to this module on the module of
    # the function to be jitted. Disable the print() function while
    # performing this operation, unless silent_print is False.
    @contextlib.contextmanager
    def hack_module_dict():
        call.module_dict[cycept_module_refname] = sys.modules['cycept']
        yield
        call.module_dict.pop(cycept_module_refname)
    @contextlib.contextmanager
    def hack_print(silent_print):
        if not silent_print:
            yield
            return
        print_in_module = ('print' in call.module_dict)
        print_func = call.module_dict.get('print')
        call.module_dict['print'] = lambda *args, **kwargs: None
        yield
        if print_in_module:
            call.module_dict['print'] = print_func
        else:
            call.module_dict.pop('print')
    with hack_module_dict(), hack_print(silent_print):
        # Define modified function within definition module
        exec(source, call.module_dict)
        # Call modified function with passed arguments
        return_val = call.module_dict[func_name_tmp](*call.args, **call.kwargs)
    # Remove modified function from definition module
    call.module_dict.pop(func_name_tmp)
    # Add return type to record
    _record_types(call, {'return': return_val})


# Function returning pretty str representation of type
def _prettify_type(tp):
    if isinstance(tp, NdarrayTypeInfo):
        return str(tp)
    tp_name = getattr(tp, '__name__', getattr(tp, 'name', None))
    if tp_name is not None:
        return tp_name
    return str(tp)


# Function for converting a Python/NumPy/Cython type to the
# str representation of the corresponding Cython type.
def _convert_to_cython_type(tp, default=object):
    def convert_array_type(tp):
        if _cython_types[tp.dtype] == 'object':
            return 'object'
        return tp.__name__
    # Type already given as str
    if isinstance(tp, str):
        return repr(tp)
    # NumPy array or Cython memoryview
    tp_arr = _construct_ndarray_type_info(tp)
    if tp_arr is not None:
        return convert_array_type(tp_arr)
    # Python/NumPy/Cython
    tp_str = _cython_types.get(tp)
    if tp_str is not None:
        return tp_str
    # Unrecognized
    if default is not None and tp is not default:
        return _convert_to_cython_type(default, None)
    tp_str = getattr(tp, '__name__', getattr(tp, 'name', None))
    if tp_str is not None:
        return tp_str
    return str(tp)


# Mapping of Python/NumPy types to str representations
# of corresponding Cython types.
_cython_default_integral = 'cython.Py_ssize_t'  # typically 64-bit
_cython_types = collections.defaultdict(
    lambda: 'object',
    {
        # Python scalars
        bool: 'cython.bint',
        int: _cython_default_integral,
        float: 'cython.double',
        complex: 'cython.complex',
        # NumPy signed integral scalars
        np.int8: 'cython.schar',
        np.int16: 'cython.short',
        np.int32: 'cython.int',
        np.int64: 'cython.longlong',
        # NumPy unsigned integral scalars
        np.uint8: 'cython.uchar',
        np.uint16: 'cython.ushort',
        np.uint32: 'cython.uint',
        np.uint64: 'cython.ulonglong',
        # NumPy floating scalars
        np.float32: 'cython.float',
        np.float64: 'cython.double',
        np.longdouble: 'cython.longdouble',  # platform dependent
        # NumPy complex floating scalars
        np.complex64: 'cython.floatcomplex',  # Cython type may not be available at compile time
        np.complex128: 'cython.doublecomplex',
        np.complex256: 'cython.longdoublecomplex',  # Cython type may not be available at compile time; np.longdoublecomplex not defined
        # Python containers
        **{
            tp: tp.__name__
            for tp in [
                bytearray,
                bytes,
                dict,
                frozenset,
                list,
                object,
                set,
                str,
                tuple,
            ]
        },
    },
)
# Also create reverse mapping
_cython_types_reverse = collections.defaultdict(
    lambda: object,
    {name: tp for tp, name in _cython_types.items()},
)
# Now add the Cython types themselves
_cython_types |= {
    getattr(cython, name.removeprefix('cython.')): name
    for tp, name in _cython_types.items()
    if name.startswith('cython.')
}

