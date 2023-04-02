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
import warnings
import webbrowser

import cython
import Cython.Build.Cythonize
import numpy as np


# Name of this module
_this_module_name = pathlib.Path(__file__).stem


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
                View the HTMl using func.view_cython_html().
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

    # Add storage for HTML annotations and C sources
    # as attributes on the wrapper. Also add a dummy version
    # of the view_cython_html() function.
    wrapper.__cython_html__ = {}
    wrapper.__cython_c__ = {}
    wrapper.view_cython_html = lambda: print(
        f'No Cython HTML annotation generated for {func.__name__}'
    )
    return wrapper


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
    # Get call types
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    # Hash of the function together with the call types
    argument_types = tuple(_get_type(val) for val in bound.arguments.values())
    transpilation_hash = _get_transpilation_hash(func, argument_types)
    # Look up cached transpilation
    func_transpiled = _cache.get(transpilation_hash)
    if func_transpiled is not None:
        return func_transpiled
    # Pretty signature description, used with printouts
    sig_description = f'{func.__name__}({{}})'.format(
        ', '.join(
            f'{argument_name}: {_prettify_type(argument_type)}'
            for argument_name, argument_type in zip(bound.arguments, argument_types)
        )
    )
    if not silent:
        print(f'Jitting {sig_description}')
    # Function name and location, used with printouts
    func_name = func.__name__
    if func_name == '<lambda>':
        func_name = f'_{_this_module_name}_lambda_{transpilation_hash}'
    module, module_dict = _get_module(func)
    func_description = f'function {func_name}() ' f'defined in {module}'
    # Record types of passed arguments
    _record_types(bound.arguments, transpilation_hash, func_description)
    # Get function source
    source = _get_source(func, func_name)
    # Extract types from explicit user annotations
    # and remove the annotations from the source.
    source, annotations = _extract_annotations(
        func,
        source,
        module_dict,
        func_description,
    )
    # Record types of locals by executing the function
    # with added introspection.
    types = _record_locals(
        func_name,
        module_dict,
        source,
        transpilation_hash,
        func_description,
        silent_print,
        args,
        kwargs,
    )
    # Merge the recorded and explicitly annotated types
    # and transform them to Cython types when appropriate.
    types = _merge_types(annotations, types)
    # Set up Cython directives
    directives = _get_directives(directives, checks, clike)
    # Convert Python function source into Cython module source
    source = _make_cython_source(
        func,
        func_name,
        func_description,
        sig_description,
        source,
        sig,
        types,
        directives,
    )
    # Cythonize and compile
    optimizations = _get_optimizations(optimizations)
    module_compiled_dict, source_c, html_annotation = _compile(
        source,
        transpilation_hash,
        optimizations,
        html,
        silent,
    )
    func_transpiled = module_compiled_dict[func_name]
    # Store the annotated Cython HTML as an attribute
    # on the wrapper function.
    _place_html_annotation(func, wrapper, argument_types, source_c, html_annotation)
    # Replace fake globals with actual globals
    for name, val in inspect.getclosurevars(func).globals.items():
        if name == func_name:
            continue
        module_compiled_dict[name] = val
    # Cache and return transpiled function
    _cache[transpilation_hash] = func_transpiled
    return func_transpiled
_cache = {}


# Function for extracting Python source code from function object
def _get_source(func, func_name):
    try:
        source = inspect.getsource(func)
    except Exception:
        # inspect.getsource() fails in the interactive REPL.
        # The dill package can hack around this limitation.
        import dill
        source = dill.source.getsource(func)
    # Dedent source lines
    source_lines = source.split('\n')
    indentation = ' ' * (len(source_lines[0]) - len(source_lines[0].lstrip()))
    if indentation:
        for i, line in enumerate(source_lines):
            source_lines[i] = line.removeprefix(indentation)
        source = '\n'.join(source_lines)
    # Ensure standard indentation (4 spaces)
    source = ast.unparse(ast.parse(source))
    # Convert lambda function to def function
    if func.__name__ == '<lambda>':
        indentation = ' ' * 4
        source = re.subn(
            r'.*(^|\W)lambda (.+?): ?',
            rf'def {func_name}(\g<2>):\n{indentation}return ',
            source,
            1,
        )[0]
    # Remove jit decorator and decorators above it
    module, module_dict = _get_module(func)
    source_lines = source.split('\n')
    for i, line in enumerate(source_lines):
        if match := re.search(r'^@(.+?)(\(|$)', line):
            deco_name = match.group(1).strip()
            try:
                deco_func = eval(deco_name, module_dict)
            except Exception:
                continue
            if deco_func is jit:
                source_lines = source_lines[i+1:]
                break
        elif line.startswith('def '):
            # Arrived at definition line without finding jit decorator.
            # This happens if the "decoration" is done as
            # func = jit(func) rather than using the @jit syntax.
            break
    source = '\n'.join(source_lines)
    return source


# Function for extracting Python module from function object
def _get_module(func):
    module = inspect.getmodule(func)
    module_dict = getattr(module, '__dict__', None)
    if module_dict is None:
        raise ModuleNotFoundError(
            f'Failed to get __dict__ of module {module} '
            f'within which function {func} is defined'
        )
    return module, module_dict


# Function constructing the source of a Cython extension
# module from the source of a Python function.
def _make_cython_source(
    func,
    func_name,
    func_description,
    sig_description,
    source,
    sig,
    types,
    directives,
):
    preamble = [
        '#',  # first comment line can have C code attached to it
        (
            f'# {_this_module_name.capitalize()} version of '
            f'{func_description} with type signature'
        ),
        f'# {sig_description}',
    ]
    excludes = (func_name, 'cython')
    fake_globals = [
        f'{name} = object()'
        for name in inspect.getclosurevars(func).globals
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
    header = [
        '\n# Function to be jitted',
    ]
    if _ccall_allowed(func_name, source, sig):
        header.append('@cython.ccall')
    for directive, val in directives.items():
        header.append(f'@cython.{directive}({val!r})')
    declaration_locals = ', '.join(
        f'{name}={tp}'
        for name, tp in types.items()
        if name not in ('return', _this_module_name)
    )
    declaration_return = types.get('return', 'object')
    if declaration_locals:
        header.append(f'@cython.locals({declaration_locals})')
    header.append(f'@cython.returns({declaration_return})')
    source = '\n'.join(
        itertools.chain(
            preamble,
            header,
            source.split('\n'),
        )
    )
    return source


# Function determining whether the function to be Cythonized
# may be done so using @cython.ccall (cpdef).
def _ccall_allowed(func_name, source, sig):
    # *args and **kwargs not allowed with ccall
    for param in sig.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            return False
    # Closures not allowed with ccall
    class ClosureVisitor(ast.NodeVisitor):
        def contains_closure(self, source):
            self._contains_closure = False
            self.visit(ast.parse(source))
            return self._contains_closure
        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            if node.name != func_name:
                self._contains_closure = True
        def visit_Lambda(self, node):
            self._contains_closure = True
    if ClosureVisitor().contains_closure(source):
        return False
    # No violations found
    return True


# Function for handling Cythonization and C compilation
# of fully constructed Cython source.
def _compile(source, transpilation_hash, optimizations, html, silent):
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
    module_name = f'_{_this_module_name}_module_{transpilation_hash}'
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
        module_path.with_suffix('.pyx').write_text(source, 'utf-8')
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
    # Fetch compiled module from temporary namespace
    module_compiled_dict = namespace[module_name].__dict__
    return module_compiled_dict, source_c, html_annotation


# Function for adding Cython HTML annotations to wrapper functions
def _place_html_annotation(func, wrapper, argument_types, source_c, html_annotation):
    # Facilitate storage and viewing of HTML annotations
    if not wrapper.__cython_html__:
        # Define function for easy viewing of the HTML annotations
        # in the browser.
        def view_cython_html(argument_types=None):
            dir_name = tempfile.mkdtemp()  # not cleaned up by Python
            if argument_types is None:
                pages = wrapper.__cython_html__
            else:
                if not isinstance(argument_types, tuple):
                    argument_types = (argument_types,)
                pages = {argument_types: wrapper.__cython_html__[argument_types]}
            paths_html = []
            for argument_types, html_annotation in pages.items():
                if html_annotation is None:
                    print(f'No Cython HTML annotation generated for {func.__name__}')
                    break
                source_c = wrapper.__cython_c__[argument_types]
                transpilation_hash = _get_transpilation_hash(func, argument_types)
                module_name = f'_{_this_module_name}_module_{transpilation_hash}'
                module_path = pathlib.Path(dir_name) / module_name
                module_path.with_suffix('.c').write_text(source_c, 'utf-8')
                path_html = module_path.with_suffix('.html')
                path_html.write_text(html_annotation, 'utf-8')
                paths_html.append(path_html.as_uri())
            for path_html in paths_html:
                webbrowser.open_new_tab(str(path_html))

        # Add above function as attribute on the wrapper
        wrapper.view_cython_html = view_cython_html
    # Add HTML annotations and C source for the given argument types
    wrapper.__cython_html__[argument_types] = html_annotation
    wrapper.__cython_c__[argument_types] = source_c


# Function for constructing hash of function with argument types
def _get_transpilation_hash(func, argument_types):
    return abs(hash((func, argument_types)))


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
    try:
        return _construct_ndarray_type_info(val)
    except TypeError:
        pass
    return type(val)


# Function responsible for constructing NdarrayTypeInfo instances,
# storing type information of arrays/memoryviews.
def _construct_ndarray_type_info(a):
    def construct(dtype, ndim, c_contig, f_contig):
        if c_contig:
            # Disregard F-contiguousness if C-contiguous
            f_contig = False
        return NdarrayTypeInfo(dtype, ndim, c_contig, f_contig)
    if isinstance(a, np.ndarray):
        return construct(
            a.dtype.type,
            a.ndim,
            a.flags['C_CONTIGUOUS'],
            a.flags['F_CONTIGUOUS'],
        )
    if isinstance(a, type(cython.int[:])):
        return NdarrayTypeInfo(
            _cython_types_reverse[_cython_types[a.dtype]],
            a.ndim,
            a.is_c_contig,
            a.is_f_contig,
        )
    raise TypeError(f'_construct_ndarray_type_info() called with type {type(a)}')


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


# Function for extracting user-defined type annotations.
# The return value is the source without the annotations,
# as well as the stripped off annotations.
def _extract_annotations(func, source, module_dict, func_description):

    class AnnotationExtractor(ast.NodeTransformer):

        def extract(self, source):
            self.annotations = {}
            source = ast.unparse(self.visit(ast.parse(source)))
            return source, self.annotations

        def add_annotation(self, name, tp):
            if not isinstance(name, str):
                name = ast.unparse(name)
            tp = eval(ast.unparse(tp), module_dict)
            tp_old = self.annotations.setdefault(name, tp)
            if tp_old == tp:
                return
            warnings.warn(
                (
                    f'Name {name!r} annotated with different types '
                    f'in jitted {func_description}: {tp}, {tp_old}. '
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
                        f'{ast.unparse(node)!r} in jitted {func_description}'
                    ),
                    RuntimeWarning,
                )
            if not node.value:
                # Annotation-only statement. Remove completely.
                return None
            # Transform to assignment statement without annotation
            return ast.Assign(**(
                vars(node) | {'targets':[node.target], 'value': node.value}
            ))

    # Separate annotations from source
    source, annotations = AnnotationExtractor().extract(source)
    return source, annotations


# Function keeping a global record of types of names
# (arguments as well as locals) for all transpilations.
def _record_types(variables, transpilation_hash, func_description):
    record = _records_types[transpilation_hash]
    for name, val in variables.items():
        tp = _get_type(val)
        tp_old = record.setdefault(name, tp)
        if tp_old == tp:
            continue
        if (
            (tp_old in (bool, int) and tp in (bool, int))
            or (tp_old == float and tp in (int, bool))
            or (tp_old == complex and tp in (float, int, bool))
        ):
            # Allow implicit upcasting of Python scalars
            # as well as casting back and forth between bools and ints.
            record[name] = tp_old
        else:
            warnings.warn(
                (
                    f'Name {name!r} assigned different types '
                    f'in jitted {func_description}: {tp}, {tp_old}. '
                    f'The first will be used'
                ),
                RuntimeWarning,
            )
_records_types = collections.defaultdict(dict)


# Function for obtaining the types of locals within a function
def _record_locals(
    func_name,
    module_dict,
    source,
    transpilation_hash,
    func_description,
    silent_print,
    args,
    kwargs,
):
    @contextlib.contextmanager
    def hack_print(silent_print):
        if not silent_print:
            yield
            return
        print_in_module = ('print' in module_dict)
        print_func = module_dict.get('print')
        module_dict['print'] = lambda *args, **kwargs: None
        yield
        if print_in_module:
            module_dict['print'] = print_func
        else:
            module_dict.pop('print')

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

    # Rename function within source
    func_name_tmp = f'_{_this_module_name}_func_{transpilation_hash}'
    source = ast.unparse(
        FunctionRenamer(func_name, func_name_tmp)
        .visit(ast.parse(source))
    )
    # Add type recording calls to source
    record_str = (
        f'import {_this_module_name}; {_this_module_name}._record_types(locals(), '
        f'{transpilation_hash}, {func_description!r})'
    )
    source = re.sub(
        r'( |;)return($|\W)',
        rf'\g<1>{record_str}; return\g<2>',
        source,
    )
    indentation = ' ' * 4
    source += f'\n{indentation}{record_str}'
    # Define and call modified function within definition module,
    # with recording of the types added in as a side effect.
    # Disable the print() function while doing this,
    # unless silent_print is False.
    with hack_print(silent_print):
        # Define modified function within definition module
        exec(source, module_dict)
        # Call modified function with passed arguments
        return_val = module_dict[func_name_tmp](*args, **kwargs)
        _record_types({'return': return_val}, transpilation_hash, func_description)
    # Remove modified function from definition module
    module_dict.pop(func_name_tmp)
    # The globally stored record of types for the given
    # function call is now complete. Pop it.
    types = _records_types.pop(transpilation_hash, {})
    return types


# Function for merging recorded types with explicitly annotated types.
# Recorded types will be converted to Cython types and all types will
# be transformed to their str representation.
def _merge_types(annotations, types):
    # Transform recorded types to str representation of Cython types
    types = {
        name: _convert_to_cython_type(tp)
        for name, tp in types.items()
    }
    # Transform annotations to str representation of Cython types.
    # Keep unrecognized types as is.
    annotations = {
        name: _convert_to_cython_type(tp, tp)
        for name, tp in annotations.items()
    }
    # Merge recorded types and annotations. On conflicts we keep
    # the annotation without emitting a warning.
    types |= annotations
    return types


# Function returning pretty str representation of type
def _prettify_type(tp):
    if isinstance(tp, NdarrayTypeInfo):
        return str(tp)
    tp_name = getattr(tp, '__name__', getattr(tp, 'name', None))
    if tp_name is None:
        tp_name = str(tp)
    return tp_name


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
    try:
        tp = _construct_ndarray_type_info(tp)
    except TypeError:
        pass
    if isinstance(tp, NdarrayTypeInfo):
        return convert_array_type(tp)
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

