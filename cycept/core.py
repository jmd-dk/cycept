import ast
import collections
import contextlib
import dataclasses
import functools
import importlib
import importlib.metadata
import inspect
import pathlib
import re
import sys
import time
import unicodedata
import warnings

import cython
import numpy as np

from .call import FunctionCall


# Get current version
def get_version():
    version = '?.?.?'
    # Get version from pyproject.toml
    def get_tomllib():
        toml_packages = ['tomllib', 'tomli', 'toml']
        for name in toml_packages:
            try:
                return importlib.import_module(name)
            except ModuleNotFoundError:
                pass
        warnings.warn(
            (
                f'For using Cycept with Python < 3.11 please install '
                f'the tomli Python package:\n'
                f'    {sys.executable} -m pip install tomli\n'
            ),
            RuntimeWarning,
        )
    path = pathlib.Path(__file__).resolve().parent
    while True:
        if (path_pyproject := path / 'pyproject.toml').is_file():
            if (tomllib := get_tomllib()) is not None:
                try:
                    info = tomllib.loads(path_pyproject.read_text('utf-8'))
                    if info['project']['name'] == 'cycept':
                        return info['project']['version']
                except Exception:
                    pass
            break
        if path.parent == path:
            break
        path = path.parent
    # Get version from meta data on installed package
    try:
        return importlib.metadata.version('cycept')
    except ModuleNotFoundError:
        pass
    return version
__version__ = get_version()


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
            compile: bool
                Set to False to disable just-in-time compilation.
                Default is True.


            silent: bool
                Set to False to display compilation messages.
                Default value is True.


            c_lang: str
                Set this to 'c' or 'c++' to select the language to which the
                Python function will be transpiled.
                Default value is 'c'.


            html: bool
                Set to True to produce HTML annotations of the compiled code.
                View the HTMl using func.__cycept__().
                Default value is False.


            checks: bool
                Set to True to enable the following checks within Cython:
                    * boundscheck
                    * initializedcheck
                Note that using

                    @jit(checks=True)

                is equivalent to

                    @jit(directives={'boundscheck': True, 'initializedcheck', True})

                Default value is False.


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


            array_args: bool
                Local NumPy arrays used within the function are always
                converted to Cython memoryviews, allowing for fast indexing.
                If such variables are used as arguments to other functions,
                passing them as memoryviews instead of NumPy arrays may not
                work. To be safe, all memoryview variables are thus converted
                back to NumPy arrays before used as arguments. Set this option
                to False to prevent the conversion back to NumPy arrays.
                Default value is True.


            directives: dict[str, bool]
                Allows for adding Cython directives to the transpiled
                function, typically in order to achieve further speedup.
                To e.g. disable boundschecking of indexing operations:

                    @jit(directives={'boundscheck': False})

                You can read about the different directives here:
                https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#compiler-directives


            optimizations: dict[str, bool]
                Specifies which compiler optimizations to enable/disable.
                Cycept defines the following optimizations, here written with
                their GCC equivalent to their right:
                    * base: -O3 -funroll-loops
                    * fastmath: -ffast-math
                    * native: -march=native
                To disable e.g. fastmath, use

                    @jit(optimizations={'fastmath': False})

                The default value for all optimizations is True.


            integral: type | str
                By default, Python ints are replaced with cython.Py_ssize_t.
                These are not fully equivalent: while cython.Py_ssize_t is
                typically 64-bit, Python ints are unbounded. If you prefer
                to e.g. use 32-bit ints, set this option to e.g. np.int32:

                    @jit(integral=np.int32)

                To use the native int of the current machine, you can specify
                any of cython.int, np.intc, 'int'. To use the (slow) Python
                ints, specify integral=object.


            floating: type | str
                By default, Python floats are replaced with cython.double.
                These are fully identical 64-bit floating-point numbers.
                If you prefer to e.g. use 64-bit floats, set this option
                to e.g. np.float32:

                    @jit(floating=np.float32)


            floating_complex: type | str
                By default, Python complex floats are replaced with
                cython.complex. These should be fully identical 128-bit (2×64)
                complex floating-point numbers. If you prefer to e.g. use
                64-bit (2×32) complex floats, set this option to e.g. np.complex64:

                    @jit(floating_complex=np.complex64)


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
        # Get compiled function or result
        func_compiled, result = transpile(func, wrapper, args, kwargs, **options)
        if func_compiled is not None:
            # Got compiled function. Call it to get result.
            kwargs = {asciify(var): val for var, val in kwargs.items()}
            result = func_compiled(*args, **kwargs)
        # If a Cython memoryview is returned, convert to NumPy array
        if 'memoryview' in str(type(result)).lower():
            result = np.asarray(result)
        return result

    # Add storage for function call objects on wrapper
    wrapper.__cycept__ = CyceptStorage()
    return wrapper


# Dict-like class for storing function call objects on the wrapper
class CyceptStorage(dict):
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
def transpile(
    func,
    wrapper,
    args,
    kwargs,
    *,
    compile=True,
    silent=True,
    c_lang=None,
    html=False,
    checks=False,
    clike=False,
    array_args=True,
    directives=None,
    optimizations=None,
    integral=None,
    floating=None,
    floating_complex=None,
):
    if not compile:
        return func, None
    # Fetch function call object, implementing dynamic evaluation
    # of various attributes.
    call = fetch_function_call(func, wrapper, args, kwargs)
    # If the call has already been transpiled and cached,
    # return immediately.
    if call.compiled is not None:
        return call.compiled.func, None
    # The function call object was not found in cache
    tic = time.perf_counter()
    # Print jitting message if not running silently
    if not silent:
        print(f'Jitting {call!r}')
    # Populate global mappings of Cython types in accordance
    # with user integral and floating specifications.
    cython_types_user, cython_types_reverse_user = get_cython_types(
        integral,
        floating,
        floating_complex,
    )
    cython_types.clear()
    cython_types.update(cython_types_user)
    cython_types_reverse.clear()
    cython_types_reverse.update(cython_types_reverse_user)
    # Transform Unicode names to ASCII names
    # Record types of passed arguments and locals. As a side effect,
    # the function is called and the return value obtained.
    record_types(call)
    result = record_locals(call)
    # Make sure that NumPy arrays are treated as such when necessary
    call.protect_arrays(array_args)
    # Convert Python function source into Cython module source
    directives = get_directives(directives, checks, clike)
    call.to_cython(directives)
    # Cythonize and compile
    time_compile = call.compile(optimizations, c_lang, html, silent)
    # Store the function call object on the wrapper function
    wrapper.__cycept__[call.arguments_types] = call
    # End and store time measurements
    toc = time.perf_counter()
    time_total = toc - tic
    call.time = call.Time(time_compile, time_total)
    if not silent:
        print(f'Compilation time:   {call.time.compile:#.4g} s')
        print(f'Total jitting time: {call.time.total:#.4g} s')
    # Return the result only
    return None, result


# Function used to fetch FunctionCall instance from the global
# cache or instantiating a new one if not found in the cache.
def fetch_function_call(func, wrapper=None, args=None, kwargs=None):
    if isinstance(func, int):
        # Called with function call hash
        return cache[func]
    # Fast lookup if called with the same objects as last time
    ids = (
        id(func),
        id(wrapper),
        *(id(arg) for arg in args),
        *kwargs.keys(),
        *(id(kwarg) for kwarg in kwargs.values()),
    )
    loot = cache.get('last')
    if loot is not None and loot[1] == ids:
        return loot[0]
    # We need to instantiate a fresh instance to obtain the hash
    call = FunctionCall(func, wrapper, args, kwargs)
    call_cached = cache.get(call.hash)
    if call_cached is not None:
        # Found in cache. Disregard the freshly created instance.
        return call_cached
    # Not found in cache. Cache and return the freshly created instance.
    cache[call.hash] = call
    cache['last'] = call, ids  # special additional store
    return call
cache = {}


# Function converting a string containing Unicode characters
# into a corresponding string using only ASCII.
def asciify(text):
    text_ascii = []
    chars = ''
    for char in text:
        if ord(char) < 128:
            text_ascii.append(char)
            continue
        chars += char
        try:
            unicode_char_name = unicodedata.name(chars)
        except Exception:
            continue
        text_ascii.append(asciify_regex.sub(asciify_repl, unicode_char_name))
        chars = ''
    return ''.join(text_ascii)
asciify_subs = {
    ' ': '_',
    '-': '_',
    '^': 'UNICODE_',
}
asciify_subs[''] = asciify_subs['^']
asciify_regex = re.compile('|'.join(asciify_subs).strip('|'))
asciify_repl = lambda match: asciify_subs[match[0]]


# Function for setting up Cython directives
def get_directives(directives, checks, clike):
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


# Function for extracting the Python/NumPy type off of a value
def get_type(val):
    tp = construct_ndarray_type_info(val)
    if tp is not None:
        return tp
    return type(val)


# Function responsible for constructing NdarrayTypeInfo instances,
# storing type information of arrays/memoryviews.
def construct_ndarray_type_info(a):
    def construct(dtype, ndim, c_contig, f_contig):
        if c_contig:
            # Disregard F-contiguousness if C-contiguous
            f_contig = False
        return NdarrayTypeInfo(dtype, ndim, c_contig, f_contig)
    if isinstance(a, NdarrayTypeInfo):
        return a
    if isinstance(a, numpy_array_type):
        return construct(
            a.dtype.type,
            a.ndim,
            a.flags['C_CONTIGUOUS'],
            a.flags['F_CONTIGUOUS'],
        )
    if isinstance(a, cython_array_type):
        return construct(
            cython_types_reverse[_cython_types[a.dtype]],
            a.ndim,
            a.is_c_contig,
            a.is_f_contig,
        )
    # Passed value is not an array
    return None
numpy_array_type = np.ndarray
cython_array_type = type(cython.int[:])


# Type used to discern different kinds of arrays (memoryviews)
@dataclasses.dataclass(frozen=True)
class NdarrayTypeInfo:
    dtype: np.dtype.type
    ndim: int
    c_contig: bool
    f_contig: bool

    @property
    def dtype_cython(self):
        return convert_to_cython_type(self.dtype)

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
def record_types(call, variables=None, do_record=True):
    if isinstance(call, int):
        # Called with function call hash
        call = fetch_function_call(call)
    if variables is None:
        # Use function call arguments
        variables = call.arguments
    for name, val in variables.items():
        if inspect.ismodule(val) or callable(val):
            # Modules, closure functions and classes should not be
            # assigned a type (would have been object).
            call.locals_excludes.add(name)
            # Similarly, non-local variables used within closures should not
            # be assigned a type. Doing so will lead Cython to perform
            # scoping violations.
            try:
                closurevars = inspect.getclosurevars(val)
            except TypeError:
                pass
            else:
                call.locals_excludes |= set(closurevars.nonlocals)
            continue
        # If this call is made from a closure instead of the outer function
        # to be jitted, do_record will be False. In this case, none of the
        # supplied variables should participate amongst the types variables.
        if not do_record:
            continue
        tp = get_type(val)
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
def record_locals(call):
    # Create copy of function source with the function renamed
    class FunctionRenamer(ast.NodeTransformer):
        def __init__(self, name_old, name_new):
            self.name_old = name_old
            self.name_new = name_new
            self.visit_Name = self.make_visitor('id')
            self.visit_FunctionDef = self.make_visitor('name')
        def make_visitor(self, attr):
            def rename(node):
                node = self.generic_visit(node)
                if getattr(node, attr) == self.name_old:
                    node = type(node)(**(vars(node) | {attr: self.name_new}))
                return node
            return rename
    func_name_tmp = f'_cycept_func_{call.hash}'
    source = ast.unparse(
        FunctionRenamer(call.func_name, func_name_tmp).visit(call.ast)
    )
    call._ast = None  # invalidate AST after renaming
    # Add type recording calls to source
    cycept_module_refname = '__cycept__'
    record_str = (
        f'import inspect; '
        f'{cycept_module_refname}.record_types('
        f'{call.hash}, locals(), inspect.currentframe().f_code.co_name == \'{func_name_tmp}\''
        f')'
    )
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
    # the function to be jitted. Also add non-local objects referenced by
    # the function directly to the module.
    @contextlib.contextmanager
    def hack_module_dict():
        call.module_dict[cycept_module_refname] = sys.modules['cycept.core']
        nonlocals_ori = {}
        for name, val in call.nonlocals.items():
            if name in call.module_dict:
                nonlocals_ori[name] = call.module_dict[name]
            call.module_dict[name] = val
        try:
            yield
        except Exception:
            raise
        finally:
            call.module_dict.pop(cycept_module_refname)
            for name in call.nonlocals:
                call.module_dict.pop(name)
            for name, val in nonlocals_ori.items():
                call.module_dict[name] = val
    with hack_module_dict():
        # Define modified function within definition module
        exec(source, call.module_dict)
        # Call modified function with passed arguments
        return_val = call.module_dict[func_name_tmp](*call.args, **call.kwargs)
    # Remove modified function from definition module
    call.module_dict.pop(func_name_tmp)
    # Add return type to record
    record_types(call, {'return': return_val})
    # Return the return value from the function call
    return return_val


# Function returning pretty str representation of type
def prettify_type(tp):
    if isinstance(tp, NdarrayTypeInfo):
        return str(tp)
    tp_name = getattr(tp, '__name__', getattr(tp, 'name', None))
    if tp_name is not None:
        return tp_name
    return str(tp)


# Function for converting a Python/NumPy/Cython type to the
# str representation of the corresponding Cython type.
def convert_to_cython_type(tp, default=object):
    def convert_array_type(tp):
        if cython_types[tp.dtype] == 'object':
            return 'object'
        return tp.__name__
    # Type already given as str
    if isinstance(tp, str):
        return repr(tp)
    # NumPy array or Cython memoryview
    tp_arr = construct_ndarray_type_info(tp)
    if tp_arr is not None:
        return convert_array_type(tp_arr)
    # Python/NumPy/Cython
    tp_str = cython_types.get(tp)
    if tp_str is not None:
        return tp_str
    # Unrecognized
    if default is not None and tp is not default:
        return convert_to_cython_type(default, None)
    tp_str = getattr(tp, '__name__', getattr(tp, 'name', None))
    if tp_str is not None:
        return tp_str
    return str(tp)


# Function creating a mapping of Python/NumPy types
# to str representations of corresponding Cython types.
@functools.cache
def get_cython_types(integral=None, floating=None, floating_complex=None):
    if integral is None:
        integral = cython_default_integral
    if floating is None:
        floating = cython_default_floating
    if floating_complex is None:
        floating_complex = cython_default_floating_complex
    # Mapping of Cython types
    cython_types = {
        # Python scalars
        bool: 'cython.bint',
        int: integral,
        float: floating,
        complex: floating_complex,
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
    }
    # Add NumPy scalar types. Some NumPy types may not be
    # available at runtime and some Cython types may not be
    # available at compile time.
    for name, val in {
        # NumPy Boolean
        'bool_': 'cython.uchar', # uchar is preferable to bint as otherwise Cython throws "ValueError: Buffer dtype mismatch, expected 'bint' but got 'bool'"
        # NumPy signed integral scalars
        'int8': 'cython.schar',
        'int16': 'cython.short',
        'int32': 'cython.int',
        'int64': 'cython.longlong',
        # NumPy unsigned integral scalars
        'uint8': 'cython.uchar',
        'uint16': 'cython.ushort',
        'uint32': 'cython.uint',
        'uint64': 'cython.ulonglong',
        # NumPy floating scalars
        'float32': 'cython.float',
        'float64': 'cython.double',
        'longdouble': 'cython.longdouble',  # platform dependent
        # NumPy complex floating scalars
        'complex64': 'cython.floatcomplex',
        'complex128': 'cython.doublecomplex',
        'complex256': 'cython.longdoublecomplex',  # numpy.longdoublecomplex not defined
    }.items():
        if (attr := getattr(np, name, None)) is not None:
            cython_types[attr] = val
    # Also create reverse mapping
    cython_types_reverse = {
        name: tp for tp, name in cython_types.items()
    }
    # Now add the Cython types themselves
    cython_types |= {
        getattr(cython, name.removeprefix('cython.')): name
        for tp, name in cython_types.items()
        if isinstance(name, str) and name.startswith('cython.')
    }
    # Ensure that the provided integral and floating types
    # are resolved to str representation of Cython types.
    cython_types[int] = cython_types.get(integral, integral)
    cython_types[float] = cython_types.get(floating, floating)
    cython_types[complex] = cython_types.get(floating_complex, floating_complex)
    cython_types_reverse[cython_types[int]] = int
    cython_types_reverse[cython_types[float]] = float
    cython_types_reverse[cython_types[complex]] = complex
    return cython_types, cython_types_reverse
# Global mappings for the Cython types
cython_types = collections.defaultdict(lambda: 'object')
cython_types_reverse = collections.defaultdict(lambda: object)
# Default integral and floating type to use
cython_default_integral = 'cython.Py_ssize_t'       # typically 64-bit
cython_default_floating = 'cython.double'           # 64-bit
cython_default_floating_complex = 'cython.complex'  # should always be 128-bit

