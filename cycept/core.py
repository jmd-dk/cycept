import ast
import collections
import contextlib
import dataclasses
import functools
import pathlib
import re
import sys
import warnings

import cython
import numpy as np

from .call import FunctionCall


# Read current version from pyproject.toml
def get_version():
    version = '?.?.?'
    # Get version from meta data on installed package
    import importlib
    try:
        return importlib.metadata.version('cycept')
    except importlib.metadata.PackageNotFoundError:
        pass
    # Look for pyproject.toml
    def get_tomllib():
        tomllib = None
        try:
            import tomllib
        except ModuleNotFoundError:
            try:
                import toml as tomllib
            except ModuleNotFoundError:
                warnings.warn(
                    (
                        f'For using Cycept with Python < 3.11 please install '
                        f'the toml Python package:\n'
                        f'    {sys.executable} -m pip install toml\n'
                    ),
                    RuntimeWarning,
                )
        return tomllib
    path = pathlib.Path(__file__).resolve().parent
    while True:
        path_pyproject = path / 'pyproject.toml'
        if path_pyproject.is_file():
            tomllib = get_tomllib()
            if tomllib is not None:
                info = tomllib.loads(path_pyproject.read_text('utf-8'))
                version = info['project']['version']
            break
        if path.parent == path:
            break
        path = path.parent
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
                Set to False to display compilation commands.
                Default value is True.


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
        # Get compiled function or result
        func_compiled, result = transpile(func, wrapper, args, kwargs, **options)
        if func_compiled is not None:
            # Got compiled function. Call it to get result.
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
    html=False,
    checks=False,
    clike=False,
    array_args=True,
    directives=None,
    optimizations=None,
):
    if not compile:
        return func, None
    # Fetch function call object, implementing dynamic evaluation
    # of various attributes.
    call = fetch_function_call(func, args, kwargs)
    # If the call has already been transpiled and cached,
    # return immediately.
    if call.compiled is not None:
        return call.compiled.func, None
    # The function call object was not found in cache
    if not silent:
        print(f'Jitting {call!r}')
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
    optimizations = get_optimizations(optimizations)
    call.compile(optimizations, html, silent)
    # Store the function call object on the wrapper function
    wrapper.__cycept__[call.arguments_types] = call
    # Return the result only
    return None, result


# Function used to fetch FunctionCall instance from the global
# cache or instantiating a new one if not found in the cache.
def fetch_function_call(func, args=None, kwargs=None):
    if isinstance(func, int):
        # Called with function call hash
        return cache[func]
    # We need to instantiate a fresh instance to obtain the hash
    call = FunctionCall(func, args, kwargs)
    call_cached = cache.get(call.hash)
    if call_cached is not None:
        # Found in cache. Disregard the freshly created instance.
        return call_cached
    # Not found in cache. Cache and return the freshly created instance.
    cache[call.hash] = call
    return call
cache = {}


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


# Function for setting up compiler optimizations (CFLAGS)
def get_optimizations(optimizations):
    get_defaults = lambda: optimizations_default.copy()
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
optimizations_default = [
    '-DNDEBUG',
    '-O3',
    '-funroll-loops',
    '-ffast-math',
    '-march=native',
]


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
def record_types(call, variables=None):
    if isinstance(call, int):
        # Called with function call hash
        call = fetch_function_call(call)
    if variables is None:
        # Use function call arguments
        variables = call.arguments
    for name, val in variables.items():
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
    record_str = f'{cycept_module_refname}.record_types({call.hash}, locals())'
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
        yield
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


# Mapping of Python/NumPy types to str representations
# of corresponding Cython types.
cython_default_integral = 'cython.Py_ssize_t'  # typically 64-bit
cython_types = collections.defaultdict(
    lambda: 'object',
    {
        # Python scalars
        bool: 'cython.bint',
        int: cython_default_integral,
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
cython_types_reverse = collections.defaultdict(
    lambda: object,
    {name: tp for tp, name in cython_types.items()},
)
# Now add the Cython types themselves
cython_types |= {
    getattr(cython, name.removeprefix('cython.')): name
    for tp, name in cython_types.items()
    if name.startswith('cython.')
}

