import contextlib
import copy
import dataclasses
import datetime
import functools
import inspect
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np


# Number of times to repeat each performance measurement
repeats = 5

# Rough number of seconds to spend on each performance measurement
t_perf = 0.2

# JITs known by this module
names = {
    'python': 'Python',
    'numpy': 'NumPy',
    # JITs
    'cycept': 'Cycept',
    'cython': 'Cython',
    'numba': 'Numba',
}
class Jit:
    def __init__(self, name, version, *decorators):
        self.name = name
        self.version = version
        self.decorators = decorators
@dataclasses.dataclass
class Findings:
    python: float = np.inf
    numpy: float = np.inf
    cycept: float = np.inf
    cython: float = np.inf
    numba: float = np.inf


# Function returning the JITs available on the system
@functools.cache
def get_jits():
    include = lambda name: not setup['jits'] or name in setup['jits']
    jits = {}
    # Cycept
    if include('cycept'):
        try:
            import cycept
        except Exception:
            pass
        else:
            jits['cycept'] = Jit(
                'cycept',
                cycept.__version__,
                functools.partial(cycept.jit, silent=setup['silent']),
            )
    # Cython
    if include('cython'):
        try:
            import cython
        except Exception:
            pass
        else:
            jits['cython'] = Jit('cython', cython.__version__, cython.compile)
            # Set CYTHON_CACHE_DIR to a new temporary directory,
            # ensuring that Cython does not reuse previously compiled modules.
            var = 'CYTHON_CACHE_DIR'
            if var not in os.environ:
                os.environ[var] = tempfile.mkdtemp()
    # Numba (nopython mode, then fallback to object mode)
    if include('numba'):
        try:
            import numba
        except Exception:
            pass
        else:
            jits['numba'] = Jit(
                'numba',
                numba.__version__,
                functools.partial(
                    numba.jit,
                    nopython=True,
                    fastmath=True,
                ),
                functools.partial(
                    numba.jit,
                    forceobj=True,
                ),
            )
    # Print out information
    print('Python', '.'.join(map(str, sys.version_info[:3])))
    if setup['numpy']:
        print('NumPy', np.__version__)
    print('JITs:')
    for jit in jits.values():
        print('*', names[jit.name], jit.version)
    return jits


# Function for running all performance tests
def bench(show_func=False, silent=True):

    @contextlib.contextmanager
    def set_globals():
        setup_backup = copy.deepcopy(setup)
        # Include every JIT found on the system
        setup['jits'].clear()
        # Include NumPy
        setup['numpy'] = True
        # Disable test asserts
        setup['asserts'] = False
        # If show_func, enable printing of the tested source code
        setup['show_func'] = show_func
        # If silent, disable messages during jitting
        setup['silent'] = silent
        try:
            yield
        except Exception:
            raise
        finally:
            setup.clear()
            setup.update(setup_backup)

    # Discover and run test functions within this module
    with set_globals():
        for var, val in globals().copy().items():
            if not var.startswith('test_') and not var.endswith('_test'):
                continue
            if callable(val):
                val()
        # Cleanup
        jits = get_jits()
        if 'cython' in jits:
            cache_dir = os.environ.get('CYTHON_CACHE_DIR')
            if cache_dir is not None and pathlib.Path(cache_dir).is_dir():
                shutil.rmtree(cache_dir, ignore_errors=True)

# Default setup, temporarily changed by the bench() function
setup = {
    'jits': ['cycept'],
    'numpy': False,
    'asserts': True,
    'show_func': False,
    'silent': True,
}


# Main function for performing the performance measurements
def perf(func, *args, **kwargs):
    func_numpy = None
    if isinstance(func, tuple):
        func, func_numpy = func
    prepare = None
    for i, arg in enumerate(args):
        if callable(arg) and getattr(arg, '__name__') == 'prepare':
            prepare = arg
            args = args[:i] + args[i + 1:]
            break
    else:
        for key, val in kwargs.items():
            if key == 'prepare' and callable(val):
                prepare = val
                kwargs.pop(key)
                break

    def measure(name, func):
        calls = determine_calls(func)
        t_best = np.inf
        for _ in range(repeats):
            result, t = run(func, calls)
            t_best = min(t_best, t)
        setattr(results, name, result)
        setattr(timings, name, t_best/calls)
        return calls

    def determine_calls(func):
        result, t = run(func)
        calls = max(int(float(f'{t_perf/t:.0e}')), 1)
        return calls

    def run(func, calls=1):
        if prepare is None:
            tic = time.perf_counter()
            for _ in range(calls):
                result = func(*args, **kwargs)
            toc = time.perf_counter()
            return result, toc - tic
        t_prepare = 0
        tic = time.perf_counter()
        for _ in range(calls):
            tic_prepare = time.perf_counter()
            args_prepare = prepare()
            toc_prepare = time.perf_counter()
            t_prepare += toc_prepare - tic_prepare
            result = func(*args, *args_prepare, **kwargs)
        toc = time.perf_counter()
        return result, (toc - tic) - t_prepare

    def print_timings(name, calls):
        def check_equal():
            result = getattr(results, name)
            result_python = results.python
            same_types = (
                isinstance(result, (int, np.integer))
                and isinstance(result_python, (int, np.integer))
            ) or (
                isinstance(result, (float, np.floating))
                and isinstance(result_python, (float, np.floating))
            ) or type(result) is type(result_python)
            is_equal = False
            if same_types:
                if isinstance(result, np.ndarray):
                    if issubclass(result.dtype.type, np.floating):
                        is_equal = np.allclose(result, result_python)
                    else:
                        is_equal = (result == result_python).all()
                else:
                    if isinstance(result, (float, np.floating)):
                        is_equal = np.isclose(result, result_python)
                    else:
                        is_equal = (result == result_python)
            return is_equal
        def get_speedup():
            t = getattr(timings, name)
            speedup = timings.python/t
            if speedup >= 10:
                return f'{speedup:.0f}'
            elif speedup >= 1:
                return f'{speedup:.1f}'
            return f'{speedup:.2f}'
        maxlen = max(map(len, names.values()))
        s0 = f'⚡ {{:<{maxlen + 1}}}'.format(names[name] + ':')
        t = getattr(timings, name)
        if t == np.inf:
            print(f'{s0} Fails to compile'.replace('⚡', '❌'))
            return
        if name != 'python':
            is_equal = check_equal()
            if not is_equal:
                print(f'{s0} Disagrees with pure Python'.replace('⚡', '❓'))
            if setup['asserts'] and name == 'cycept':
                assert is_equal
            if not is_equal:
                return
        s1 = 's' if calls > 1 else ' '
        s2 = ''
        if name != 'python':
            speedup = get_speedup()
            s2 = f' ({speedup}x)'
        calls_pretty = pretty_int(calls)
        t_pretty = pretty_time(t)
        print(
            f'{s0} {calls_pretty} loop{s1}, best of {repeats}: '
            f'{t_pretty} per loop{s2}'
        )

    def print_heading():
        def dedent(text, indentation='auto'):
            lines = text.split('\n')
            if indentation == 'auto':
                indentation = ' ' * (len(lines[0]) - len(lines[0].lstrip()))
            lines = [
                l for line in lines
                if (l := line.removeprefix(indentation).rstrip())
            ]
            text = '\n'.join(lines)
            return text
        def get_source(func):
            source = dedent(inspect.getsource(func))
            return source
        caller_name = inspect.currentframe().f_back.f_back.f_code.co_name
        hashes = '#' * (1 + setup['show_func'])
        print(f'\n{hashes} {caller_name}')
        if not setup['show_func']:
            return
        doc = dedent(globals()[caller_name].__doc__, ' ' * 4)
        print('\n'.join(f'# {line}' for line in doc.split('\n')))
        source = get_source(func)
        print(source)
        if func_numpy is not None:
            source_numpy = get_source(func_numpy)
            print(source_numpy)
    jits = get_jits()
    print_heading()
    timings = Findings()
    results = Findings()
    # Pure Python
    calls = measure('python', func)
    print_timings('python', calls)
    # NumPy
    if setup['numpy'] and func_numpy is not None:
        calls = measure('numpy', func_numpy)
        print_timings('numpy', calls)
    # JITs
    for name, jit in jits.items():
        for jit_decorator in jit.decorators:
            func_jitted = jit_decorator(func)
            try:
                with silence():
                    run(func_jitted)
            except Exception:
                if name == 'cycept':
                    raise
            else:
                calls = measure(name, func_jitted)
                break
        print_timings(name, calls)
    return timings


# Context manager for silencing stdout and stderr, Python and compiler warnings
@contextlib.contextmanager
def silence():

    @contextlib.contextmanager
    def hack_subprocess_Popen():
        subprocess_Popen = subprocess.Popen
        def Popen(*args, **kwargs):
            kwargs['stdout'] = subprocess.DEVNULL
            kwargs['stderr'] = subprocess.DEVNULL
            return subprocess_Popen(*args, **kwargs)
        subprocess.Popen = Popen
        try:
            yield
        except Exception:
            raise
        finally:
            subprocess.Popen = subprocess_Popen

    if not setup['silent']:
        yield
        return
    with (
        (devnull := open(os.devnull, 'w')),
        contextlib.redirect_stdout(devnull),
        contextlib.redirect_stderr(devnull),
        warnings.catch_warnings(),
        hack_subprocess_Popen(),
    ):
        warnings.simplefilter('ignore')
        yield

# Function for converting int of one significant digit to pretty str
def pretty_int(n):
    superscripts = '⁰¹²³⁴⁵⁶⁷⁸⁹'
    if n < 10_000:
        pretty_i =  str(n)
    else:
        factor, _, exponent = f'{n:.0e}'.partition('e')
        pretty_i = f'{factor}×10{superscripts[int(exponent)]}'
    return f'{pretty_i:>5}'

# Function for converting time interval in seconds to pretty str
def pretty_time(t):
    if t <= 0:
        return 'no time at all'
    if t == np.inf:
        return '∞'
    units = {
        'ns': 1e-9,
        'μs': 1e-6,
        'ms': 1e-3,
        's': 1,
    }
    for unit, ratio in units.items():
        factor = 59.95 if unit == 's' else 999.5
        if t < factor*ratio:
            num = f'{t/ratio:#.3g}'.rstrip('.')
            t_pretty = f'{num} {unit}'
            break
    else:
        t_pretty = str(datetime.timedelta(seconds=int(round(t)))).removeprefix('0:')
    return f'{t_pretty:>7}'


# Test functions below

def test_prime():
    """Compute the n'th prime number.
    This tests the performance of integer operations.
    """
    def f(n):
        count = 0
        i = 1
        while True:
            i += 1
            for j in range(2, i):
                if i % j == 0:
                    break
            else:
                count += 1
                if count == n:
                    return i
    n = 500
    timings = perf(f, n)
    if setup['asserts']:
        assert timings.cycept < timings.python / 4


def test_wallis():
    """Compute π using the Wallis product.
    This tests the performance of integer and floating-point operations.
    """
    def f(n):
        i : np.int64
        π = 2
        for i in range(1, n):
            π *= 4 * i ** 2 / (4 * i ** 2 - 1)
        return π
    def f_numpy(n):
        a = 4 * np.arange(1, n, dtype=np.int64) ** 2
        return 2 * np.prod(a / (a - 1))
    n = 2_000_000
    timings = perf((f, f_numpy), n)
    if setup['asserts']:
        assert timings.cycept < timings.python / 50


def test_leibniz():
    """Compute π using the Leibniz formula.
    This tests the performance of floating-point operations.
    """
    def f(n):
        π = 1
        sgn = 1
        for i in range(3, n, 2):
            sgn = -sgn
            π += sgn / i
        π *= 4
        return π
    def f_numpy(n):
        denom = np.arange(1, n, 2, dtype=np.int64)
        num = np.ones_like(denom)
        num[1::2] = -1
        return 4 * (num / denom).sum()
    n = 3_000_000
    timings = perf((f, f_numpy), n)
    if setup['asserts']:
        assert timings.cycept < timings.python / 10


def test_fibonacci():
    """Compute the n'th Fibonacci number using recursion.
    This tests the performance of recursive function calls.
    """
    def f(n):
        if n < 2:
            return n
        return f(n - 2) + f(n - 1)
    n = 30
    timings = perf(f, n)
    if setup['asserts']:
        assert timings.cycept < timings.python / 25


def test_mostcommon():
    """Find the most common object in a list.
    This tests the performance of pure Python operations.
    """
    def f(objs) -> object:
        obj: object
        objs = objs.copy()
        counter = {}
        while objs:
            obj = objs.pop()
            if not obj:
                continue
            if obj not in counter:
                counter[obj] = 0
            counter[obj] += 1
        n = max(counter.values())
        for obj, count in counter.items():
            if count == n:
                return obj
    n = 100
    objs =  1 * n * list(np.arange(n, dtype=np.int64))
    objs += 2 * n * list(np.linspace(0, 1, n, dtype=np.float64))
    objs += 3 * n * [None, 'hello', True, (0, 1, 2), 'hello']
    timings = perf(f, objs)
    if setup['asserts']:
        assert timings.cycept < timings.python


def test_life():
    """Evolve a glider in Conway's Game of Life.
    This tests the performance of pure Python operations and closures.
    """
    def f(n):
        glider = {(0, 2), (1, 0), (1, 2), (2, 1), (2, 2)}
        def evolve(state):
            get_neighbors = lambda x, y: {
                (x + dx, y + dy)
                for dx in range(-1, 2)
                for dy in range(-1, 2)
                if not dx == dy == 0
            }
            squares = state.copy()
            for x, y in state:
                squares |= get_neighbors(x, y)
            state_new = set()
            for x, y in squares:
                neighbors = get_neighbors(x, y)
                count = len(neighbors & state)
                if count == 3 or (count == 2 and (x, y) in state):
                    state_new.add((x, y))
            return state_new
        state = glider.copy()
        for _ in range(4 * n):
            state = evolve(state)
        return state
    n = 30
    timings = perf(f, n)
    if setup['asserts']:
        assert timings.cycept < timings.python


def test_array_0():
    """Compute the value sum((a - b)**2) with a and b being arrays.
    This tests the performance of array indexing.
    """
    def f(a, b):
        x = 0
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                x += (a[i, j] - b[i, j]) ** 2
        return x
    def f_numpy(a, b):
        return ((a - b)**2).sum()
    m, n = 1_300, 1_400
    a = np.linspace(0, 1, m * n, dtype=np.float64).reshape((m, n))
    b = np.linspace(1, 0, m * n, dtype=np.float64).reshape((m, n))
    timings = perf((f, f_numpy), a, b)
    if setup['asserts']:
        assert timings.cycept < timings.python / 50


def test_array_1():
    """Transform array in-place as a[i, j] /= i**2 + j**2.
    This tests the performance of array indexing.
    """
    def f(a):
        for i in range(a.shape[0]):
            for j in range(i == 0, a.shape[1]):
                a[i, j] /= i**2 + j**2
        return a
    def f_numpy(a):
        i2 = np.arange(a.shape[0]) ** 2
        j2 = np.arange(a.shape[1]) ** 2
        k2 = i2[:, None] + j2[None, :]
        k2[0, 0] = 1
        a /= k2
        return a
    def prepare():
        m, n = 1_100, 1_300
        a = np.linspace(0, 1, m * n, dtype=np.float64).reshape((m, n))
        return a,
    timings = perf((f, f_numpy), prepare)
    if setup['asserts']:
        assert timings.cycept < timings.python / 50


def test_matmul():
    """Compute the matrix multiplication a @ b.
    This tests the performance of array indexing.
    Here NumPy is expected to be the fastest due to its
    much more sophisticated implementation.
    """
    def f(a, b):
        m, n = a.shape
        p, q = b.shape
        b = b.transpose().copy()
        c = np.empty((m, q), dtype=a.dtype)
        for i in range(m):
            for j in range(q):
                val = 0
                for k in range(n):
                    val += a[i, k] * b[j, k]
                c[i, j] = val
        return c
    def f_numpy(a, b):
        return a @ b
    m, n = 150, 350
    p, q = n, 250
    a = np.linspace(0, 1, m * n, dtype=np.float64).reshape((m, n))
    b = np.linspace(0, 1, p * q, dtype=np.float64).reshape((p, q))
    timings = perf((f, f_numpy), a, b)
    if setup['asserts']:
        assert timings.cycept < timings.python / 200


def test_mandelbrot():
    """Compute an image of the Mandelbrot set.
    This tests the performance of complex numbers and iteration.
    """
    def f(x_min, x_max, y_min, y_max, n_max, image):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                x = x_min + i*(x_max - x_min)/(image.shape[0] - 1)
                y = y_min + j*(y_max - y_min)/(image.shape[1] - 1)
                c = x + y*1j
                z = 0j
                for n in range(n_max):
                    z = z * z + c
                    if abs(z) >= 2:
                        break
                image[i, j] = n
        return image
    x_min, x_max = -2, 0.5
    y_min, y_max = -1.2, 1.2
    n_max = 30
    width = 300
    height = int(width*(y_max - y_min)/(x_max - x_min))
    image = np.empty((width, height), dtype=np.uint8)
    timings = perf(f, x_min, x_max, y_min, y_max, n_max, image)
    if setup['asserts']:
        assert timings.cycept < timings.python / 30


def test_trapz():
    """Compute a definite integral using the trapezoidal rule.
    This tests the performance of floating-point operations.
    """
    def f(x, y):
        s = 0
        for i in range(1, x.shape[0]):
            s += (y[i - 1] + y[i]) * (x[i] - x[i - 1])
        s /= 2
        return s
    def f_numpy(x, y):
        return np.trapz(y, x)
    n, m = 100_001, 20
    x = np.linspace(0, n * np.pi, n * m)
    x += np.sin(x)
    y = np.sin(x)
    timings = perf((f, f_numpy), x, y)
    if setup['asserts']:
        assert timings.cycept < timings.python / 200


def test_nbody():
    """Perform an N-body simulation.
    This tests the performance of array indexing
    and floating-point operations.
    """
    def f(r, v, eps, steps, dt=1e-2):
        n, ndim = r.shape
        d = np.empty(3, dtype=np.float64)
        for t in range(steps):
            for i in range(n):
                for j in range(i + 1, n):
                    r2 = 0
                    for dim in range(ndim):
                        d[dim] = r[i, dim] - r[j, dim]
                        r2 += d[dim] ** 2
                    r3_inv_softened = (r2 + eps ** 2) ** (-1.5)
                    for dim in range(ndim):
                        f = -d[dim] * r3_inv_softened
                        v[i, dim] += f
                        v[j, dim] -= f
            for i in range(n):
                for dim in range(ndim):
                    r[i, dim] += v[i, dim] * dt
    def prepare():
        n = 8
        ndim = 3
        eps = 0.1
        steps = 2
        r = (
            np.asarray(np.meshgrid(*[np.arange(n, dtype=np.float64)]*ndim))
            .reshape((ndim, n**ndim)).T.copy()
        )
        r += eps*np.sin(2*np.pi*r)
        v = np.zeros_like(r)
        return r, v, eps, steps
    timings = perf(f, prepare)
    if setup['asserts']:
        assert timings.cycept < timings.python / 200

