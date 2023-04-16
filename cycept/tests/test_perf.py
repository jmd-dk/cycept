import contextlib
import dataclasses
import datetime
import functools
import inspect
import os
import time
import warnings

import numpy as np


# Number of times to repeat the measurement
repeats = 5

# Rough minimum and maximum number of seconds to spend
# on each pure Python performance measurement.
t_perf = (0.1, 10)

# Minimum number of loop iterations deemed appropriate
calls_appropriate = 100

# Minimum number of loop iterations allowed
calls_min = 1

# JITs known by this module
names = {
    'python': 'pure Python',
    'numpy': 'NumPy',
    'cycept': 'Cycept',
    'cython': 'Cython',
    'numba': 'Numba',
}
class Jit:
    def __init__(self, *decorators, suppress_compiler_warnings=False):
        self.decorators = decorators
        self.suppress_compiler_warnings = suppress_compiler_warnings
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
    jits = {}
    # Cycept
    try:
        import cycept
    except Exception:
        pass
    else:
        jits['cycept'] = Jit(cycept.jit)
    # Cython
    try:
        import cython
    except Exception:
        pass
    else:
        jits['cython'] = Jit(cython.compile, suppress_compiler_warnings=True)
    # Numba (nopython mode, then fallback to object mode)
    try:
        import numba
    except Exception:
        pass
    else:
        jits['numba'] = Jit(
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
    print('JITs:', ', '.join(names[name] for name in jits))
    return jits


# Function for running all performance tests without using pytest.
# (timings will be shown this way).
def bench(show_func=False):
    @contextlib.contextmanager
    def set_globals():
        global test_asserts, print_source
        test_asserts_backup = test_asserts
        print_source_backup = print_source
        # Disable test asserts when not testing with pytest
        test_asserts = False
        # If show_func, enable printing of the tested source code
        if show_func:
            print_source = True
        yield
        test_asserts = test_asserts_backup
        print_source = print_source_backup
    # Discover and run test functions within this module
    with set_globals():
        for var, val in globals().copy().items():
            if not var.startswith('test_') and not var.endswith('_test'):
                continue
            if callable(val):
                val()
test_asserts = True
print_source = False


# Main function for performing the performance measurements
def perf(func, *args, **kwargs):
    func_numpy = None
    if isinstance(func, tuple):
        func, func_numpy = func

    def run(func, calls=1):
        tic = time.perf_counter()
        for _ in range(calls):
            result = func(*args, **kwargs)
        toc = time.perf_counter()
        return result, toc - tic

    def measure(name, func, calls=1):
        t_best = np.inf
        for _ in range(repeats):
            result, t = run(func, calls)
            t_best = min(t_best, t)
        setattr(results, name, result)
        setattr(timings, name, t_best)

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
        s0 = f'* {{:<{maxlen + 1}}}'.format(names[name] + ':')
        t = getattr(timings, name)
        if t == np.inf:
            print(f'{s0} Fails to compile')
            return
        if name != 'python':
            is_equal = check_equal()
            if not is_equal:
                print(f'{s0} Disagrees with pure Python')
            if test_asserts and name == 'cycept':
                assert is_equal
            if not is_equal:
                return
        s1 = s2 = ''
        if calls > 1:
            s1 = 's'
        if name != 'python':
            speedup = get_speedup()
            s2 = f' ({speedup}x)'
        t_pretty = stringify_time(t)
        print(f'{s0} {calls} loop{s1}, best of {repeats}: {t_pretty:<7} per loop{s2}')

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
        hashes = '#' * (1 + print_source)
        print(f'\n{hashes} {caller_name}')
        if not print_source:
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
    result, t = run(func)
    calls = int(float(f'{t_perf[0]/(repeats*t):.0e}'))
    if calls < calls_appropriate:
        calls = int(float(f'{t_perf[1]/(repeats*t):.0e}'))
        calls = min(calls, calls_appropriate)
    calls = max(calls, calls_min)
    measure('python', func, calls)
    print_timings('python', calls)
    # NumPy
    if func_numpy is not None:
        measure('numpy', func_numpy, calls)
        print_timings('numpy', calls)
    # Jits
    for name, jit in jits.items():
        for jit_decorator in jit.decorators:
            func_jitted = jit_decorator(func)
            try:
                with silence(jit.suppress_compiler_warnings):
                    run(func_jitted)  # to compile
            except Exception:
                if name == 'cycept':
                    raise
            else:
                measure(name, func_jitted, calls)
                break
        print_timings(name, calls)
    return timings


# Context manager for silencing stdout and stderr, Python and compiler warnings
@contextlib.contextmanager
def silence(silence_compiler_warnings=False):

    @contextlib.contextmanager
    def hack_cflags():
        suppress_warnings = {
            'gcc': '-w',
            'clang': '-Wno-everything',
        }
        if not silence_compiler_warnings:
            yield
            return
        cflags = ''
        cflags_backup = None
        if 'CFLAGS' in os.environ:
            cflags = os.environ['CFLAGS']
            cflags_backup = cflags
        os.environ['CFLAGS'] = ' '.join([cflags, *suppress_warnings.values()])
        yield
        if cflags_backup:
            os.environ['CFLAGS'] = cflags_backuo
        else:
            os.environ.pop('CFLAGS')

    with (
        (devnull := open(os.devnull, 'w')),
        contextlib.redirect_stdout(devnull),
        contextlib.redirect_stderr(devnull),
        warnings.catch_warnings(),
        hack_cflags(),
    ):
        warnings.simplefilter("ignore")
        yield


# Function for converting time interval in seconds to pretty str
def stringify_time(t):
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
            return f'{num} {unit}'
    return str(datetime.timedelta(seconds=int(round(t)))).removeprefix('0:')


# Test functions below

def test_prime():
    """Computes the n'th prime number.
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
    n = 1_000
    timings = perf(f, n)
    if test_asserts:
        assert timings.cycept < timings.python / 4


def test_wallis():
    """Computes π using the Wallis product.
    This tests the performance of floating-point operations.
    """
    def f(n):
        π = 2
        for i in range(1, n):
            π *= 4 * i ** 2 / (4 * i ** 2 - 1)
        return π
    def f_numpy(n):
        a = 4 * np.arange(1, n) ** 2
        return 2 * np.prod(a / (a - 1))
    n = 100_000
    timings = perf((f, f_numpy), n)
    if test_asserts:
        assert timings.cycept < timings.python / 50


def test_fibonacci():
    """Computes the n'th Fibonacci number using recursion.
    This tests the performance of recursive function calls.
    """
    def f(n):
        if n < 3:
            return 1
        return f(n - 1) + f(n - 2)
    n = 30
    timings = perf(f, n)
    if test_asserts:
        assert timings.cycept < timings.python / 25


def test_mostcommon():
    """Finds the most common object in a list.
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
    objs =  1 * n * list(np.arange(n))
    objs += 2 * n * list(np.linspace(0, 1, n))
    objs += 3 * n * [None, 'hello', True, (0, 1, 2), 'hello']
    timings = perf(f, objs)
    if test_asserts:
        assert timings.cycept < timings.python


def test_life():
    """Evolves a glider in Conway's Game of Life.
    This tests the performance of pure Python operations and closures.
    """
    def f(n):
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
        state = {(0, 2), (1, 0), (1, 2), (2, 1), (2, 2)}
        for _ in range(4 * n):
            state = evolve(state)
    n = 100
    timings = perf(f, n)
    if test_asserts:
        assert timings.cycept < timings.python


def test_array():
    """Computes the value sum((a - b)**2) with a and b being arrays.
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
    n = 1_000
    a = np.linspace(0, 1, n ** 2, dtype=np.float64).reshape((n, n))
    b = np.linspace(1, 0, n ** 2, dtype=np.float64).reshape((n, n))
    timings = perf((f, f_numpy), a, b)
    if test_asserts:
        assert timings.cycept < timings.python / 50


def test_matmul():
    """Computes the matrix multiplication a @ b.
    This tests the performance of array indexing.
    Note that the NumPy implementation is by far the fastest
    (expected as its implementation is much more sophisticated).
    """
    def f(a, b):
        c = np.empty((a.shape[0], b.shape[1]), dtype=a.dtype)
        for i in range(a.shape[0]):
            for j in range(b.shape[1]):
                val = 0
                for k in range(a.shape[1]):
                    val += a[i, k] * b[k, j]
                c[i, j] = val
        return c
    def f_numpy(a, b):
        return a @ b
    m, n = 100, 50
    p, q = n, 200
    a = np.linspace(0, 1, m * n, dtype=np.float64).reshape((m, n))
    b = np.linspace(0, 1, p * q, dtype=np.float64).reshape((p, q))
    timings = perf((f, f_numpy), a, b)
    if test_asserts:
        assert timings.cycept < timings.python / 500

