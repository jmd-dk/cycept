from .core import jit, NdarrayTypeInfo, __version__
from .tests import bench, test


# Function for checking whether Cycept functions correctly
def check(silent_if_ok=False):
    import sys
    import timeit
    import numpy as np
    def get_test_func():
        def wallis(n):
            π = 2
            for i in range(1, n):
                π *= 4 * i ** 2 / (4 * i ** 2 - 1)
            return π
        return wallis
    func = get_test_func()
    n = 10_000
    calls = 100
    ok = True
    try:
        func_jitted = jit(func, silent=True)
        func_jitted(n)
    except Exception:
        ok = False
    if not ok:
        print('Cycept failed to compile test function', file=sys.stderr)
    if ok and not np.isclose(func(n), func_jitted(n)):
        ok = False
        print(
            'Cycept compiled the test function without errors,',
            'but the return value of jitted function is inconsistent',
            'with that of the pure Python version',
            file=sys.stderr,
        )
    if ok:
        t_python = min(timeit.repeat(lambda: func(n), number=calls))
        t_cycept = min(timeit.repeat(lambda: func_jitted(n), number=calls))
        if t_cycept > t_python:
            ok = False
            print(
                'Cycept compiled the test function without errors,',
                'but the jitted function is slower than the pure Python version',
                file=sys.stderr,
            )
    if ok:
        if not silent_if_ok:
            print('Cycept functions correctly')
    else:
        print('Below we attempt compilation once more', file=sys.stderr)
        func = get_test_func()
        jit(func, silent=False)(n)
        raise RuntimeError('Cycept failure')  # if above recompilation somehow passes

