from cycept import jit

import cython
import numpy as np


def test_pass():
    @jit
    def f():
        pass
    assert f() is None

def test_object():
    @jit
    def f(a):
        return a
    a = object()
    assert f(a) is a

def test_type_arg():
    @jit
    def f(a):
        pass
    tps = [int, dict]
    for tp in tps:
        f(tp())
    for tp, key in zip(tps, f.__cycept__.keys()):
        assert key[0] is tp

def test_type_local():
    @jit
    def f(a):
        b = a
    tps = [float, str]
    for tp in tps:
        f(tp())
    for tp, call in zip(tps, f.__cycept__.values()):
        assert call.locals_types['b'] is tp

def test_type_return():
    @jit
    def f(a):
        return a
    tps = [bool, set]
    for tp in tps:
        f(tp())
    for tp, call in zip(tps, f.__cycept__.values()):
        assert call.locals_types['return'] is tp

def test_type_cython():
    @jit
    def f(a):
        b = [a]
        return 2*b.pop()
    f(True)
    types = next(iter(f.__cycept__.values())).types
    assert types == {'a': 'cython.bint', 'b': 'list', 'return': 'cython.Py_ssize_t'}

def test_type_array():
    @jit
    def f(a):
        b = a**0.5
        return b + 1j
    f(np.arange(3))
    call = next(iter(f.__cycept__.values()))
    assert call.locals_types['a'].dtype is np.int64
    assert call.locals_types['b'].dtype is np.float64
    assert call.locals_types['return'].dtype is np.complex128
    assert call.types == {'a': 'cython.longlong[::1]', 'b': 'cython.double[::1]', 'return': 'cython.doublecomplex[::1]'}

def test_type_annotation():
    @jit
    def f(a: int) -> float:
        b : object = [a]
        return 2*b.pop()
    f(True)
    types = next(iter(f.__cycept__.values())).types
    assert types == {'a': 'cython.Py_ssize_t', 'b': 'object', 'return': 'cython.double'}

def test_manual():
    def f(a):
        return 2*a
    g = jit(f)
    assert f(1) == g(1)
    assert next(iter(g.__cycept__.values())).types['return'] == 'cython.Py_ssize_t'

def test_lambda():
    def test(f):
        g = jit(f)
        assert f(1) == g(1)
        assert next(iter(g.__cycept__.values())).types['return'] == 'cython.Py_ssize_t'
    f = lambda a: 2*a
    test(f)
    test(lambda a: 2*a)

def test_closure():
    @jit
    def f(a):
        def g(a):
            return 2*a
        return 1 + g(a)
    for _ in range(2):
        assert f(3) == 7

def test_arg_default():
    @jit
    def f(a, b=3):
        return a + 2*b
    f(1)
    assert f(1) == 7
    assert f(1, 3) == 7
    assert f(1, 5) == 11
    assert f(1, b=5) == 11
    assert f(a=1, b=5) == 11
    assert f(b=5, a=1) == 11

def test_arg_posonly():
    @jit
    def f(a, /, b):
        return a + 2*b
    f(1, 3)
    assert f(1, 3) == 7
    assert f(1, b=3) == 7
    @jit
    def g(a, /, b=3):
        return a + 2*b
    g(1)
    assert g(1) == 7
    assert g(1, 5) == 11
    assert g(1, b=5) == 11

def test_arg_kwonly():
    @jit
    def f(a, *, b):
        return a + 2*b
    f(1, b=3)
    assert f(1, b=3) == 7
    assert f(a=1, b=3) == 7
    assert f(b=3, a=1) == 7
    @jit
    def g(a, *, b=3):
        return a + 2*b
    g(1)
    assert g(1) == 7
    assert g(1, b=5) == 11
    assert g(b=5, a=1) == 11

def test_arg_starargs():
    @jit
    def f(a, *args):
        return a + sum(args)
    f(1)
    assert next(iter(f.__cycept__.values())).locals_types['args'] is tuple
    assert f(1) == 1
    assert f(a=0) == 0
    assert f(1, 2, 3, 4) == 10
    @jit
    def g(*args, a=1):
        return a + sum(args)
    g()
    assert next(iter(g.__cycept__.values())).locals_types['args'] is tuple
    assert g() == 1
    assert g(2, 3, 4) == 10
    assert g(2, 3, 4, a=0) == 9

def test_arg_starstarkwargs():
    @jit
    def f(a, **kwargs):
        return a + kwargs.get('b', 0) + kwargs.get('c', 0)
    f(1)
    assert next(iter(f.__cycept__.values())).locals_types['kwargs'] is dict
    assert f(1) == 1
    assert f(a=0) == 0
    assert f(1, b=2, c=3) == 6
    @jit
    def g(a=1, **kwargs):
        return a + kwargs.get('b', 0) + kwargs.get('c', 0)
    g()
    assert next(iter(g.__cycept__.values())).locals_types['kwargs'] is dict
    assert g() == 1
    assert g(b=2, c=3) == 6
    assert g(b=2, c=3, a=0) == 5

def test_option_none():
    @jit()
    def f():
        pass
    assert f() is None

def test_option_compile():
    for compile in (False, True):
        @jit(compile=compile)
        def f():
            return cython.compiled
        f()  # to compile
        assert f() is compile

def test_option_silent(capfd):
    for silent in (False, True):
        @jit(silent=silent)
        def f():
            pass
        f()
        out, err = capfd.readouterr()
        if silent:
            assert out == ''
        else:
            assert 'Jitting f()' in out  # Cycept
            assert 'Compiling' in out    # Cython
            assert '-O3' in out          # C compiler

def test_option_html():
    for html in (False, True):
        @jit(html=html)
        def f():
            a = 1234
            pass
        f()
        html_annotation = f.__cycept__[()].compiled.html
        if html:
            assert '1234' in html_annotation
        else:
            assert html_annotation is None

def test_option_checks():
    for checks in (False, True):
        @jit(checks=checks)
        def f(a, i):
            return a[i]
        f(np.arange(3), 2)
        source = next(iter(f.__cycept__.values())).source
        deactivated_boundscheck = '@cython.boundscheck(False)' in source
        deactivated_initializedcheck = '@cython.initializedcheck(False)' in source
        assert deactivated_boundscheck is not checks
        assert deactivated_initializedcheck is not checks

def test_option_clike():
    for clike in (False, True):
        @jit(clike=clike)
        def f(a, b):
            return a//b
        for _ in range(2):  # twice in order to use compiled function
            result = f(-1, 2)
        source = next(iter(f.__cycept__.values())).source
        deactivated_wraparoundcheck = '@cython.wraparound(False)' in source
        deactivated_pydivisioncheck = '@cython.cdivision(True)' in source
        assert deactivated_wraparoundcheck is clike
        assert deactivated_pydivisioncheck is clike
        assert result == (0 if clike else -1)

def test_option_array_args():
    def g(a):
        try:
            a *= 2
        except TypeError:
            # Augmented assignment fails for memoryviews
            a[:] = 0
        return a
    for array_args in (False, True):
        @jit(array_args=array_args)
        def f(a):
            return g(a)
        for _ in range(2):  # twice in order to use compiled function
            result = f(np.arange(3))
        assert result.sum() == 6*array_args

def test_option_directives():
    for cdivision in (False, True):
        @jit(directives={'cdivision': cdivision})
        def f(a, b):
            return a//b
        for _ in range(2):  # twice in order to use compiled function
            result = f(-1, 2)
        assert result == (0 if cdivision else -1)

def test_option_optimizations(capfd):
    for level, optimizations in enumerate([False, 1, '-O2', {'-O3': True}]):
        @jit(optimizations=optimizations, silent=False)
        def f():
            pass
        f()
        out, err = capfd.readouterr()
        assert f'-O{level}' in out

