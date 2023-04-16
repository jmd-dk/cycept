import re
import sys

import cython
import numpy as np
import pytest

from cycept import jit, NdarrayTypeInfo


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
    call = next(iter(f.__cycept__.values()))
    assert call.types == {
        'a': 'cython.bint',
        'b': 'list',
        'return': 'cython.Py_ssize_t',
    }


def test_type_array():
    @jit
    def f(a):
        b = a**0.5
        return b + 1j
    f(np.arange(3, dtype=np.int64))
    key = next(iter(f.__cycept__))
    assert isinstance(key[0], NdarrayTypeInfo)
    call = f.__cycept__[key]
    assert call.locals_types['a'].dtype is np.int64
    assert call.locals_types['b'].dtype is np.float64
    assert call.locals_types['return'].dtype is np.complex128
    assert call.types == {
        'a': 'cython.longlong[::1]',
        'b': 'cython.double[::1]',
        'return': 'cython.doublecomplex[::1]',
    }


def test_type_annotation():
    @jit
    def f(a: int) -> float:
        b : object = [a]
        return 2*b.pop()
    f(True)
    call = next(iter(f.__cycept__.values()))
    assert call.types == {
        'a': 'cython.Py_ssize_t',
        'b': 'object',
        'return': 'cython.double',
    }


def test_manual():
    def f(a):
        return 2*a
    g = jit(f)
    assert f(1) == g(1)
    call = next(iter(g.__cycept__.values()))
    assert call.types['return'] == 'cython.Py_ssize_t'


def test_lambda():
    def test(f):
        g = jit(f)
        assert f(1) == g(1)
        call = next(iter(g.__cycept__.values()))
        assert call.types['return'] == 'cython.Py_ssize_t'
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


def test_method():
    class Adder:
        def __init__(self, a):
            self.a = a
        @jit
        def f(self, b):
            return self.a + b
    a, b = 2, 3
    adder = Adder(a)
    for _ in range(2):
        assert adder.f(b) == 5


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
    call = next(iter(f.__cycept__.values()))
    assert call.locals_types['args'] is tuple
    assert f(1) == 1
    assert f(a=0) == 0
    assert f(1, 2, 3, 4) == 10
    @jit
    def g(*args, a=1):
        return a + sum(args)
    g()
    call = next(iter(g.__cycept__.values()))
    assert call.locals_types['args'] is tuple
    assert g() == 1
    assert g(2, 3, 4) == 10
    assert g(2, 3, 4, a=0) == 9


def test_arg_starstarkwargs():
    @jit
    def f(a, **kwargs):
        return a + kwargs.get('b', 0) + kwargs.get('c', 0)
    f(1)
    call = next(iter(f.__cycept__.values()))
    assert call.locals_types['kwargs'] is dict
    assert f(1) == 1
    assert f(a=0) == 0
    assert f(1, b=2, c=3) == 6
    @jit
    def g(a=1, **kwargs):
        return a + kwargs.get('b', 0) + kwargs.get('c', 0)
    g()
    call = next(iter(g.__cycept__.values()))
    assert call.locals_types['kwargs'] is dict
    assert g() == 1
    assert g(b=2, c=3) == 6
    assert g(b=2, c=3, a=0) == 5


def test_option_none():
    @jit()
    def f():
        pass
    assert f() is None


def test_option_compile():
    for compile in [False, True]:
        @jit(compile=compile)
        def f():
            return cython.compiled
        f()  # to compile
        assert f() is compile


def test_option_silent(capfd):
    for silent in [False, True]:
        @jit(silent=silent)
        def f(a):
            pass
        f(np.linspace(0, 1, 3, dtype=np.float64))
        captured = capfd.readouterr()
        if silent:
            assert captured.out == ''
        else:
            assert 'Jitting f(a: double[::1])' in captured.out       # Cycept
            assert 'Compilation time' in captured.out                # Cycept
            assert 'Compiling' in captured.out                       # Cython
            assert re.search(r'cycept_module_\d+\.o', captured.out)  # C compiler


def test_option_c_lang():
    for c_lang in [None, 'c', 'c++']:
        @jit(c_lang=c_lang)
        def f():
            pass
        f()
        call = next(iter(f.__cycept__.values()))
        assert re.match('c' + '.'*(c_lang == 'c++'), call.compiled.source_ext)
        if c_lang == 'c++':
            assert len(re.findall('template', call.compiled.source))


def test_option_html():
    for html in [False, True]:
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
    for checks in [False, True]:
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
    for clike in [False, True]:
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
    for array_args in [False, True]:
        @jit(array_args=array_args)
        def f(a):
            return g(a)
        for _ in range(2):  # twice in order to use compiled function
            result = f(np.arange(3))
        assert result.sum() == 6*array_args


def test_option_directives():
    for cdivision in [False, True]:
        @jit(directives={'cdivision': cdivision})
        def f(a, b):
            return a//b
        for _ in range(2):  # twice in order to use compiled function
            result = f(-1, 2)
        assert result == (0 if cdivision else -1)


def test_option_optimizations(capfd):
    for fastmath in [False, True]:
        @jit(optimizations={'fastmath': fastmath}, silent=False)
        def f():
            pass
        f()
        captured = capfd.readouterr()
        output = captured.out + captured.err
        if fastmath:
            assert '-ffast-math' in output or '/fp:fast' in output
        else:
            assert '-ffast-math' not in output and '/fp:fast' not in output


def test_option_integral():
    for integral in [None, object, 'int', np.int8, np.int32, np.int64]:
        @jit(integral=integral)
        def f(a):
            return a + 1
        f(0)
        if integral in {None, 'int'}:
            for size in [8, 16, 32, 64, 128]:
                a = 2**(size - 1) - 1
                result = f(a)
                assert result in {-2**(size - 1), a + 1}
                if result < 0:
                    break
            assert result < 0
        elif integral is object:
            size = 1024
            a = 2**(size - 1) - 1
            assert f(a) == a + 1
        else:
            size = 8*integral().itemsize
            assert f(2**(size - 1) - 1) == -2**(size - 1)


def test_option_floating():
    for floating in [None, 'double', np.float32, np.float64]:
        @jit(floating=floating)
        def f(a):
            return 2*a
        f(0.0)
        dtype = floating
        if dtype in {None, 'double'}:
            dtype = np.float64
        a = float(np.finfo(dtype).max)
        assert f(a) == float('inf')
        assert f(a/2) == a


def test_option_floating_complex():
    for floating_complex in [None, np.complex64, np.complex128]:
        @jit(floating_complex=floating_complex)
        def f(a):
            return 2*a
        f(0.0j)
        dtype = floating_complex
        if dtype is None:
            dtype = np.complex128
        a = float(np.finfo(dtype).max)
        a -= a*1j
        result = f(a)
        assert result.real == +float('inf')
        assert result.imag == -float('inf')
        assert f(a/2) == a


def test_array_mutability():
    @jit
    def f(a):
        a[0] += 1
    a = np.zeros(3, dtype=int)
    n = 3
    for _ in range(n):
        f(a)
    assert a[0] == n


def test_array_operations():
    @jit
    def f(a):
        b = a + 1
        return b
    a = np.zeros(3, dtype=int)
    for _ in range(2):
        b = f(a)
        assert b.sum() == 3


def test_array_augmentation():
    @jit
    def f(a):
        a += 1
    a = np.zeros(3, dtype=int)
    for _ in range(2):
        f(a)
    assert a.sum() == 6


def test_array_slice_augmentation():
    @jit
    def f(a):
        a[1:] += 1
    a = np.zeros(3, dtype=int)
    for _ in range(2):
        f(a)
    assert a.sum() == 4


def test_array_element_augmentation():
    @jit
    def f(a, b):
        a[b] += 1
    a = np.zeros(3, dtype=int)
    b = np.zeros(3, dtype=bool)
    b[2] = True
    for _ in range(2):
        f(a, 1)  # should use memoryview for a
        f(a, b)  # should use array for a
    assert (a == [0, 2, 2]).all()
    sourcelengths = [
        len(call.compiled.source)
        for call in f.__cycept__.values()
    ]
    assert sourcelengths[0] < sourcelengths[1]


def test_array_comparison():
    @jit
    def f(a):
        b = a < 2
        return a >= b[0]
    a = np.arange(3, dtype=int)
    for _ in range(2):
        assert f(a).sum() == 2

