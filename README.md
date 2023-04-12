# Cycept
Effortless just-in-time compilation of Python functions,
powered by [Cython](https://cython.org/).


## Installation
Cycept is available on [PyPi](https://pypi.org/project/cycept/):
```bash
python -m pip install cycept
```

Cycept requires Python 3.9 or later.

To run Cycept a C compiler needs to be installed on the system.

* On **Linux** you may install [GCC](https://gcc.gnu.org/) (Debian-like distros: `sudo apt install build-essential`).
* On **macOS** you may install [Xcode](https://developer.apple.com/xcode/).
* On **Windows** you may install [MSVC](https://visualstudio.microsoft.com/visual-cpp-build-tools/).


## Quick demo
```python
"""Comparison of Python function JITs

Below we implement the sample function sum((a - b)**2) where a and b
are both 2D NumPy arrays. The following strategies are implemented and
compared against each other:
* Pure Python (baseline)
* NumPy
* Cycept JIT
* Cython JIT
* Numba JIT
"""

from time import time
import numpy as np

a = np.random.random((2_000, 3_000))
b = np.random.random((2_000, 3_000))

# Pure Python
def func(a, b):
    x = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            x += (a[i, j] - b[i, j])**2
    return x
tic = time()
result = func(a, b)
toc = time()
t_ref = toc - tic
print(f'Pure python: {result:<18} in {t_ref:.3e} s')

# NumPy
def func_numpy(a, b):
    return np.sum((a - b)**2)
tic = time()
result = func_numpy(a, b)
toc = time()
t = toc - tic
print(f'NumPy:       {result:<18} in {t:.3e} s ({int(t_ref/t)}x)')

# Cycept
import cycept
@cycept.jit
def func_cycept(a, b):
    x = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            x += (a[i, j] - b[i, j])**2
    return x
func_cycept(a[:1, :1], b[:1, :1])  # to compile
tic = time()
result = func_cycept(a, b)
toc = time()
t = toc - tic
print(f'Cycept:      {result:<18} in {t:.3e} s ({int(t_ref/t)}x)')

# Cython
import cython
@cython.compile
def func_cython(a, b):
    x = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            x += (a[i, j] - b[i, j])**2
    return x
func_cython(a[:1, :1], b[:1, :1])  # to compile
tic = time()
result = func_cython(a, b)
toc = time()
t = toc - tic
print(f'Cython:      {result:<18} in {t:.3e} s ({int(t_ref/t)}x)')

# Numba
import numba
@numba.jit
def func_numba(a, b):
    x = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            x += (a[i, j] - b[i, j])**2
    return x
func_numba(a[:1, :1], b[:1, :1])  # to compile
tic = time()
result = func_numba(a, b)
toc = time()
t = toc - tic
print(f'Numba:       {result:<18} in {t:.3e} s ({int(t_ref/t)}x)')
```

Running the above results in something similar to
```
Pure python: 1000265.9355757801 in 2.316e+00 s
NumPy:       1000265.9355757139 in 2.967e-02 s (78x)
Cycept:      1000265.9355757138 in 6.429e-03 s (360x)
Cython:      1000265.9355757801 in 7.103e-02 s (32x)
Numba:       1000265.9355757801 in 7.376e-03 s (314x)
```
For scientific codebases in the wild, code of the NumPy style is the
most widespread. However, writing out the loops while adding a JIT can
often lead to dramatic performance improvements, even when compared
to NumPy. A further benefit of this is a reduced memory footprint,
as no temporary arrays are created behind the scenes by the computation.

See the help info on `cycept.jit` for optional arguments:
```bash
python -c 'import cycept; help(cycept.jit)'
```


## Test suite
The code contains a unit test suite which may be run as
```
python -c "import cycept; cycept.test()"
```
This requires [pytest](https://docs.pytest.org/) to be installed
(`python -m pip install pytest`).


## What's up with the name?
'Cycept' is an amalgamation of '[Cython](https://cython.org/)' and
'[CO*N*CEPT](https://github.com/jmd-dk/concept)', the latter of which is a
cosmological simulation code that makes heavy use of code transformation,
both custom and through Cython. As the author of both projects, Cycept is my
attempt to extract some of the code transformation ideas buried within
CO*N*CEPT, making them available within an easy-to-use library.
Though no code is shared between the projects, in many respects Cycept
can be considered a spiritual descendant of CO*N*CEPT.

