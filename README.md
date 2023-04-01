# Cycept
Effortless just-in-time compilation of Python functions,
powered by [Cython](https://cython.org/).


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
print(f'Pure python: {result} in {t_ref:.3e} s')

# NumPy
def func_numpy(a, b):
    return np.sum((a - b)**2)
tic = time()
result = func_numpy(a, b)
toc = time()
t = toc - tic
print(f'NumPy:       {result} in {t:.3e} s ({int(t_ref/t)}x)')

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
print(f'Cycept:      {result} in {t:.3e} s ({int(t_ref/t)}x)')

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
print(f'Cython:      {result} in {t:.3e} s ({int(t_ref/t)}x)')

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
print(f'Numba:       {result} in {t:.3e} s ({int(t_ref/t)}x)')
```

Running the above results in something similar to
```
Pure python: 1000174.1891294761 in 2.305e+00 s
NumPy:       1000174.1891295158 in 2.956e-02 s (77x)
Cycept:      1000174.1891294761 in 6.517e-03 s (353x)
Cython:      1000174.1891294761 in 7.060e-02 s (32x)
Numba:       1000174.1891294761 in 7.414e-03 s (310x)
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

