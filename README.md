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

* On **Linux** you may install [GCC](https://gcc.gnu.org/)
  (Debian-like distros: `sudo apt install build-essential`).
* On **macOS** you may install [Clang](https://clang.llvm.org/)
  (available through [Xcode](https://developer.apple.com/xcode/)).
* On **Windows** you may install
  [MSVC](https://en.wikipedia.org/wiki/Microsoft_Visual_C%2B%2B)
  (available through
  [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)).

If you are using [Anaconda](https://www.anaconda.com/) on Linux or macOS,
you may also obtain a C compiler through
`conda install -c conda-forge c-compiler`.

Once installed you can check whether Cycept functions correctly using
```bash
python -c "import cycept; cycept.check()"
```
If it does not work due to missing `Python.h` and you are running Linux,
make sure to install the Python development headers (Debian-like distros:
`sudo apt install python3-dev` if you are using the system Python).


## Demo
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

from time import perf_counter
import numpy as np

m, n = 2_000, 3_000
a = np.random.random((m, n))
b = np.random.random((m, n))

# Pure Python
def func(a, b):
    x = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            x += (a[i, j] - b[i, j])**2
    return x
tic = perf_counter()
result = func(a, b)
toc = perf_counter()
t_ref = toc - tic
print(f'Python: {result:<18} in {t_ref:.3e} s')

# NumPy
def func_numpy(a, b):
    return ((a - b)**2).sum()
tic = perf_counter()
result = func_numpy(a, b)
toc = perf_counter()
t = toc - tic
print(f'NumPy:  {result:<18} in {t:.3e} s ({int(t_ref/t)}x)')

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
tic = perf_counter()
result = func_cycept(a, b)
toc = perf_counter()
t = toc - tic
print(f'Cycept: {result:<18} in {t:.3e} s ({int(t_ref/t)}x)')

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
tic = perf_counter()
result = func_cython(a, b)
toc = perf_counter()
t = toc - tic
print(f'Cython: {result:<18} in {t:.3e} s ({int(t_ref/t)}x)')

# Numba
import numba
@numba.njit
def func_numba(a, b):
    x = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            x += (a[i, j] - b[i, j])**2
    return x
func_numba(a[:1, :1], b[:1, :1])  # to compile
tic = perf_counter()
result = func_numba(a, b)
toc = perf_counter()
t = toc - tic
print(f'Numba:  {result:<18} in {t:.3e} s ({int(t_ref/t)}x)')
```

Running the above results in something similar to
```
Python: 1000265.9355757801 in 2.316e+00 s
NumPy:  1000265.9355757139 in 2.967e-02 s (78x)
Cycept: 1000265.9355757138 in 6.429e-03 s (360x)
Cython: 1000265.9355757801 in 7.103e-02 s (32x)
Numba:  1000265.9355757801 in 7.376e-03 s (314x)
```
For scientific codebases in the wild, code of the NumPy style is the
most widespread. However, writing out the loops while adding a JIT can
often lead to dramatic performance improvements, even when compared
to NumPy. A further benefit of this is a reduced memory footprint,
as no temporary arrays are created behind the scenes by the computation.

While Cycept especially shines on "array code",
speedups are generally achievable on all kinds of code.


## Performance benchmarks
Cycept comes with a suite of automated performance tests, all similar in
nature to the demo above. To run this, do
```bash
python -c "import cycept; cycept.bench(show_func=True)"
```
Currently, the competitors included are

* Pure Python
* NumPy
* Cycept
* Cython
* Numba

Note that you do not need to have *all* of the above installed in order to
run the benchmarks.


## How does it work?
When a function is to be jitted by Cycept, the first step is to infer the
types of all variables. This is done by simply running the function once in
pure Python and applying introspection. For this reason, it it best not to
ask for too heavy a computation for the initial call. You might want to
manually set the type of some variables, which is possible via Python
[type hints](https://docs.python.org/3/library/typing.html).

Special treatment is given to NumPy arrays. These are converted into Cython
memoryviews to enable fast indexing, but converted back whenever NumPy array
operations need to be performed.

With the types and array modifications in place, the modified source code is
handed to Cython, which transpiles it into C/C++. The C compiler then compiles
this into an extension module, from which we finally get the jitted function.
By default, aggressive optimizations are applied at all stages.

The jitted version of the function is cached, so that the whole process does
not have to be repeated for every call to the function. If however the
function is called with new types, another jitted version of the function will
be made, specialized to those types.


## Optional arguments to `@cycept.jit()`
The transpilation carried out by Cycept can be tuned by various optional
arguments to `@cycept.jit()`. These are all documented in the docstring,
viewable through
```bash
python -c "import cycept; help(cycept.jit)"
```
Of particular interest is `html`, which allows for viewing of the transpiled
Cython and C code by calling the `__cycept__()` method on the jitted function,
e.g.
```python
import cycept

@cycept.jit(html=True)
def twice(x):
    return 2*x

twice(42)           # int call
twice('hello')      # str call
twice.__cycept__()  # view all transpilations
```


## Unit tests
The code contains a unit test suite which may be run as
```bash
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
can be considered a spiritual successor to CO*N*CEPT.
Furthermore, 'Cy*cept*' has a nice in*cept*ion ring to it,
which seems fitting for a piece of code that generates code.

