## Cycept changelog

<br/>


### 🚀 [0.0.4](https://github.com/jmd-dk/cycept/releases/tag/v0.0.4) — *Still under development*

#### ✨ Features
- Added `cycept.check()` for quickly checking
  whether Cycept functions correctly.

[Commits since 0.0.3](https://github.com/jmd-dk/cycept/compare/v0.0.3...v0.0.4)

---
<br/><br/>


### 🚀 [0.0.3](https://github.com/jmd-dk/cycept/releases/tag/v0.0.3) — 2023-04-16

#### ✨ Features
- It is now possible to transpile to C++ rather than C
  (the `c_lang` option of `@jit`).

#### ⚡ Optimizations
- Improved system for compiler optimizations.
- Now generates optimal code for chained array indexing.

#### 🔧 Portability
- Now uses `setuptools` and `Cython.Distutils` instead of `distutils`.
- Improved packaging.

#### 🤖 Tests
- Added performance test suite.
- Added more unit tests.

[Commits since 0.0.2](https://github.com/jmd-dk/cycept/compare/v0.0.2...v0.0.3)

---
<br/><br/>


### 🚀 [0.0.2](https://github.com/jmd-dk/cycept/releases/tag/v0.0.2) — 2023-04-12

#### ✨ Features
- More array/memoryview operations allowed:
  - Comparisons (arrays).
  - Scalar indexing (memoryviews).
- New `@jit` options `integral`, `floating` and `floating_complex` for
  overwriting the default Cython types used for the Python types `int`,
  `float` and `complex`.
- Compilation times are now measured and stored as `FunctionCall.time`.

#### 🔧 Portability
- Support for Windows.
- Better support for macOS.
- Better support for Python 3.9.
- Available NumPy types now checked at runtime.
- Support using `tomli` and `toml` if `tomllib` is missing (Python < 3.11).

#### 🐛 Bugs fixed
- NumPy Boolean arrays now correctly translated to Cython memoryviews.

#### 👌 Other changes
- Compiler error messages are now also suppressed when running `silent`ly.

[Commits since 0.0.1](https://github.com/jmd-dk/cycept/compare/v0.0.1...v0.0.2)

---
<br/><br/>


### 🚀 [0.0.1](https://github.com/jmd-dk/cycept/releases/tag/v0.0.1) — 2023-04-11

#### ✨ Features
- `@jit` decorator powered by [Cython](https://cython.org/).
- Automatic type inference
  - Python types (`str`, `list`, `dict`, ...)
  - C/Cython types (`cython.Py_ssize_t`, `cython.double`, ...).
- Manual type annotations using [PEP 484](https://peps.python.org/pep-0484/)
  and [PEP 526](https://peps.python.org/pep-0526/).
- NumPy arrays treated as memoryviews when possible, while still supporting
  NumPy array operations.
- Source code extraction possible for `lambda` functions.
- Jitting of closures supported.
- Easy viewing of generated code in the browser via `func.__cycept__()`.
- Available options to `jit`:
  - `compile`: Used to disable the jitting.
  - `silent`: Used to display compilation text.
  - `checks`: Used to enable certain runtime checks.
  - `clike`: Used to change certain Python semantics to C semantics.
  - `array_args`: Used to allow using memoryviews as function arguments.
  - `directives`: Used to specify Cython directives.
  - `optimizations`: Used to set C compiler optimizations.
- Test suite.

