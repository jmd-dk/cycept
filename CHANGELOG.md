## Cycept changelog

<br/>

### 🚀 [0.1.0](https://github.com/jmd-dk/cycept/releases/tag/v0.1.0) — 2023-??-??

#### ✨ Features
- `jit` decorator powered by [Cython](https://cython.org/).
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

