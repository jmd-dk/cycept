import ast
import collections
import contextlib
import inspect
import itertools
import pathlib
import re
import subprocess
import sys
import tempfile
import time
import typing
import warnings
import webbrowser


# Class for storing and dynamically handling
# various aspects of the function call to be jitted.
class FunctionCall:

    class Compiled(typing.NamedTuple):
        func: object
        module: object
        source: str
        source_ext: str
        html: str

    class Time(typing.NamedTuple):
        compile: float
        total: float

    def __init__(self, func, wrapper, args, kwargs):
        # Function and call arguments
        self.func = func
        self.wrapper = wrapper
        self.args = args
        self.kwargs = kwargs
        # Properties to be lazily constructed
        self._func_name = None
        self._signature = None
        self._arguments = None
        self._arguments_types = None
        self._hash = None
        self._source = None
        self._ast = None
        self._annotations = None
        self._types = None
        self._module = None
        self._module_dict = None
        self._globals = None
        self._nonlocals = None
        # Record of inferred types of local variables
        self.locals_types = {}
        # Local names which should be disregarded when
        # creating typed local variables.
        self.locals_excludes = {'return'}
        # Additional source lines to be included in the Cython extension module
        self.cython_module_lines = []
        # Compilation products
        self.compiled = None
        # Time measurements
        self.time = None

    # Name of function (with lambda functions renamed)
    @property
    def func_name(self):
        if self._func_name is not None:
            return self._func_name
        self._func_name = self.func.__name__
        if self._func_name == '<lambda>':
            self._func_name = f'_cycept_lambda_{self.hash}'
        return self._func_name

    # Function signature (no type information)
    @property
    def signature(self):
        if self._signature is not None:
            return self._signature
        self._signature = inspect.signature(self.func)
        return self._signature

    # Argument values (dict)
    @property
    def arguments(self):
        if self._arguments is not None:
            return self._arguments
        bound = self.signature.bind(*self.args, **self.kwargs)
        bound.apply_defaults()
        self._arguments = bound.arguments
        return self._arguments

    # Argument types (tuple)
    @property
    def arguments_types(self):
        if self._arguments_types is not None:
            return self._arguments_types
        from .core import get_type
        self._arguments_types = tuple(
            get_type(val) for val in self.arguments.values()
        )
        return self._arguments_types

    # Hash of the function call
    @property
    def hash(self):
        if self._hash is not None:
            return self._hash
        self._hash = abs(hash((self.func, self.wrapper, self.arguments_types)))
        return self._hash

    # Source code of the function
    @property
    def source(self):
        if self._source is not None:
            return self._source
        self._source = self.get_source()
        # Separate annotations from source
        self._annotations, self._source = self.extract_annotations()
        # Invalidate AST
        self._ast = None
        return self._source

    @property
    def ast(self):
        if self._ast is not None:
            return self._ast
        self._ast = ast.parse(self.source)
        return self._ast

    # User-defined annotations within the function
    @property
    def annotations(self):
        if self._annotations is not None:
            return self._annotations
        self._annotations, _ = self.extract_annotations()
        return self._annotations

    # Types to be used during compilation
    @property
    def types(self):
        if self._types is not None:
            return self._types
        from .core import convert_to_cython_type
        # Transform recorded types of locals to str
        # representations of corresponding Cython types.
        locals_types = {
            name: convert_to_cython_type(tp)
            for name, tp in self.locals_types.items()
        }
        # Transform annotations to str representations
        # of Cython types.  Keep unrecognized types as is.
        annotations = {
            name: convert_to_cython_type(tp, tp)
            for name, tp in self.annotations.items()
        }
        # Merge recorded types and annotations.
        # On conflict we keep the annotation.
        self._types = locals_types | annotations
        return self._types

    # Module within which function was defined
    @property
    def module(self):
        if self._module is not None:
            return self._module
        self._module = inspect.getmodule(self.func)
        return self._module

    # Module dictionary associated with function module
    @property
    def module_dict(self):
        if self._module_dict is not None:
            return self._module_dict
        self._module_dict = getattr(self.module, '__dict__', None)
        if self._module_dict is None:
            raise ModuleNotFoundError(
                f'Failed to get __dict__ of module {self.module} '
                f'within which function {self.func} is defined'
            )
        return self._module_dict

    # Global names referenced within the function
    @property
    def globals(self):
        if self._globals is not None:
            return self._globals
        self._globals = inspect.getclosurevars(self.func).globals
        return self._globals

    # Non-local names referenced within the function
    @property
    def nonlocals(self):
        if self._nonlocals is not None:
            return self._nonlocals
        self._nonlocals = inspect.getclosurevars(self.func).nonlocals
        return self._nonlocals

    # Method for obtaining the function source
    def get_source(self):
        try:
            source = inspect.getsource(self.func)
        except Exception:
            # inspect.getsource() fails in the interactive REPL.
            # The dill package can hack around this limitation.
            try:
                import dill
            except ModuleNotFoundError:
                warnings.warn(
                    (
                        f'To use cycept.jit interactively, '
                        f'please install the dill Python package:\n'
                        f'    {sys.executable} -m pip install dill\n'
                    ),
                    RuntimeWarning,
                )
                raise
            source = dill.source.getsource(self.func)
        # Dedent source lines
        lines = source.split('\n')
        indentation = ' ' * (len(lines[0]) - len(lines[0].lstrip()))
        if indentation:
            for i, line in enumerate(lines):
                lines[i] = line.removeprefix(indentation)
            source = '\n'.join(lines)
        # Convert lambda function to def function
        if self.func.__name__ == '<lambda>':
            # Extract just the lambda expression
            # (both inspect.getsource() and dill.source.getsource()
            # returns the surrounding code as well in the case of unassigned
            # lambda expressions).
            class LambdaVisitor(ast.NodeVisitor):
                def __init__(self, root):
                    self.root = root
                    self._lambda = None
                def get_lambda(self):
                    self._lambda = None
                    self.visit(self.root)
                    return self._lambda
                def visit_Lambda(self, node):
                    if self._lambda is None:
                        self._lambda = node
            source = ast.unparse(LambdaVisitor(ast.parse(source)).get_lambda())
            # Transform to def function
            indentation = ' ' * 4
            source = re.subn(
                r'.*(^|\W)lambda (.+?): ?',
                rf'def {self.func_name}(\g<2>):\n{indentation}return ',
                source,
                1,
            )[0]
        else:
            # Do AST round-trip, ensuring canonical code style
            # (e.g. 4 space indentation).
            source = ast.unparse(ast.parse(source))
        # Remove jit decorator and decorators above it
        from .core import jit
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if match := re.search(r'^@(.+?)(\(|$)', line):
                deco_name = match[1].strip()
                try:
                    deco_func = eval(deco_name, self.module_dict)
                except Exception:
                    continue
                if deco_func is jit:
                    lines = lines[i+1:]
                    break
            elif line.startswith('def '):
                # Arrived at definition line without finding jit decorator.
                # This happens if the "decoration" is done as
                # func = jit(func) rather than using the @jit syntax.
                break
        source = '\n'.join(lines)
        return source

    # Method for extracting user-defined type annotations.
    # The return value is the annotations as well as
    # the modified source without the annotations.
    def extract_annotations(self):
        class AnnotationExtractor(ast.NodeTransformer):
            def __init__(self, call):
                self.call = call
                self.annotations = {}
            def extract(self):
                self.annotations.clear()
                source = ast.unparse(self.visit(self.call.ast))
                return self.annotations, source
            def add_annotation(self, name, tp):
                if not isinstance(name, str):
                    name = ast.unparse(name)
                tp = eval(ast.unparse(tp), self.call.module_dict)
                tp_old = self.annotations.setdefault(name, tp)
                if tp_old == tp:
                    return
                warnings.warn(
                    (
                        f'Name {name!r} annotated with different types '
                        f'in jitted {self.call}: {tp}, {tp_old}. '
                        f'The first will be used'
                    ),
                    RuntimeWarning,
                )
            def visit_arg(self, node):
                if not node.annotation:
                    return node
                self.add_annotation(node.arg, node.annotation)
                return type(node)(**(vars(node) | {'annotation': None}))
            def visit_FunctionDef(self, node):
                node = self.generic_visit(node)
                if not node.returns:
                    return node
                self.add_annotation('return', node.returns)
                return type(node)(**(vars(node) | {'returns': None}))
            def visit_AnnAssign(self, node):
                if node.simple:
                    self.add_annotation(node.target, node.annotation)
                else:
                    warnings.warn(
                        (
                            f'Ignoring complex type annotation '
                            f'{ast.unparse(node)!r} in jitted {self.call}'
                        ),
                        RuntimeWarning,
                    )
                if not node.value:
                    # Annotation-only statement. Remove completely.
                    return None
                # Transform to assignment statement without annotation
                return ast.Assign(**(
                    vars(node) | {'targets': [node.target], 'value': node.value}
                ))
        annotations, source = AnnotationExtractor(self).extract()
        return annotations, source

    # Method for transforming all Unicode names to ASCII names
    def asciify_names(self):
        from .core import asciify
        class UnicodeRenamer(ast.NodeTransformer):
            def __init__(self, call):
                self.call = call
            def rename(self):
                source = ast.unparse(self.visit(self.call.ast))
                return source
            def visit_Name(self, node):
                name = asciify(node.id)
                return type(node)(**(vars(node) | {'id': name}))
            def visit_ClassDef(self, node):
                node = self.generic_visit(node)
                name = asciify(node.name)
                return type(node)(**(vars(node) | {'name': name}))
            def visit_FunctionDef(self, node):
                node = self.generic_visit(node)
                name = asciify(node.name)
                return type(node)(**(vars(node) | {'name': name}))
            def visit_arg(self, node):
                node = self.generic_visit(node)
                name = asciify(node.arg)
                return type(node)(**(vars(node) | {'arg': name}))
            def visit_keyword(self, node):
                node = self.generic_visit(node)
                name = asciify(node.arg)
                return type(node)(**(vars(node) | {'arg': name}))
            def visit_Attribute(self, node):
                node = self.generic_visit(node)
                name = asciify(node.attr)
                return type(node)(**(vars(node) | {'attr': name}))
        self._source = UnicodeRenamer(self).rename()
        # Note that name changes caused by the above ASCIIfication
        # makes the AST, the function signature and many other attributes
        # invalid, yet we do not flag any attributes as such.

    # Method for wrapping NumPy array variables in numpy.asarray()
    # whenever operations are used that do not work with Cython memoryviews.
    def protect_arrays(self, array_args):
        from .core import NdarrayTypeInfo
        # Find local NumPy arrays
        names = {
            name
            for name, tp in self.locals_types.items()
            if isinstance(tp, NdarrayTypeInfo)
        }
        names.discard('return')
        if not names:
            # No local arrays found
            return
        # Wrap local arrays
        class ArrayWrapper(ast.NodeTransformer):
            class ArrayAttribute(typing.NamedTuple):
                name: str
                is_scalar: bool = False
                is_sequence: bool = False
            array_attributes = {
                'shape': ArrayAttribute('shape', is_sequence=True),
                'strides': ArrayAttribute('strides', is_sequence=True),
                'ndim': ArrayAttribute('ndim', is_scalar=True),
                'size': ArrayAttribute('size', is_scalar=True),
                'itemsize': ArrayAttribute('itemsize', is_scalar=True),
                'nbytes': ArrayAttribute('nbytes', is_scalar=True),
            }
            def __init__(self, call, names, array_args, wrapper_func, tmp_name):
                self.call = call
                self.names = names
                self.array_args = array_args
                self.wrapper_func = wrapper_func
                self.tmp_name = tmp_name
                self.sequence_attrs = set()
                self.wrapped_any = False
                self.tmp_any = False
            def wrap(self):
                self.sequence_attrs.clear()
                self.wrapped_any = False
                self.tmp_any = False
                return self.visit(self.call.ast)
            def wrap_node(self, node, kind=0):
                nodes_subscript = []
                while isinstance(node, ast.Subscript):
                    nodes_subscript.append(node)
                    node = node.value
                if not isinstance(node, ast.Name) or node.id not in self.names:
                    return
                if nodes_subscript:
                    # Memoryview/array indexing arr[x0][x1, x2][...].
                    # If we can prove that all x are scalars
                    # we use the memoryview as is.
                    for node_subscript in nodes_subscript:
                        if isinstance(node_subscript.slice, ast.Tuple):
                            if not all(self.is_scalar(el) for el in node_subscript.slice.elts):
                                break
                        elif not self.is_scalar(node_subscript.slice):
                            break
                    else:
                        return
                # Wrap node
                self.wrapped_any = True
                if kind == 0:
                    node.id = f'{self.wrapper_func}({node.id})'
                elif kind == 1:
                    self.tmp_any = True
                    node.id = f'{self.tmp_name} = {self.wrapper_func}({node.id}); {self.tmp_name}'
            def is_scalar(self, node):
                if isinstance(node, ast.Name):
                    tp = self.call.types[node.id]
                    return tp.startswith('cython.') and '[' not in tp
                elif isinstance(node, ast.Constant):
                    return isinstance(node.value, int)
                elif isinstance(node, ast.UnaryOp):
                    return self.is_scalar(node.operand)
                elif isinstance(node, ast.BinOp):
                    return self.is_scalar(node.left) and self.is_scalar(node.right)
                elif isinstance(node, ast.Subscript):
                    if isinstance(node.value, ast.Name):
                        return node.value.id in self.names and self.is_scalar(node.slice)
                    elif isinstance(node.value, ast.Attribute):
                        return (
                            node.value.value.id in self.names
                            and (array_attribute := self.array_attributes.get(node.value.attr)) is not None
                            and array_attribute.is_sequence
                            and self.is_scalar(node.slice)
                        )
                elif isinstance(node, ast.Attribute):
                    return (
                        isinstance(node.value, ast.Name)
                        and node.value.id in self.names
                        and (array_attribute := self.array_attributes.get(node.attr)) is not None
                        and array_attribute.is_scalar
                    )
                return False
            def visit_UnaryOp(self, node):
                node = self.generic_visit(node)
                self.wrap_node(node.operand)
                return node
            def visit_BinOp(self, node):
                node = self.generic_visit(node)
                self.wrap_node(node.left)
                self.wrap_node(node.right)
                return node
            def visit_Compare(self, node):
                node = self.generic_visit(node)
                self.wrap_node(node.left)
                for comparator in node.comparators:
                    self.wrap_node(comparator)
                return node
            def visit_AugAssign(self, node):
                node = self.generic_visit(node)
                self.wrap_node(node.target, kind=1)
                return node
            # Use NumPy arrays as arguments within function calls
            # if array_args is True.
            def visit_Call(self, node):
                node = self.generic_visit(node)
                if not self.array_args:
                    return node
                for arg in node.args:
                    self.wrap_node(arg)
                return node
            # Cython memoryviews and NumPy arrays share a subset of
            # their attributes, as specified by array_attributes.
            # Below, the visit_Attribute() method will check if an attribute
            # lookup is a recognized memoryview attribute, in which case the
            # lookup will not be wrapped. Otherwise it will. However, for the
            # sequence attributes, e.g. 'shape', the value of the Cython
            # memoryview attribute and the NumPy array attribute are not
            # equivalent (they have different lengths in general). What is
            # equivalent is the result of subsequent indexing into these
            # sequences, e.g. .shape[i], with i an integer (scalar).
            # The visit_Subscript() method below will find such subscript
            # nodes and flag them, so that the visit_Attribute() method will
            # know not to wrap these. Sequence attribute lookups that are not
            # immediately indexed will not be flagged
            # and will thus be wrapped.
            def visit_Subscript(self, node):
                value = node.value
                if (
                    isinstance(value, ast.Attribute)
                    and isinstance(value.value, ast.Name)
                    and value.value.id in self.names
                    and (array_attribute := self.array_attributes.get(value.attr)) is not None
                    and array_attribute.is_sequence
                    and self.is_scalar(node.slice)
                ):
                    # Record usage of memoryview.shape[i]
                    self.sequence_attrs.add(value)
                node = self.generic_visit(node)
                return node
            def visit_Attribute(self, node):
                node = self.generic_visit(node)
                if (array_attribute := self.array_attributes.get(node.attr)) is None:
                    # Not a memoryview attribute lookup
                    self.wrap_node(node.value)
                elif array_attribute.is_sequence and node not in self.sequence_attrs:
                    # Memoryview attribute lookup of sequence
                    # that is not immediately indexed by a scalar.
                    self.wrap_node(node.value)
                return node
        wrapper_func = '_cycept_asarray_'
        tmp_name = '_cycept_tmp_'
        array_wrapper = ArrayWrapper(self, names, array_args, wrapper_func, tmp_name)
        self._source = ast.unparse(array_wrapper.wrap())
        self._ast = None  # invalidate AST (cannot use the above as expressions are used as names)
        if not array_wrapper.wrapped_any:
            return
        if array_wrapper.tmp_any:
            from .core import record_types
            record_types(self, {tmp_name: None})
        # Add wrapper_func as local variable
        class NodeAdder(ast.NodeTransformer):
            def __init__(self, call, node_to_add):
                self.call = call
                self.node_to_add = node_to_add
            def add_node(self):
                return self.visit(self.call.ast)
            def visit_FunctionDef(self, node):
                if self.node_to_add is None:
                    return node
                node = ast.FunctionDef(**(vars(node) | {'body': [self.node_to_add] + node.body}))
                self.node_to_add = None
                return node
        self._source = ast.unparse(NodeAdder(self, ast.parse(f'{wrapper_func} = _{wrapper_func}_')).add_node())
        self._ast = None  # invalidate AST (above node addition is not strictly legal)
        # We need to assign _{wrapper_func}_ to numpy.asarray() in the extension module
        self.cython_module_lines += [
            '\n# For converting memoryviews back to NumPy arrays',
            f'from numpy import asarray as _{wrapper_func}_'
        ]

    # Method for updating the source from being a Python function
    # to being a Cython extension module.
    def to_cython(self, directives):
        from .core import asciify
        preamble = [
            '#',  # first comment line can have C code attached to it
            f'# Cycept version of {self} with type signature',
            f'# {self!r}',
        ]
        preamble += self.cython_module_lines
        excludes = (self.func_name, 'cython')
        fake_globals = [
            f'{asciify(name)} = object()'
            for name in collections.ChainMap(self.nonlocals, self.globals)
            if name not in excludes
        ]
        if fake_globals:
            preamble += [
                '\n# Declare fake globals',
                *fake_globals,
            ]
        preamble += [
            '\n# Import Cython',
            'import cython',
            'cimport cython',
        ]
        header_top = ['\n# Function to be jitted']
        if self.ccall_allowed():
            header_top.append('@cython.ccall')
        for directive, val in directives.items():
            header_top.append(f'@cython.{directive}({val!r})')
        declaration_locals = ', '.join(
            f'{name}={tp}'
            for name, tp in self.types.items()
            if name not in self.locals_excludes
        )
        declaration_return = self.types.get('return', 'object')
        header_bot = []
        if declaration_locals:
            header_bot.append(f'@cython.locals({declaration_locals})')
        header_bot.append(f'@cython.returns({declaration_return})')
        # ASCIIfy Unicode variable names in the source and bottom header
        self._source = '\n'.join(
            itertools.chain(
                header_bot,
                self.source.split('\n'),
            )
        )
        self._ast = None
        self.asciify_names()
        # Combine with preamble and top header
        self._source = '\n'.join(
            itertools.chain(
                preamble,
                header_top,
                self.source.split('\n'),
            )
        )
        # ASCIIfy the stored function name itself
        self._func_name = asciify(self.func_name)

    # Method for determining whether the function to be Cythonized
    # may be done so using @cython.ccall (cpdef).
    def ccall_allowed(self):
        # *args and **kwargs not allowed with ccall
        for param in self.signature.parameters.values():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                return False
        # Closures not allowed with ccall
        class ClosureVisitor(ast.NodeVisitor):
            def __init__(self, call):
                self.call = call
                self._contains_closure = False
            def contains_closure(self):
                self._contains_closure = False
                self.visit(self.call.ast)
                return self._contains_closure
            def visit_FunctionDef(self, node):
                self.generic_visit(node)
                if node.name != self.call.func_name:
                    self._contains_closure = True
            def visit_Lambda(self, node):
                self._contains_closure = True
        if ClosureVisitor(self).contains_closure():
            return False
        # No violations found
        return True

    # Method for handling Cythonization and C compilation
    def compile(self, optimizations, c_lang, html, silent):
        from .core import asciify
        # Cythonize and compile extension module within temporary directory.
        # The compiled module is then imported (through hacking of sys.path)
        # before it is removed from disk.
        @contextlib.contextmanager
        def hack_sys_path():
            sys.path.append(dir_name)
            try:
                yield
            except Exception:
                raise
            finally:
                sys.path.remove(dir_name)
        module_name = f'_cycept_module_{self.hash}'
        namespace = {}
        source_c = None
        html_annotation = None
        with (
            tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as dir_name,
            hack_sys_path(),
        ):
            # Write Cython source to file
            module_path = pathlib.Path(dir_name) / module_name
            module_path.with_suffix('.pyx').write_text(self.source, 'utf-8')
            # Call Cython. We do so within a subprocess in order
            # to capture stdout an stderr when running silently.
            cmd = [
                sys.executable,
                '-c',
                '; '.join([
                    'import cycept.compile',
                    'cycept.compile.compile({})'
                    .format(', '.join([
                        f'{str(module_path)!r}',
                        f'{optimizations!r}',
                        f'{c_lang!r}',
                        f'{html!r}',
                    ]))
                ]),
            ]
            run_kwargs = {}
            if silent:
                run_kwargs |= {
                    'stdout': subprocess.PIPE,
                    'stderr': subprocess.STDOUT,
                    'text': True,
                }
            tic = time.perf_counter()
            cproc = subprocess.run(cmd, **run_kwargs)
            toc = time.perf_counter()
            if cproc.returncode != 0:
                if sys.flags.interactive:
                    # Do not remove the compilation files immediately when
                    # running interactively, allowing the user to inspect them.
                    input(f'Press Enter to clean up temporary build directory {dir_name} ')
                msg = ['Cythonization failed.']
                if silent:
                    msg += ['Subprocess output:', cproc.stdout]
                raise OSError('\n'.join(msg))
            # Import function from compiled module into temporary namespace
            exec(f'import {module_name}', namespace)
            # Read in C source and annotated HTML
            for ext in ['c', 'cc', 'cpp', 'cxx']:
                if (path_c := module_path.with_suffix(f'.{ext}')).is_file():
                    source_c = path_c.read_text('utf-8')
                    break
            else:
                ext = None
            if (path_html := module_path.with_suffix('.html')).is_file():
                html_annotation = path_html.read_text('utf-8')
        # Extract compiled function
        module_compiled = namespace[module_name]
        module_compiled_dict = module_compiled.__dict__
        func_compiled = module_compiled_dict[self.func_name]
        # Replace fake globals and non-locals with actual
        # globals and non-locals within extension module.
        for name, val in collections.ChainMap(self.nonlocals, self.globals).items():
            if name == self.func_name:
                continue
            module_compiled_dict[name] = val
            module_compiled_dict[asciify(name)] = val
        # Store compilation products
        self.compiled = self.Compiled(
            func_compiled,
            module_compiled,
            source_c,
            ext,
            html_annotation,
        )
        # Return compilation time (in seconds)
        return toc - tic

    # Method for viewing the annotated HTML
    def __call__(self, dir_name=None):
        if self.compiled is None or self.compiled.html is None:
            print(f'No Cython HTML annotation generated for {self.func_name}()')
            return
        if dir_name is None:
            dir_name = tempfile.mkdtemp()  # not cleaned up
        module_path = pathlib.Path(dir_name) / self.compiled.module.__name__
        if self.compiled.source is not None and self.compiled.source_ext is not None:
            module_path.with_suffix(f'.{self.compiled.source_ext}').write_text(
                self.compiled.source,
                'utf-8',
             )
        path_html = module_path.with_suffix('.html')
        path_html.write_text(self.compiled.html, 'utf-8')
        webbrowser.open_new_tab(str(path_html.as_uri()))
        return dir_name

    # Method for pretty printing (showing types)
    def __repr__(self):
        from .core import prettify_type
        return f'{self.func_name}({{}})'.format(
            ', '.join(
                f'{name}: {prettify_type(tp)}'
                for name, tp in zip(self.arguments, self.arguments_types)
            )
        )

    # Method for pretty printing (showing module)
    def __str__(self):
        return f'function {self.func_name}() defined in {self.module}'

