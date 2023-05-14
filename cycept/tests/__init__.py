import contextlib
import sys

from .test_bench import bench


def test(kind=None):
    testfiles = ['cycept.tests']
    if isinstance(kind, str):
        kind = [kind]
    if isinstance(kind, (list, tuple)):
        kind = [k.removeprefix('test_') for k in kind]
        testfiles = [f'cycept.tests.test_{k}' for k in kind]
    # Run pytest on the cycept.test submodule.
    # We prevent pytest from writing to __pycache__ by hacking on
    # sys.dont_write_bytecode. We do this as such files are not cleanly
    # removed by 'pip uninstall cycept'.
    @contextlib.contextmanager
    def hack_sys_dont_write_bytecode():
        backup = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            yield
        except Exception:
            raise
        finally:
            sys.dont_write_bytecode = backup
    with hack_sys_dont_write_bytecode():
        import pytest
        pytest.main(['-p', 'no:cacheprovider', '--pyargs', *testfiles])

