import contextlib
import sys

def test():
    # Run pytest on the cycept.test package.
    # We prevent pytest from writing to __pycache__ by hacking on
    # sys.dont_write_bytecode. We do this as such files are not cleanly
    # removed by 'pip uninstall cycept'.
    @contextlib.contextmanager
    def hack_sys_dont_write_bytecode():
        backup = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        yield
        sys.dont_write_bytecode = backup
    with hack_sys_dont_write_bytecode():
        import pytest
        pytest.main(['-q', '-p', 'no:cacheprovider', '--pyargs', 'cycept.tests'])

