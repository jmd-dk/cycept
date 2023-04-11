def test():
    import pytest
    pytest.main(['-q', '-p', 'no:cacheprovider', '--pyargs', 'cycept.tests'])

