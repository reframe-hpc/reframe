# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import reframe as rfm
import reframe.core.meta as meta


def test_class_attr_access():
    '''Test that `__getattr__` avoids infinite recursion.'''
    def my_test(key):
        class MyMeta(meta.RegressionTestMeta):
            def __init__(cls, name, bases, namespace, **kwargs):
                getattr(cls, f'{key}')

        msg = f'has no attribute {key!r}'
        with pytest.raises(AttributeError, match=msg):
            class Foo(metaclass=MyMeta):
                pass

    my_test('_rfm_var_space')
    my_test('_rfm_param_space')


def test_directives():
    '''Test that directives are not available as instance attributes.'''

    def ext_fn(x):
        pass

    class MyTest(metaclass=meta.RegressionTestMeta):
        p = parameter()
        v = variable(int)
        bind(ext_fn, name='ext')
        run_before('run')(ext)
        run_after('run')(ext)
        require_deps(ext)

        def __init__(self):
            assert not hasattr(self, 'parameter')
            assert not hasattr(self, 'variable')
            assert not hasattr(self, 'bind')
            assert not hasattr(self, 'run_before')
            assert not hasattr(self, 'run_after')
            assert not hasattr(self, 'require_deps')

    MyTest()


def test_bind_directive():
    def ext_fn(x):
        return x

    ext_fn._rfm_foo = True

    class MyTest(metaclass=meta.RegressionTestMeta):
        bind(ext_fn)
        bind(ext_fn, name='ext')

        # Bound as different objects
        assert ext_fn is not ext

        # Correct object type
        assert all([
            type(x) is meta.RegressionTestMeta.WrappedFunction
            for x in [ext_fn, ext]
        ])

        # Test __setattr__ and __getattr__
        assert hasattr(ext, '_rfm_foo')
        ext._rfm_foo = False
        assert ext._rfm_foo == ext.fn._rfm_foo
        assert ext_fn._rfm_foo

        def __init__(self):
            assert self.ext_fn() is self
            assert self.ext() is self

    # Test __get__
    MyTest()

    # Test __call__
    assert MyTest.ext_fn(2) == 2
    assert MyTest.ext(2) == 2
