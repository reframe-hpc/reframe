# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import reframe.core.meta as meta
import reframe.core.deferrable as deferrable

from reframe.core.exceptions import ReframeSyntaxError


@pytest.fixture
def MyMeta():
    '''Utility fixture just for convenience.'''
    class Foo(metaclass=meta.RegressionTestMeta):
        pass

    yield Foo


def test_class_attr_access():
    '''Catch access to sub-namespaces when they do not exist.'''
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


def test_directives(MyMeta):
    '''Test that directives are not available as instance attributes.'''

    def ext_fn(x):
        pass

    class MyTest(MyMeta):
        p = parameter()
        v = variable(int)
        bind(ext_fn, name='ext')
        run_before('run')(ext)
        run_after('run')(ext)
        require_deps(ext)
        deferrable(ext)
        sanity_function(ext)
        v = required

        def __init__(self):
            assert not hasattr(self, 'parameter')
            assert not hasattr(self, 'variable')
            assert not hasattr(self, 'bind')
            assert not hasattr(self, 'run_before')
            assert not hasattr(self, 'run_after')
            assert not hasattr(self, 'require_deps')
            assert not hasattr(self, 'deferrable')
            assert not hasattr(self, 'sanity_function')
            assert not hasattr(self, 'required')

    MyTest()


def test_bind_directive(MyMeta):
    def ext_fn(x):
        return x

    ext_fn._rfm_foo = True

    class MyTest(MyMeta):
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


def test_sanity_function_decorator(MyMeta):
    class Foo(MyMeta):
        @sanity_function
        def my_sanity(self):
            return True

    assert hasattr(Foo, '_rfm_sanity')
    assert Foo._rfm_sanity.__name__ == 'my_sanity'
    assert type(Foo._rfm_sanity()) is deferrable._DeferredExpression

    # Test override sanity
    class Bar(Foo):
        @sanity_function
        def extended_sanity(self):
            return self.my_sanity()

    assert hasattr(Bar, '_rfm_sanity')
    assert Bar._rfm_sanity.__name__ == 'extended_sanity'
    assert type(Bar._rfm_sanity()) is deferrable._DeferredExpression

    # Test bases lookup
    class Baz(MyMeta):
        pass

    class MyTest(Baz, Foo):
        pass

    assert hasattr(MyTest, '_rfm_sanity')
    assert MyTest._rfm_sanity.__name__ == 'my_sanity'
    assert type(MyTest._rfm_sanity()) is deferrable._DeferredExpression

    # Test incomplete sanity override
    with pytest.raises(ReframeSyntaxError):
        class MyWrongTest(Foo):
            def my_sanity(self):
                pass

    # Test error when double-declaring @sanity_function in the same class
    with pytest.raises(ReframeSyntaxError):
        class MyWrongTest(MyMeta):
            @sanity_function
            def sn_fn_a(self):
                pass

            @sanity_function
            def sn_fn_b(self):
                pass


def test_deferrable_decorator(MyMeta):
    class MyTest(MyMeta):
        @deferrable
        def my_deferrable(self):
            pass

    assert type(MyTest.my_deferrable()) is deferrable._DeferredExpression


def test_hook_attachments(MyMeta):
    class Foo(MyMeta):
        @run_after('setup')
        def hook_a(self):
            pass

        @run_before('compile')
        def hook_b(self):
            pass

        @run_after('run')
        def hook_c(self):
            pass

        @classmethod
        def hook_in_stage(cls, hook, stage):
            '''Assert that a hook is in a given registry stage.'''
            try:
                return hook in {h.__name__
                                for h in cls._rfm_pipeline_hooks[stage]}
            except KeyError:
                return False

    assert Foo.hook_in_stage('hook_a', 'post_setup')
    assert Foo.hook_in_stage('hook_b', 'pre_compile')
    assert Foo.hook_in_stage('hook_c', 'post_run_wait')

    class Bar(Foo):
        @run_before('sanity')
        def hook_a(self):
            "Convert to a pre-sanity hook"

        def hook_b(self):
            "No longer a hook"

    assert not Bar.hook_in_stage('hook_a', 'post_setup')
    assert not Bar.hook_in_stage('hook_b', 'pre_compile')
    assert Bar.hook_in_stage('hook_c', 'post_run_wait')
    assert Bar.hook_in_stage('hook_a', 'pre_sanity')
