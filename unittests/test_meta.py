# Copyright 2016-2023 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import reframe as rfm
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

    class MyFixture(rfm.RunOnlyRegressionTest):
        pass

    class MyTest(MyMeta):
        p = parameter()
        v = variable(int)
        f = fixture(MyFixture)
        bind(ext_fn, name='ext')
        run_before('run')(ext)
        run_after('run')(ext)
        require_deps(ext)
        deferrable(ext)
        sanity_function(ext)
        v = required
        final(ext)
        performance_function('some_units')(ext)

        def __init__(self):
            assert not hasattr(self, 'parameter')
            assert not hasattr(self, 'variable')
            assert not hasattr(self, 'fixture')
            assert not hasattr(self, 'bind')
            assert not hasattr(self, 'run_before')
            assert not hasattr(self, 'run_after')
            assert not hasattr(self, 'require_deps')
            assert not hasattr(self, 'deferrable')
            assert not hasattr(self, 'sanity_function')
            assert not hasattr(self, 'required')
            assert not hasattr(self, 'final')
            assert not hasattr(self, 'performance_function')

    MyTest()


def test_bind_directive(MyMeta):
    def ext_fn(x):
        return x

    ext_fn._rfm_foo = True

    class MyTest(MyMeta):
        bind(ext_fn)
        bind(ext_fn, name='ext')

        # Catch bug #2146
        final(bind(ext_fn, name='my_final'))

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

    # Catch bug #2146
    assert 'my_final' in MyTest._rfm_final_methods

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
        '''Base class with three random hooks.

        This class has the ``hook_in_stage`` method, which asserts that a given
        hook is registered into a specified stage.
        '''

        @run_after('setup')
        def hook_a(self):
            pass

        @run_before('compile', always_last=True)
        def hook_b(self):
            pass

        @run_after('run')
        def hook_c(self):
            pass

        @classmethod
        def hook_in_stage(cls, hook, stage, always_last=False):
            '''Assert that a hook is in a given registry stage.'''
            for h in cls._rfm_hook_registry:
                if h.__name__ == hook:
                    if (stage, always_last) in h.stages:
                        return True

                    break

            return False

    assert Foo.hook_in_stage('hook_a', 'post_setup')
    assert Foo.hook_in_stage('hook_b', 'pre_compile', True)
    assert Foo.hook_in_stage('hook_c', 'post_run')

    class Bar(Foo):
        '''Derived class that overrides and invalidates hooks from Foo.'''

        @run_before('sanity')
        def hook_a(self):
            '''Convert to a pre-sanity hook'''

        def hook_b(self):
            '''No longer a hook'''

    assert not Bar.hook_in_stage('hook_a', 'post_setup')
    assert not Bar.hook_in_stage('hook_b', 'pre_compile')
    assert Bar.hook_in_stage('hook_c', 'post_run')
    assert Bar.hook_in_stage('hook_a', 'pre_sanity')

    class Baz(MyMeta):
        '''Class to test hook attachments with multiple inheritance.'''

        @run_before('setup')
        @run_after('compile')
        def hook_a(self):
            '''Force a name-clash with hook_a from Bar.'''

        @run_before('run')
        def hook_d(self):
            '''An extra hook to attach.'''

    class MyTest(Bar, Baz):
        '''Test multiple inheritance override.'''

    assert MyTest.hook_in_stage('hook_a', 'pre_sanity')
    assert not MyTest.hook_in_stage('hook_a', 'pre_setup')
    assert not MyTest.hook_in_stage('hook_a', 'post_compile')
    assert MyTest.hook_in_stage('hook_c', 'post_run')
    assert MyTest.hook_in_stage('hook_d', 'pre_run')


def test_final(MyMeta):
    class Base(MyMeta):
        @final
        def foo(self):
            pass

    with pytest.raises(ReframeSyntaxError):
        class Derived(Base):
            def foo(self):
                '''Override attempt.'''

    class AllowFinalOverride(Base):
        '''Use flag to bypass the final override check.'''
        _rfm_override_final = True

        def foo(self):
            '''Overriding foo is now allowed.'''


def test_callable_attributes(MyMeta):
    '''Test issue #2113.

    Setting a callable without the __name__ attribute would crash the
    metaclass.
    '''

    class Callable:
        def __call__(self):
            pass

    class Base(MyMeta):
        f = Callable()


def test_performance_function(MyMeta):
    assert hasattr(MyMeta, '_rfm_perf_fns')

    class Base(MyMeta):
        @performance_function('units')
        def perf_a(self):
            pass

        @performance_function('units')
        def perf_b(self):
            pass

        def assert_perf_fn_return(self):
            assert isinstance(
                self.perf_a(), deferrable._DeferredPerformanceExpression
            )

    # Test the return type of the performance functions
    Base().assert_perf_fn_return()

    # Test the performance function dict has been built correctly
    perf_dict = {fn for fn in Base._rfm_perf_fns}
    assert perf_dict == {'perf_a', 'perf_b'}

    class Derived(Base):
        '''Test perf fn inheritance and perf_key argument.'''

        def perf_a(self):
            '''Override perf fn with a non perf fn.'''

        @performance_function('units')
        def perf_c(self):
            '''Add a new perf fn.'''

        @performance_function('units', perf_key='my_perf_fn')
        def perf_d(self):
            '''Perf function with custom key.'''

    # Test the performance function set is correct with class inheritance
    perf_dict = {fn._rfm_perf_key for fn in Derived._rfm_perf_fns.values()}
    assert perf_dict == {'perf_b', 'perf_c', 'my_perf_fn'}

    # Test multiple inheritance and name conflict resolution
    class ClashingBase(MyMeta):
        @performance_function('units', perf_key='clash')
        def perf_a(self):
            return 'A'

    class Join(ClashingBase, Base):
        '''Test that we follow MRO's order.'''

    class JoinAndOverride(ClashingBase, Base):
        @performance_function('units')
        def perf_a(self):
            return 'B'

    assert Join._rfm_perf_fns['perf_a']('self').evaluate() == 'A'
    assert JoinAndOverride._rfm_perf_fns['perf_a']('self').evaluate() == 'B'


def test_double_define_performance_function(MyMeta):
    with pytest.raises(ReframeSyntaxError):
        class Foo(MyMeta):
            @performance_function('unit')
            def foo(self):
                pass

            @performance_function('unit')
            def foo(self):
                '''This doesn't make sense, so we raise an error'''


def test_performance_function_errors(MyMeta):
    with pytest.raises(TypeError):
        class wrong_perf_key_type(MyMeta):
            @performance_function('units', perf_key=3)
            def perf_fn(self):
                pass

    with pytest.raises(TypeError):
        class wrong_function_signature(MyMeta):
            @performance_function('units')
            def perf_fn(self, extra_arg):
                pass

    with pytest.raises(TypeError):
        class wrong_units(MyMeta):
            @performance_function(5)
            def perf_fn(self):
                pass


def test_setting_variables_on_instantiation(MyMeta):
    class Foo(MyMeta):
        v = variable(int, value=1)

    assert Foo().v == 1
    assert Foo(fixt_vars={'v': 10}).v == 10

    # Non-variables are silently ignored
    assert not hasattr(Foo(fixt_vars={'vv': 10}), 'vv')

    with pytest.raises(TypeError):
        Foo(fixt_vars='not a mapping')


def test_variants(MyMeta):
    class Foo(MyMeta):
        p = parameter(['a', 'b'])

    assert Foo.num_variants == 2
    assert Foo.get_variant_info(0)['params']['p'] == 'a'
    assert Foo(variant_num=0).p == 'a'
    assert Foo(variant_num=1).p == 'b'


def test_get_info(MyMeta):
    class Fixt(rfm.RegressionTest):
        p = parameter(['a', 'b'])

    class Foo(rfm.RegressionTest):
        f = fixture(Fixt)

    assert Foo.num_variants == 2
    assert Foo.get_variant_info(0, recurse=False) == {
        'params': {},
        'fixtures': {
            'f': (0,)
        }
    }

    class Bar(MyMeta):
        p = parameter(['c'])
        f = fixture(Foo)

    assert Bar.get_variant_info(0, recurse=False) == {
        'params': {
            'p': 'c'
        },
        'fixtures': {
            'f': (0,)
        }
    }
    assert Bar.get_variant_info(0, recurse=True, max_depth=0) == {
        'params': {
            'p': 'c'
        },
        'fixtures': {
            'f': (0,)
        }
    }
    assert Bar.get_variant_info(0, recurse=True, max_depth=1) == {
        'params': {
            'p': 'c'
        },
        'fixtures': {
            'f': {
                'params': {},
                'fixtures': {
                    'f': (0,)
                }
            }
        }
    }
    assert Bar.get_variant_info(0, recurse=True) == {
        'params': {
            'p': 'c'
        },
        'fixtures': {
            'f': {
                'params': {},
                'fixtures': {
                    'f': {
                        'params': {
                            'p': 'a'
                        },
                        'fixtures': {}
                    }
                }
            }
        }
    }

    class Baz(Bar):
        ff = fixture(Fixt, action='join')

    assert Baz.get_variant_info(0, recurse=True) == {
        'params': {
            'p': 'c'
        },
        'fixtures': {
            'f': {
                'params': {},
                'fixtures': {
                    'f': {
                        'params': {
                            'p': 'a'
                        },
                        'fixtures': {}
                    }
                },
            },
            'ff': (0, 1,)
        }
    }


def test_get_variant_nums(MyMeta):
    class Foo(MyMeta):
        p = parameter(range(10))
        q = parameter(range(10))

    variants = Foo.get_variant_nums(p=lambda x: x < 5, q=lambda x: x > 3)
    for v in variants:
        assert Foo.get_variant_info(v)['params']['p'] < 5
        assert Foo.get_variant_info(v)['params']['q'] > 3

    assert Foo.get_variant_nums() == list(range(Foo.num_variants))

    # Check condensed syntax
    variants = Foo.get_variant_nums(p=5, q=4)
    for v in variants:
        assert Foo.get_variant_info(v)['params']['p'] == 5
        assert Foo.get_variant_info(v)['params']['q'] == 4


def test_loggable_attrs():
    class T(metaclass=meta.RegressionTestMeta):
        x = variable(int, value=3, loggable=True)
        y = variable(int, loggable=True)    # loggable but undefined
        z = variable(int)
        p = parameter(range(3), loggable=True)

        @loggable
        @property
        def foo(self):
            return 10

        @loggable_as('w')
        @property
        def bar(self):
            return 10

        @run_after('init')
        def set_z(self):
            self.z = 20

    assert T.loggable_attrs() == [('bar', 'w'), ('foo', None),
                                  ('p', None), ('x', None), ('y', None)]
    assert T(variant_num=0).foo == 10
    assert T(variant_num=0).bar == 10

    # Test error conditions
    with pytest.raises(ValueError):
        class T(metaclass=meta.RegressionTestMeta):
            @loggable
            def foo(self):
                pass


def test_inherited_loggable_attrs():
    class T(rfm.RegressionTest):
        pass

    attrs = [x[0] for x in T.loggable_attrs()]
    assert 'num_tasks' in attrs
    assert 'prefix' in attrs


def test_deprecated_loggable_attrs():
    class T(metaclass=meta.RegressionTestMeta):
        x = deprecate(variable(int, value=3, loggable=True), 'deprecated')

    assert T.loggable_attrs() == [('x', None)]
