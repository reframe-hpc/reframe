# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import pytest

import reframe as rfm
import reframe.core.runtime as rt
import reframe.frontend.dependency as dependency
import reframe.frontend.executors as executors
import reframe.utility as util
from reframe.core.environments import Environment
from reframe.core.exceptions import DependencyError
from reframe.frontend.loader import RegressionCheckLoader

import unittests.fixtures as fixtures


class Node:
    '''A node in the test case graph.

    It's simply a wrapper to a (test_name, partition, environment) tuple
    that can interact seemlessly with a real test case.
    It's meant for convenience in unit testing.
    '''

    def __init__(self, cname, pname, ename):
        self.cname, self.pname, self.ename = cname, pname, ename

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (self.cname == other.cname and
                    self.pname == other.pname and
                    self.ename == other.ename)

        if isinstance(other, executors.TestCase):
            return (self.cname == other.check.name and
                    self.pname == other.partition.fullname and
                    self.ename == other.environ.name)

        return NotImplemented

    def __hash__(self):
        return hash(self.cname) ^ hash(self.pname) ^ hash(self.ename)

    def __repr__(self):
        return 'Node(%r, %r, %r)' % (self.cname, self.pname, self.ename)


def has_edge(graph, src, dst):
    return dst in graph[src]


def num_deps(graph, cname):
    return sum(len(deps) for c, deps in graph.items()
               if c.check.name == cname)


def in_degree(graph, node):
    for v in graph.keys():
        if v == node:
            return v.num_dependents


def find_check(name, checks):
    for c in checks:
        if c.name == name:
            return c

    return None


def find_case(cname, ename, cases):
    for c in cases:
        if c.check.name == cname and c.environ.name == ename:
            return c


@pytest.fixture
def temp_runtime(tmp_path):
    def _temp_runtime(site_config, system=None, options={}):
        options.update({'systems/prefix': tmp_path})
        with rt.temp_runtime(site_config, system, options):
            yield rt.runtime

    yield _temp_runtime


@pytest.fixture
def exec_ctx(temp_runtime):
    yield from temp_runtime(fixtures.TEST_CONFIG_FILE, 'sys0')


@pytest.fixture
def loader():
    return RegressionCheckLoader([
        'unittests/resources/checks_unlisted/deps_simple.py'
    ])


def test_eq_hash(loader, exec_ctx):
    cases = executors.generate_testcases(loader.load_all())
    case0 = find_case('Test0', 'e0', cases)
    case1 = find_case('Test0', 'e1', cases)
    case0_copy = case0.clone()

    assert case0 == case0_copy
    assert hash(case0) == hash(case0_copy)
    assert case1 != case0
    assert hash(case1) != hash(case0)


def test_build_deps(loader, exec_ctx):
    checks = loader.load_all()
    cases = executors.generate_testcases(checks)

    # Test calling getdep() before having built the graph
    t = find_check('Test1_exact', checks)
    with pytest.raises(DependencyError):
        t.getdep('Test0', 'e0')

    # Build dependencies and continue testing
    deps = dependency.build_deps(cases)
    dependency.validate_deps(deps)

    # Check DEPEND_FULLY dependencies
    assert num_deps(deps, 'Test1_fully') == 8
    for p in ['sys0:p0', 'sys0:p1']:
        for e0 in ['e0', 'e1']:
            for e1 in ['e0', 'e1']:
                assert has_edge(deps,
                                Node('Test1_fully', p, e0),
                                Node('Test0', p, e1))

    # Check DEPEND_BY_ENV
    assert num_deps(deps, 'Test1_by_env') == 4
    assert num_deps(deps, 'Test1_default') == 4
    for p in ['sys0:p0', 'sys0:p1']:
        for e in ['e0', 'e1']:
            assert has_edge(deps,
                            Node('Test1_by_env', p, e),
                            Node('Test0', p, e))
            assert has_edge(deps,
                            Node('Test1_default', p, e),
                            Node('Test0', p, e))

    # Check DEPEND_EXACT
    assert num_deps(deps, 'Test1_exact') == 6
    for p in ['sys0:p0', 'sys0:p1']:
        assert has_edge(deps,
                        Node('Test1_exact', p, 'e0'),
                        Node('Test0', p, 'e0'))
        assert has_edge(deps,
                        Node('Test1_exact', p, 'e0'),
                        Node('Test0', p, 'e1'))
        assert has_edge(deps,
                        Node('Test1_exact', p, 'e1'),
                        Node('Test0', p, 'e1'))

    # Check in-degree of Test0

    # 2 from Test1_fully,
    # 1 from Test1_by_env,
    # 1 from Test1_exact,
    # 1 from Test1_default
    assert in_degree(deps, Node('Test0', 'sys0:p0', 'e0')) == 5
    assert in_degree(deps, Node('Test0', 'sys0:p1', 'e0')) == 5

    # 2 from Test1_fully,
    # 1 from Test1_by_env,
    # 2 from Test1_exact,
    # 1 from Test1_default
    assert in_degree(deps, Node('Test0', 'sys0:p0', 'e1')) == 6
    assert in_degree(deps, Node('Test0', 'sys0:p1', 'e1')) == 6

    # Pick a check to test getdep()
    check_e0 = find_case('Test1_exact', 'e0', cases).check
    check_e1 = find_case('Test1_exact', 'e1', cases).check

    with pytest.raises(DependencyError):
        check_e0.getdep('Test0')

    # Set the current environment
    check_e0._current_environ = Environment('e0')
    check_e1._current_environ = Environment('e1')

    assert check_e0.getdep('Test0', 'e0').name == 'Test0'
    assert check_e0.getdep('Test0', 'e1').name == 'Test0'
    assert check_e1.getdep('Test0', 'e1').name == 'Test0'
    with pytest.raises(DependencyError):
        check_e0.getdep('TestX', 'e0')

    with pytest.raises(DependencyError):
        check_e0.getdep('Test0', 'eX')

    with pytest.raises(DependencyError):
        check_e1.getdep('Test0', 'e0')


def test_build_deps_unknown_test(loader, exec_ctx):
    checks = loader.load_all()

    # Add some inexistent dependencies
    test0 = find_check('Test0', checks)
    for depkind in ('default', 'fully', 'by_env', 'exact'):
        test1 = find_check('Test1_' + depkind, checks)
        if depkind == 'default':
            test1.depends_on('TestX')
        elif depkind == 'exact':
            test1.depends_on('TestX', rfm.DEPEND_EXACT, {'e0': ['e0']})
        elif depkind == 'fully':
            test1.depends_on('TestX', rfm.DEPEND_FULLY)
        elif depkind == 'by_env':
            test1.depends_on('TestX', rfm.DEPEND_BY_ENV)

        with pytest.raises(DependencyError):
            dependency.build_deps(executors.generate_testcases(checks))


def test_build_deps_unknown_target_env(loader, exec_ctx):
    checks = loader.load_all()

    # Add some inexistent dependencies
    test0 = find_check('Test0', checks)
    test1 = find_check('Test1_default', checks)
    test1.depends_on('Test0', rfm.DEPEND_EXACT, {'e0': ['eX']})
    with pytest.raises(DependencyError):
        dependency.build_deps(executors.generate_testcases(checks))


def test_build_deps_unknown_source_env(loader, exec_ctx):
    checks = loader.load_all()

    # Add some inexistent dependencies
    test0 = find_check('Test0', checks)
    test1 = find_check('Test1_default', checks)
    test1.depends_on('Test0', rfm.DEPEND_EXACT, {'eX': ['e0']})

    # Unknown source is ignored, because it might simply be that the test
    # is not executed for eX
    deps = dependency.build_deps(executors.generate_testcases(checks))
    assert num_deps(deps, 'Test1_default') == 4


def test_build_deps_empty(exec_ctx):
    assert {} == dependency.build_deps([])


@pytest.fixture
def make_test():
    class MyTest(rfm.RegressionTest):
        def __init__(self, name):
            self.name = name
            self.valid_systems = ['*']
            self.valid_prog_environs = ['*']
            self.executable = 'echo'
            self.executable_opts = [name]

    def _make_test(name):
        return MyTest(name)

    return _make_test


def test_valid_deps(make_test, exec_ctx):
    #
    #       t0       +-->t5<--+
    #       ^        |        |
    #       |        |        |
    #   +-->t1<--+   t6       t7
    #   |        |            ^
    #   t2<------t3           |
    #   ^        ^            |
    #   |        |            t8
    #   +---t4---+
    #
    t0 = make_test('t0')
    t1 = make_test('t1')
    t2 = make_test('t2')
    t3 = make_test('t3')
    t4 = make_test('t4')
    t5 = make_test('t5')
    t6 = make_test('t6')
    t7 = make_test('t7')
    t8 = make_test('t8')
    t1.depends_on('t0')
    t2.depends_on('t1')
    t3.depends_on('t1')
    t3.depends_on('t2')
    t4.depends_on('t2')
    t4.depends_on('t3')
    t6.depends_on('t5')
    t7.depends_on('t5')
    t8.depends_on('t7')
    dependency.validate_deps(
        dependency.build_deps(
            executors.generate_testcases([t0, t1, t2, t3, t4,
                                          t5, t6, t7, t8])
        )
    )


def test_cyclic_deps(make_test, exec_ctx):
    #
    #       t0       +-->t5<--+
    #       ^        |        |
    #       |        |        |
    #   +-->t1<--+   t6       t7
    #   |   |    |            ^
    #   t2  |    t3           |
    #   ^   |    ^            |
    #   |   v    |            t8
    #   +---t4---+
    #
    t0 = make_test('t0')
    t1 = make_test('t1')
    t2 = make_test('t2')
    t3 = make_test('t3')
    t4 = make_test('t4')
    t5 = make_test('t5')
    t6 = make_test('t6')
    t7 = make_test('t7')
    t8 = make_test('t8')
    t1.depends_on('t0')
    t1.depends_on('t4')
    t2.depends_on('t1')
    t3.depends_on('t1')
    t4.depends_on('t2')
    t4.depends_on('t3')
    t6.depends_on('t5')
    t7.depends_on('t5')
    t8.depends_on('t7')
    deps = dependency.build_deps(
        executors.generate_testcases([t0, t1, t2, t3, t4,
                                      t5, t6, t7, t8])
    )

    with pytest.raises(DependencyError) as exc_info:
        dependency.validate_deps(deps)

    assert ('t4->t2->t1->t4' in str(exc_info.value) or
            't2->t1->t4->t2' in str(exc_info.value) or
            't1->t4->t2->t1' in str(exc_info.value) or
            't1->t4->t3->t1' in str(exc_info.value) or
            't4->t3->t1->t4' in str(exc_info.value) or
            't3->t1->t4->t3' in str(exc_info.value))


def test_cyclic_deps_by_env(make_test, exec_ctx):
    t0 = make_test('t0')
    t1 = make_test('t1')
    t1.depends_on('t0', rfm.DEPEND_EXACT, {'e0': ['e0']})
    t0.depends_on('t1', rfm.DEPEND_EXACT, {'e1': ['e1']})
    deps = dependency.build_deps(
        executors.generate_testcases([t0, t1])
    )
    with pytest.raises(DependencyError) as exc_info:
        dependency.validate_deps(deps)

    assert ('t1->t0->t1' in str(exc_info.value) or
            't0->t1->t0' in str(exc_info.value))


def test_validate_deps_empty(exec_ctx):
    dependency.validate_deps({})


def assert_topological_order(cases, graph):
    cases_order = []
    visited_tests = set()
    tests = util.OrderedSet()
    for c in cases:
        check, part, env = c
        cases_order.append((check.name, part.fullname, env.name))
        tests.add(check.name)
        visited_tests.add(check.name)

        # Assert that all dependencies of c have been visited before
        for d in graph[c]:
            if d not in cases:
                # dependency points outside the subgraph
                continue

            assert d.check.name in visited_tests

    # Check the order of systems and prog. environments
    # We are checking against all possible orderings
    valid_orderings = []
    for partitions in itertools.permutations(['sys0:p0', 'sys0:p1']):
        for environs in itertools.permutations(['e0', 'e1']):
            ordering = []
            for t in tests:
                for p in partitions:
                    for e in environs:
                        ordering.append((t, p, e))

            valid_orderings.append(ordering)

    assert cases_order in valid_orderings


def test_toposort(make_test, exec_ctx):
    #
    #       t0       +-->t5<--+
    #       ^        |        |
    #       |        |        |
    #   +-->t1<--+   t6       t7
    #   |        |            ^
    #   t2<------t3           |
    #   ^        ^            |
    #   |        |            t8
    #   +---t4---+
    #
    t0 = make_test('t0')
    t1 = make_test('t1')
    t2 = make_test('t2')
    t3 = make_test('t3')
    t4 = make_test('t4')
    t5 = make_test('t5')
    t6 = make_test('t6')
    t7 = make_test('t7')
    t8 = make_test('t8')
    t1.depends_on('t0')
    t2.depends_on('t1')
    t3.depends_on('t1')
    t3.depends_on('t2')
    t4.depends_on('t2')
    t4.depends_on('t3')
    t6.depends_on('t5')
    t7.depends_on('t5')
    t8.depends_on('t7')
    deps = dependency.build_deps(
        executors.generate_testcases([t0, t1, t2, t3, t4,
                                      t5, t6, t7, t8])
    )
    cases = dependency.toposort(deps)
    assert_topological_order(cases, deps)


def test_toposort_subgraph(make_test, exec_ctx):
    #
    #       t0
    #       ^
    #       |
    #   +-->t1<--+
    #   |        |
    #   t2<------t3
    #   ^        ^
    #   |        |
    #   +---t4---+
    #
    t0 = make_test('t0')
    t1 = make_test('t1')
    t2 = make_test('t2')
    t3 = make_test('t3')
    t4 = make_test('t4')
    t1.depends_on('t0')
    t2.depends_on('t1')
    t3.depends_on('t1')
    t3.depends_on('t2')
    t4.depends_on('t2')
    t4.depends_on('t3')
    full_deps = dependency.build_deps(
        executors.generate_testcases([t0, t1, t2, t3, t4])
    )
    partial_deps = dependency.build_deps(
        executors.generate_testcases([t3, t4]), full_deps
    )
    cases = dependency.toposort(partial_deps, is_subgraph=True)
    assert_topological_order(cases, partial_deps)
