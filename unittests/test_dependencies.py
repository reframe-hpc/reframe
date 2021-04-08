# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import functools
import itertools
import pytest

import reframe as rfm
import reframe.core.runtime as rt
import reframe.frontend.dependencies as dependencies
import reframe.frontend.executors as executors
import reframe.utility as util
import reframe.utility.udeps as udeps
from reframe.core.environments import Environment
from reframe.core.exceptions import DependencyError
from reframe.core.warnings import ReframeDeprecationWarning
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


def find_case(cname, ename, partname, cases):
    for c in cases:
        if (c.check.name == cname and
            c.environ.name == ename and
            c.partition.name == partname):
            return c


@pytest.fixture
def temp_runtime(tmp_path):
    def _temp_runtime(site_config, system=None, options=None):
        options = options or {}
        options.update({'systems/prefix': tmp_path})
        with rt.temp_runtime(site_config, system, options):
            yield

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
    case0 = find_case('Test0', 'e0', 'p0', cases)
    case1 = find_case('Test0', 'e1', 'p0', cases)
    case0_copy = case0.clone()

    assert case0 == case0_copy
    assert hash(case0) == hash(case0_copy)
    assert case1 != case0
    assert hash(case1) != hash(case0)


def test_dependecies_how_functions():
    t0_cases = [(p, e)
                for p in ['p0', 'p1']
                for e in ['e0', 'e1', 'e2']]
    t1_cases = [(p, e)
                for p in ['p0', 'p1', 'p2']
                for e in ['e0', 'e1']]

    how = udeps.fully
    deps = {(t0, t1)
            for t0 in t0_cases
            for t1 in t1_cases
            if how(t0, t1)}

    assert deps == {
        (t0, t1)
        for t0 in t0_cases
        for t1 in t1_cases
    }
    assert len(deps) == 36

    how = udeps.by_part
    deps = {(t0, t1)
            for t0 in t0_cases
            for t1 in t1_cases
            if how(t0, t1)}
    assert deps == {
        (t0, t1)
        for t0 in t0_cases
        for t1 in t1_cases
        if t0[0] == t1[0]
    }
    assert len(deps) == 12

    how = udeps.by_xpart
    deps = {(t0, t1)
            for t0 in t0_cases
            for t1 in t1_cases
            if how(t0, t1)}
    assert deps == {
        (t0, t1)
        for t0 in t0_cases
        for t1 in t1_cases
        if t0[0] != t1[0]
    }
    assert len(deps) == 24

    how = udeps.by_env
    deps = {(t0, t1)
            for t0 in t0_cases
            for t1 in t1_cases
            if how(t0, t1)}
    assert deps == {
        (t0, t1) for t0 in t0_cases
        for t1 in t1_cases
        if t0[1] == t1[1]
    }
    assert len(deps) == 12

    how = udeps.by_xenv
    deps = {(t0, t1)
            for t0 in t0_cases
            for t1 in t1_cases
            if how(t0, t1)}
    assert deps == {
        (t0, t1) for t0 in t0_cases
        for t1 in t1_cases
        if t0[1] != t1[1]
    }
    assert len(deps) == 24

    how = udeps.by_case
    deps = {(t0, t1)
            for t0 in t0_cases
            for t1 in t1_cases
            if how(t0, t1)}
    assert deps == {
        (t0, t1)
        for t0 in t0_cases
        for t1 in t1_cases
        if (t0[0] == t1[0] and t0[1] == t1[1])
    }
    assert len(deps) == 4

    how = udeps.by_xcase
    deps = {(t0, t1)
            for t0 in t0_cases
            for t1 in t1_cases
            if how(t0, t1)}
    assert deps == {
        (t0, t1)
        for t0 in t0_cases
        for t1 in t1_cases
        if (t0[0] != t1[0] or t0[1] != t1[1])
    }
    assert len(deps) == 32


def test_dependecies_how_functions_undoc():
    t0_cases = [(p, e)
                for p in ['p0', 'p1']
                for e in ['e0', 'e1', 'e2']]
    t1_cases = [(p, e)
                for p in ['p0', 'p1', 'p2']
                for e in ['e0', 'e1']]

    how = udeps.part_is('p0')
    deps = {(t0, t1) for t0 in t0_cases
            for t1 in t1_cases
            if how(t0, t1)}
    assert deps == {
        (t0, t1)
        for t0 in t0_cases
        for t1 in t1_cases
        if (t0[0] == 'p0' and t1[0] == 'p0')
    }
    assert len(deps) == 6

    how = udeps.source(udeps.part_is('p0'))
    deps = {(t0, t1)
            for t0 in t0_cases
            for t1 in t1_cases
            if how(t0, t1)}
    assert deps == {
        (t0, t1)
        for t0 in t0_cases
        for t1 in t1_cases
        if t0[0] == 'p0'
    }
    assert len(deps) == 18

    how = udeps.dest(udeps.part_is('p0'))
    deps = {(t0, t1)
            for t0 in t0_cases
            for t1 in t1_cases
            if how(t0, t1)}
    assert deps == {
        (t0, t1)
        for t0 in t0_cases
        for t1 in t1_cases
        if t1[0] == 'p0'
    }
    assert len(deps) == 12

    how = udeps.env_is('e0')
    deps = {(t0, t1)
            for t0 in t0_cases
            for t1 in t1_cases
            if how(t0, t1)}
    assert deps == {
        (t0, t1)
        for t0 in t0_cases
        for t1 in t1_cases
        if (t0[1] == 'e0' and t1[1] == 'e0')
    }
    assert len(deps) == 6

    how = udeps.source(udeps.env_is('e0'))
    deps = {(t0, t1)
            for t0 in t0_cases
            for t1 in t1_cases
            if how(t0, t1)}
    assert deps == {
        (t0, t1)
        for t0 in t0_cases
        for t1 in t1_cases
        if t0[1] == 'e0'
    }
    assert len(deps) == 12

    how = udeps.dest(udeps.env_is('e0'))
    deps = {(t0, t1)
            for t0 in t0_cases
            for t1 in t1_cases
            if how(t0, t1)}
    assert deps == {
        (t0, t1)
        for t0 in t0_cases
        for t1 in t1_cases
        if t1[1] == 'e0'
    }
    assert len(deps) == 18

    how = udeps.any(udeps.source(udeps.part_is('p0')),
                    udeps.dest(udeps.env_is('e1')))
    deps = {(t0, t1)
            for t0 in t0_cases
            for t1 in t1_cases
            if how(t0, t1)}
    assert deps == {
        (t0, t1)
        for t0 in t0_cases
        for t1 in t1_cases
        if (t0[0] == 'p0' or t1[1] == 'e1')
    }
    assert len(deps) == 27

    how = udeps.all(udeps.source(udeps.part_is('p0')),
                    udeps.dest(udeps.env_is('e1')))
    deps = {(t0, t1)
            for t0 in t0_cases
            for t1 in t1_cases
            if how(t0, t1)}
    assert deps == {
        (t0, t1)
        for t0 in t0_cases
        for t1 in t1_cases
        if (t0[0] == 'p0' and t1[1] == 'e1')
    }
    assert len(deps) == 9


def test_build_deps_deprecated_syntax(loader, exec_ctx):
    class Test0(rfm.RegressionTest):
        def __init__(self):
            self.valid_systems = ['sys0:p0', 'sys0:p1']
            self.valid_prog_environs = ['e0', 'e1']
            self.executable = 'echo'
            self.executable_opts = [self.name]
            self.sanity_patterns = sn.assert_found(self.name, self.stdout)

    @rfm.parameterized_test(*([kind] for kind in ['fully', 'by_case',
                                                  'exact']))
    class Test1_deprecated(rfm.RunOnlyRegressionTest):
        def __init__(self, kind):
            kindspec = {
                'fully': rfm.DEPEND_FULLY,
                'by_case': rfm.DEPEND_BY_ENV,
                'exact': rfm.DEPEND_EXACT,
            }
            self.valid_systems = ['sys0:p0', 'sys0:p1']
            self.valid_prog_environs = ['e0', 'e1']
            self.executable = 'echo'
            self.executable_opts = [self.name]
            if kindspec[kind] == rfm.DEPEND_EXACT:
                self.depends_on('Test0', kindspec[kind],
                                {'e0': ['e0', 'e1'], 'e1': ['e1']})
            else:
                self.depends_on('Test0', kindspec[kind])

    with pytest.warns(ReframeDeprecationWarning):
        t1 = Test1_deprecated('fully')
        assert(t1._userdeps == [('Test0', udeps.by_part)])

    with pytest.warns(ReframeDeprecationWarning):
        t1 = Test1_deprecated('by_case')
        assert(t1._userdeps == [('Test0', udeps.by_case)])

    with pytest.warns(ReframeDeprecationWarning):
        t1 = Test1_deprecated('exact')
        how = t1._userdeps[0][1]
        t0_cases = [(p, e) for p in ['p0', 'p1']
                    for e in ['e0', 'e1']]
        t1_cases = [(p, e) for p in ['p0', 'p1']
                    for e in ['e0', 'e1']]
        deps = {(t0, t1) for t0 in t0_cases
                for t1 in t1_cases if how(t0, t1)}
        assert deps == {
            (t0, t1) for t0 in t0_cases
            for t1 in t1_cases
            if ((t0[0] == t1[0] and t0[1] == 'e0') or
                (t0[0] == t1[0] and t0[1] == 'e1' and t1[1] == 'e1'))
        }
        assert len(deps) == 6


def test_build_deps(loader, exec_ctx):
    checks = loader.load_all()
    cases = executors.generate_testcases(checks)

    # Test calling getdep() before having built the graph
    t = find_check('Test1_fully', checks)
    with pytest.raises(DependencyError):
        t.getdep('Test0', 'e0', 'p0')

    # Build dependencies and continue testing
    deps, _ = dependencies.build_deps(cases)
    dependencies.validate_deps(deps)

    # Check dependencies for fully connected graph
    assert num_deps(deps, 'Test1_fully') == 16
    for p0 in ['sys0:p0', 'sys0:p1']:
        for p1 in ['sys0:p0', 'sys0:p1']:
            for e0 in ['e0', 'e1']:
                for e1 in ['e0', 'e1']:
                    assert has_edge(deps,
                                    Node('Test1_fully', p0, e0),
                                    Node('Test0', p1, e1))

    # Check dependencies with same partition
    assert num_deps(deps, 'Test1_by_part') == 8
    for p in ['sys0:p0', 'sys0:p1']:
        for e0 in ['e0', 'e1']:
            for e1 in ['e0', 'e1']:
                assert has_edge(deps,
                                Node('Test1_by_part', p, e0),
                                Node('Test0', p, e1))

    # Check dependencies with same partition environment
    assert num_deps(deps, 'Test1_by_case') == 4
    assert num_deps(deps, 'Test1_default') == 4
    for p in ['sys0:p0', 'sys0:p1']:
        for e in ['e0', 'e1']:
            assert has_edge(deps,
                            Node('Test1_by_case', p, e),
                            Node('Test0', p, e))
            assert has_edge(deps,
                            Node('Test1_default', p, e),
                            Node('Test0', p, e))

    assert num_deps(deps, 'Test1_any') == 12
    for p0 in ['sys0:p0', 'sys0:p1']:
        for p1 in ['sys0:p0', 'sys0:p1']:
            for e0 in ['e0', 'e1']:
                for e1 in ['e0', 'e1']:
                    if (p0 == 'sys0:p0' or e1 == 'e1'):
                        assert has_edge(deps,
                                        Node('Test1_any', p0, e0),
                                        Node('Test0', p1, e1))

    assert num_deps(deps, 'Test1_all') == 2
    for p0 in ['sys0:p0', 'sys0:p1']:
        for p1 in ['sys0:p0', 'sys0:p1']:
            for e0 in ['e0', 'e1']:
                for e1 in ['e0', 'e1']:
                    if (p0 == 'sys0:p0' and p1 == 'sys0:p0' and e1 == 'e1'):
                        assert has_edge(deps,
                                        Node('Test1_any', p0, e0),
                                        Node('Test0', p1, e1))

    # Check custom dependencies
    assert num_deps(deps, 'Test1_custom') == 1
    assert has_edge(deps,
                    Node('Test1_custom', 'sys0:p0', 'e0'),
                    Node('Test0', 'sys0:p1', 'e1'))

    # Check dependencies of Test1_nodeps
    assert num_deps(deps, 'Test1_nodeps') == 0

    # Check in-degree of Test0

    # 4 from Test1_fully,
    # 2 from Test1_by_part,
    # 1 from Test1_by_case,
    # 2 from Test1_any,
    # 2 from Test1_all,
    # 0 from Test1_custom,
    # 1 from Test1_default
    # 0 from Test1_nodeps
    assert in_degree(deps, Node('Test0', 'sys0:p0', 'e0')) == 12

    # 4 from Test1_fully,
    # 2 from Test1_by_part,
    # 1 from Test1_by_case,
    # 2 from Test1_any,
    # 0 from Test1_all,
    # 0 from Test1_custom,
    # 1 from Test1_default
    # 0 from Test1_nodeps
    assert in_degree(deps, Node('Test0', 'sys0:p1', 'e0')) == 10

    # 4 from Test1_fully,
    # 2 from Test1_by_part,
    # 1 from Test1_by_case,
    # 4 from Test1_any,
    # 0 from Test1_all,
    # 0 from Test1_custom,
    # 1 from Test1_default
    # 0 from Test1_nodeps
    assert in_degree(deps, Node('Test0', 'sys0:p0', 'e1')) == 12

    # 4 from Test1_fully,
    # 2 from Test1_by_part,
    # 1 from Test1_by_case,
    # 4 from Test1_any,
    # 0 from Test1_all,
    # 1 from Test1_custom,
    # 1 from Test1_default
    # 0 from Test1_nodeps
    assert in_degree(deps, Node('Test0', 'sys0:p1', 'e1')) == 13

    # Pick a check to test getdep()
    check_e0 = find_case('Test1_by_part', 'e0', 'p0', cases).check
    check_e1 = find_case('Test1_by_part', 'e1', 'p0', cases).check

    with pytest.raises(DependencyError):
        check_e0.getdep('Test0', 'p0')

    # Set the current environment
    check_e0._current_environ = Environment('e0')
    check_e1._current_environ = Environment('e1')

    assert check_e0.getdep('Test0', 'e0', 'p0').name == 'Test0'
    assert check_e0.getdep('Test0', 'e1', 'p0').name == 'Test0'
    assert check_e1.getdep('Test0', 'e1', 'p0').name == 'Test0'
    with pytest.raises(DependencyError):
        check_e0.getdep('TestX_deprecated', 'e0', 'p0')

    with pytest.raises(DependencyError):
        check_e0.getdep('Test0', 'eX', 'p0')

    with pytest.raises(DependencyError):
        check_e1.getdep('Test0', 'e0', 'p1')


def test_build_deps_empty(exec_ctx):
    assert {} == dependencies.build_deps([])[0]


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
    dependencies.validate_deps(
        dependencies.build_deps(
            executors.generate_testcases([t0, t1, t2, t3, t4,
                                          t5, t6, t7, t8])
        )[0]
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
    deps, _ = dependencies.build_deps(
        executors.generate_testcases([t0, t1, t2, t3, t4,
                                      t5, t6, t7, t8])
    )

    with pytest.raises(DependencyError) as exc_info:
        dependencies.validate_deps(deps)

    assert ('t4->t2->t1->t4' in str(exc_info.value) or
            't2->t1->t4->t2' in str(exc_info.value) or
            't1->t4->t2->t1' in str(exc_info.value) or
            't1->t4->t3->t1' in str(exc_info.value) or
            't4->t3->t1->t4' in str(exc_info.value) or
            't3->t1->t4->t3' in str(exc_info.value))


def test_cyclic_deps_by_env(make_test, exec_ctx):
    t0 = make_test('t0')
    t1 = make_test('t1')
    t1.depends_on('t0', udeps.env_is('e0'))
    t0.depends_on('t1', udeps.env_is('e1'))
    deps, _ = dependencies.build_deps(
        executors.generate_testcases([t0, t1])
    )
    with pytest.raises(DependencyError) as exc_info:
        dependencies.validate_deps(deps)

    assert ('t1->t0->t1' in str(exc_info.value) or
            't0->t1->t0' in str(exc_info.value))


def test_validate_deps_empty(exec_ctx):
    dependencies.validate_deps({})


def test_skip_unresolved_deps(make_test, temp_runtime):
    #
    #       t0    t4
    #      ^  ^   ^
    #     /    \ /
    #    t1    t2
    #           ^
    #           |
    #          t3
    #

    rt = temp_runtime(fixtures.TEST_CONFIG_FILE, system='sys0:p0')
    next(rt)

    t0 = make_test('t0')
    t0.valid_systems = ['sys0:p1']
    t1 = make_test('t1')
    t2 = make_test('t2')
    t3 = make_test('t3')
    t4 = make_test('t4')
    t1.depends_on('t0')
    t2.depends_on('t0')
    t2.depends_on('t4')
    t3.depends_on('t2')
    deps, skipped_cases = dependencies.build_deps(
        executors.generate_testcases([t0, t1, t2, t3, t4])
    )
    assert len(skipped_cases) == 6

    skipped_tests = {c.check.name for c in skipped_cases}
    assert skipped_tests == {'t1', 't2', 't3'}


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


def test_prune_deps(make_test, exec_ctx):
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

    testcases_all = executors.generate_testcases([t0, t1, t2, t3, t4,
                                                  t5, t6, t7, t8])
    testcases = executors.generate_testcases([t3, t7])
    full_deps, _ = dependencies.build_deps(testcases_all)
    pruned_deps = dependencies.prune_deps(full_deps, testcases)

    # Check the connectivity
    assert len(pruned_deps) == 6*4
    for p in ['sys0:p0', 'sys0:p1']:
        for e in ['e0', 'e1']:
            node = functools.partial(Node, pname=p, ename=e)
            assert has_edge(pruned_deps, node('t3'), node('t2'))
            assert has_edge(pruned_deps, node('t3'), node('t1'))
            assert has_edge(pruned_deps, node('t2'), node('t1'))
            assert has_edge(pruned_deps, node('t1'), node('t0'))
            assert has_edge(pruned_deps, node('t7'), node('t5'))
            assert len(pruned_deps[node('t3')]) == 2
            assert len(pruned_deps[node('t2')]) == 1
            assert len(pruned_deps[node('t1')]) == 1
            assert len(pruned_deps[node('t7')]) == 1
            assert len(pruned_deps[node('t5')]) == 0
            assert len(pruned_deps[node('t0')]) == 0


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
    deps, _ = dependencies.build_deps(
        executors.generate_testcases([t0, t1, t2, t3, t4,
                                      t5, t6, t7, t8])
    )
    cases = dependencies.toposort(deps)
    assert_topological_order(cases, deps)

    # Assert the level assignment
    cases_by_level = {}
    for c in cases:
        cases_by_level.setdefault(c.level, set())
        cases_by_level[c.level].add(c.check.name)

    assert cases_by_level[0] == {'t0', 't5'}
    assert cases_by_level[1] == {'t1', 't6', 't7'}
    assert cases_by_level[2] == {'t2', 't8'}
    assert cases_by_level[3] == {'t3'}
    assert cases_by_level[4] == {'t4'}


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
    full_deps, _ = dependencies.build_deps(
        executors.generate_testcases([t0, t1, t2, t3, t4])
    )
    partial_deps, _ = dependencies.build_deps(
        executors.generate_testcases([t3, t4]), full_deps
    )
    cases = dependencies.toposort(partial_deps, is_subgraph=True)
    assert_topological_order(cases, partial_deps)

    # Assert the level assignment
    cases_by_level = {}
    for c in cases:
        cases_by_level.setdefault(c.level, set())
        cases_by_level[c.level].add(c.check.name)

    assert cases_by_level[1] == {'t3'}
    assert cases_by_level[2] == {'t4'}
