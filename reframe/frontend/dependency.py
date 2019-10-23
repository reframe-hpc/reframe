#
# Test case graph functionality
#

import collections
import itertools

import reframe as rfm
import reframe.utility as util
from reframe.core.exceptions import DependencyError


def build_deps(cases):
    '''Build dependency graph from test cases.

    The graph is represented as an adjacency list in a Python dictionary
    holding test cases. The dependency information is also encoded inside each
    test cases.
    '''

    # Index cases for quick access
    cases_by_part = {}
    cases_revmap = {}
    for c in cases:
        cname = c.check.name
        pname = c.partition.fullname
        ename = c.environ.name
        cases_by_part.setdefault((cname, pname), [])
        cases_revmap.setdefault((cname, pname, ename), None)
        cases_by_part[cname, pname].append(c)
        cases_revmap[cname, pname, ename] = c

    def resolve_dep(target, from_map, *args):
        errmsg = 'could not resolve dependency: %s' % target
        try:
            ret = from_map[args]
        except KeyError:
            raise DependencyError(errmsg)
        else:
            if not ret:
                raise DependencyError(errmsg)

            return ret

    # NOTE on variable names
    #
    # c stands for check or case depending on the context
    # p stands for partition
    # e stands for environment
    # t stands for target

    # We use an ordered dict here, because we need to keep the order of
    # partitions and environments
    graph = collections.OrderedDict()
    for c in cases:
        cname = c.check.name
        pname = c.partition.fullname
        ename = c.environ.name
        for dep in c.check.user_deps():
            tname, how, subdeps = dep
            if how == rfm.DEPEND_FULLY:
                c.deps.extend(resolve_dep(c, cases_by_part, tname, pname))
            elif how == rfm.DEPEND_BY_ENV:
                c.deps.append(resolve_dep(c, cases_revmap,
                                          tname, pname, ename))
            elif how == rfm.DEPEND_EXACT:
                for env, tenvs in subdeps.items():
                    if env != ename:
                        continue

                    for te in tenvs:
                        c.deps.append(resolve_dep(c, cases_revmap,
                                                  tname, pname, te))

        graph[c] = util.OrderedSet(c.deps)

    return graph


def print_deps(graph):
    for c, deps in graph.items():
        print(c, '->', deps)


def _reduce_deps(graph):
    '''Reduce test case graph to a test-only graph.'''
    ret = {}
    for case, deps in graph.items():
        test_deps = util.OrderedSet(d.check.name for d in deps)
        try:
            ret[case.check.name] |= test_deps
        except KeyError:
            ret[case.check.name] = test_deps

    return ret


def validate_deps(graph):
    '''Validate dependency graph.'''

    # Reduce test case graph to a test name only graph; this disallows
    # pseudo-dependencies as follows:
    #
    # (t0, e1) -> (t1, e1)
    # (t1, e0) -> (t0, e0)
    #
    test_graph = _reduce_deps(graph)

    # Check for cyclic dependencies in the test name graph
    visited = set()
    sources = set(test_graph.keys())
    path = []

    # Since graph may comprise multiple not connected subgraphs, we search for
    # cycles starting from all possible sources
    while sources:
        unvisited = [(sources.pop(), None)]
        while unvisited:
            node, parent = unvisited.pop()
            while path and path[-1] != parent:
                path.pop()

            adjacent = test_graph[node]
            path.append(node)
            for n in adjacent:
                if n in path:
                    cycle_str = '->'.join(path + [n])
                    raise DependencyError(
                        'found cyclic dependency between tests: ' + cycle_str)

                if n not in visited:
                    unvisited.append((n, node))

            visited.add(node)

        sources -= visited


def toposort(graph):
    test_deps = _reduce_deps(graph)
    visited = util.OrderedSet()

    def visit(node, path):
        # We assume an acyclic graph
        assert node not in path

        path.add(node)

        # Do a DFS visit of all the adjacent nodes
        for adj in test_deps[node]:
            if adj not in visited:
                visit(adj, path)

        path.pop()
        visited.add(node)

    for r in test_deps.keys():
        if r not in visited:
            visit(r, util.OrderedSet())

    # Index test cases by test name
    cases_by_name = {}
    for c in graph.keys():
        try:
            cases_by_name[c.check.name].append(c)
        except KeyError:
            cases_by_name[c.check.name] = [c]

    return list(itertools.chain(*(cases_by_name[n] for n in visited)))
