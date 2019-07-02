#
# Test case graph functionality
#

import reframe as rfm
import reframe.utility as util
from reframe.core.exceptions import DependencyError


def build_deps(cases):
    """Build dependency graph from test cases.

    The graph is represented as an adjacency list in a Python dictionary
    holding test cases. The dependency information is also encoded inside each
    test cases.
    """

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

    graph = {}
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
    """Reduce test case graph to a test-only graph."""
    ret = {}
    for case, deps in graph.items():
        test_deps = util.OrderedSet(d.check.name for d in deps)
        try:
            ret[case.check.name] |= test_deps
        except KeyError:
            ret[case.check.name] = test_deps

    return ret


def validate_deps(graph):
    """Validate dependency graph."""

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


def _reverse_deps(graph):
    ret = {}
    for n, deps in graph.items():
        ret.setdefault(n, util.OrderedSet({}))
        for d in deps:
            try:
                ret[d] |= {n}
            except KeyError:
                ret[d] = util.OrderedSet({n})

    return ret


def toposort(graph):
    # NOTES on implementation:
    #
    # 1. This function assumes a directed acyclic graph.
    # 2. The purpose of this function is to topologically sort the test cases,
    #    not only the tests. However, since we do not allow cycles between
    #    tests in any case (even if this could be classified a
    #    pseudo-dependency), we first do a topological sort of the tests and we
    #    subsequently sort the test cases by partition and by programming
    #    environment.
    # 3. To achieve this 3-step sorting with a single sort operations, we rank
    #    the test cases by associating them with an integer key based on the
    #    result of the topological sort of the tests and by choosing an
    #    arbitrary ordering of the partitions and the programming environment.

    test_deps = _reduce_deps(graph)
    rev_deps  = _reverse_deps(test_deps)

    # We do a BFS traversal from each root
    visited = {}
    roots = set(t for t, deps in test_deps.items() if not deps)
    for r in roots:
        unvisited = util.OrderedSet([r])
        visited[r] = util.OrderedSet()
        while unvisited:
            # Next node is one whose all dependencies are already visited
            # FIXME: This makes sorting's complexity O(V^2)
            node = None
            for n in unvisited:
                if test_deps[n] <= visited[r]:
                    node = n
                    break

            # If node is None, graph has a cycle and this is a bug; this
            # function assumes acyclic graphs only
            assert node is not None

            unvisited.remove(node)
            adjacent = rev_deps[node]
            unvisited |= util.OrderedSet(
                n for n in adjacent if n not in visited
            )
            visited[r].add(node)

    # Combine all individual sequences into a single one
    ordered_tests = util.OrderedSet()
    for tests in visited.values():
        ordered_tests |= tests

    # Get all partitions and programming environments from test cases
    partitions = util.OrderedSet()
    environs = util.OrderedSet()
    for c in graph.keys():
        partitions.add(c.partition.fullname)
        environs.add(c.environ.name)

    # Rank test cases; we first need to calculate the base for the rank number
    base = max(len(partitions), len(environs)) + 1
    ranks = {}
    for i, test in enumerate(ordered_tests):
        for j, part in enumerate(partitions):
            for k, env in enumerate(environs):
                ranks[test, part, env] = i*base**2 + j*base + k

    return sorted(graph.keys(),
                  key=lambda x: ranks[x.check.name,
                                      x.partition.fullname, x.environ.name])
