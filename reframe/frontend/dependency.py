#
# Test case graph functionality
#

import collections

import reframe as rfm
from reframe.core.exceptions import DependencyError


def build_deps(cases):
    """Build dependency graph from test cases.

    The dependency information is encoded inside the test cases. Each test case
    points internally to a list of its dependencies, that can be retrieved
    through its ``deps`` attribute. This function updates the test cases by
    setting up their dependencies and returns a dictionary that essentially
    indexes the test cases for easy access. In fact, the returned dictionary is
    an adjacency list representation of the graph, where the list of
    dependencies is encoded inside each test case.
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
        graph[c] = c
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

    return graph


def print_deps(graph):
    for c, deps in graph.items():
        print(c, '->', deps)


def validate_deps(graph):
    """Validate dependency graph."""
