# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

from enum import IntEnum

import reframe.core.builtins as builtins
import reframe.core.runtime as runtime
import reframe.utility as util

from reframe.core.decorators import TestRegistry
from reframe.core.exceptions import ReframeError
from reframe.core.fields import make_convertible
from reframe.core.logging import getlogger, time_function
from reframe.core.meta import make_test
from reframe.core.schedulers import Job, filter_nodes_by_state
from reframe.frontend.executors import generate_testcases


@time_function
def getallnodes(state, jobs_cli_options=None):
    rt = runtime.runtime()
    nodes = {}
    for part in rt.system.partitions:
        # This job will not be submitted, it's used only to filter
        # the nodes based on the partition configuration
        dummy_job = Job.create(part.scheduler,
                               part.launcher_type(),
                               name='placeholder-job',
                               sched_access=part.access,
                               sched_options=jobs_cli_options)

        available_nodes = part.scheduler.allnodes()
        available_nodes = part.scheduler.filternodes(dummy_job,
                                                     available_nodes)
        getlogger().debug(
            f'Total available nodes for {part.name}: {len(available_nodes)}'
        )

        available_nodes = filter_nodes_by_state(
            available_nodes,
            state,
            part.scheduler
        )
        nodes[part.fullname] = [n.name for n in available_nodes]

    return nodes


class _GenKind(IntEnum):
    BY_CHECK = 1
    BY_PARTITION = 2


def _generate_tests(testcases, gen_fn, kind: _GenKind):
    tmp_registry = TestRegistry()

    def _testcase_key(tc):
        check, partition, _ = tc
        if kind == _GenKind.BY_CHECK:
            return check.unique_name
        elif kind == _GenKind.BY_PARTITION:
            return (check.unique_name, partition.fullname)

        assert False, '[BUG] unknown _GenKind'

    def _variant_key(cls, partition):
        if kind == _GenKind.BY_CHECK:
            return cls.__name__
        if kind == _GenKind.BY_PARTITION:
            return (cls.__name__, partition)

        assert False, '[BUG] unknown _GenKind'

    def _remove_params(params, variant_info):
        for p in params:
            variant_info['params'].pop(p, None)

    # We don't want to register the same check for every environment
    # per partition
    known_testcases = set()
    registered_variants = {}
    for tc in testcases:
        check, partition, _ = tc
        tc_key = _testcase_key(tc)
        if check.is_fixture() or tc_key in known_testcases:
            continue

        known_testcases.add(tc_key)

        # We want to instantiate only the variants of the original test cases.
        # For this reason, we store the original test case variant info and
        # compare it with the newly generated one after having removed its new
        # parameters. In case of reparameterizations, we remove the redefined
        # parameter also from the original test case info. We then instantiate
        # only the test cases that have a matching variant info. This
        # technique essentially re-applies any parameter filtering that has
        # happened previously in the CLI, e.g., with `-n MyTest%param=3`
        cls = type(check)
        tc_info = cls.get_variant_info(check.variant_num, recurse=True)
        nc, params = gen_fn(tc)

        # Remove any redefined parameters
        _remove_params(params, tc_info)

        nc_key = _variant_key(nc, partition.fullname)
        registered_variants.setdefault(nc_key, set())
        nc._rfm_custom_prefix = check.prefix
        for i in range(nc.num_variants):
            nc_info = nc.get_variant_info(i, recurse=True)

            # Remove all the parameters that we have (re)defined and compare
            # it with the original info; we will only instantiate test that
            # have a matching remaining info and that we have not already
            # registered.
            _remove_params(params, nc_info)
            if nc_info == tc_info and i not in registered_variants[nc_key]:
                tmp_registry.add(nc, variant_num=i)
                registered_variants[nc_key].add(i)

    new_checks = tmp_registry.instantiate_all()
    return generate_testcases(new_checks)


@time_function
def distribute_tests(testcases, node_map):
    def _rfm_pin_run_nodes(obj):
        nodelist = getattr(obj, '.nid')
        if not obj.local:
            obj.job.pin_nodes = nodelist

    def _rfm_pin_build_nodes(obj):
        pin_nodes = getattr(obj, '.nid')
        if obj.build_job and not obj.local and not obj.build_locally:
            obj.build_job.pin_nodes = pin_nodes

    def _make_dist_test(testcase):
        check, partition, _ = testcase
        cls = type(check)

        def _rfm_set_valid_systems(obj):
            obj.valid_systems = [partition.fullname]

        return make_test(
            cls.__name__, (cls,),
            {
                'valid_systems': [partition.fullname],
                # We add a partition parameter so as to differentiate the test
                # in case another test has the same nodes in another partition
                '.part': builtins.parameter([partition.fullname],
                                            loggable=False),
                '.nid': builtins.parameter(
                    [[n] for n in node_map[partition.fullname]],
                    fmt=util.nodelist_abbrev, loggable=False
                )
            },
            methods=[
                builtins.run_before('run')(_rfm_pin_run_nodes),
                builtins.run_before('compile')(_rfm_pin_build_nodes),
                # We re-set the valid system in a hook to make sure that it
                # will not be overwritten by a parent post-init hook
                builtins.run_after('init')(_rfm_set_valid_systems),
            ]
        ), ['.part', '.nid']

    return _generate_tests(testcases, _make_dist_test, _GenKind.BY_PARTITION)


@time_function
def repeat_tests(testcases, num_repeats):
    '''Returns new test cases parameterized over their repetition number'''

    def _make_repeat_test(testcase):
        cls = type(testcase.check)
        return make_test(
            cls.__name__, (cls,),
            {
                '.repeat_no': builtins.parameter(range(num_repeats),
                                                 loggable=False)
            }
        ), ['.repeat_no']

    return _generate_tests(testcases, _make_repeat_test, _GenKind.BY_CHECK)


@time_function
def parameterize_tests(testcases, paramvars):
    '''Returns new test cases parameterized over specific variables.'''

    def _make_parameterized_test(testcase):
        check = testcase.check
        cls = type(check)

        # Check that all the requested variables exist
        body = {}
        methods = []
        for var, values in paramvars.items():
            var_parts = var.split('.')
            if len(var_parts) == 1:
                var = var_parts[0]
            elif len(var_parts) == 2:
                var_check, var = var_parts
                if var_check != cls.__name__:
                    continue
            else:
                getlogger().warning('cannot set a variable in a fixture')
                continue

            if var not in cls.var_space and var not in cls.param_space:
                getlogger().warning(
                    f'{var!r} is neither a variable nor a parameter of test '
                    f'{check.display_name!r}; ignoring parameterization'
                )
                continue

            if var in cls.var_space:
                body[f'.{var}'] = builtins.parameter(values, loggable=False)
                def _set_vars(self):
                    for var in body.keys():
                        setattr(self, var[1:],
                                make_convertible(getattr(self, f'{var}')))

                methods = [builtins.run_after('init')(_set_vars)]
            elif var in cls.param_space:
                p = cls.param_space.params[var]
                if p.type is None:
                    raise ReframeError(
                        f'cannot parameterize test {cls.__name__!r}: '
                        f'no type information associated with parameter {var!r}: '  # noqa: E501
                        'consider defining the parameter as follows:\n'
                        f'    {var} = parameter({list(p.values)}, type=<type>, '    # noqa: E501
                        f'loggable={p.is_loggable()})'
                    )

                body[var] = builtins.parameter([p.type(v) for v in values], type=p.type)
            else:
                assert 0, f'[BUG] {var} cannot be defined as both a variable and a parameter'

        return (make_test(cls.__name__, (cls,), body=body, methods=methods),
                body.keys())

    return _generate_tests(testcases, _make_parameterized_test, _GenKind.BY_CHECK)
