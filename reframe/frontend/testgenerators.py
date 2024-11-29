# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause


import reframe.core.builtins as builtins
import reframe.core.runtime as runtime
import reframe.utility as util

from reframe.core.decorators import TestRegistry
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

        available_nodes = filter_nodes_by_state(available_nodes, state)
        nodes[part.fullname] = [n.name for n in available_nodes]

    return nodes


def _generate_tests(testcases, gen_fn):
    tmp_registry = TestRegistry()

    # We don't want to register the same check for every environment
    # per partition
    check_part_combs = set()
    for tc in testcases:
        check, partition, _ = tc
        candidate_comb = (check.unique_name, partition.fullname)
        if check.is_fixture() or candidate_comb in check_part_combs:
            continue

        check_part_combs.add(candidate_comb)
        cls = type(check)
        variant_info = cls.get_variant_info(
            check.variant_num, recurse=True
        )
        nc, params = gen_fn(tc)
        nc._rfm_custom_prefix = check.prefix
        for i in range(nc.num_variants):
            # Check if this variant should be instantiated
            vinfo = nc.get_variant_info(i, recurse=True)
            for p in params:
                vinfo['params'].pop(p)

            if vinfo == variant_info:
                tmp_registry.add(nc, variant_num=i)

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

    return _generate_tests(testcases, _make_dist_test)


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

    return _generate_tests(testcases, _make_repeat_test)


@time_function
def parameterize_tests(testcases, paramvars):
    '''Returns new test cases parameterized over specific variables.'''

    def _make_parameterized_test(testcase):
        check = testcase.check
        cls = type(check)

        # Check that all the requested variables exist
        body = {}
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

            if var not in cls.var_space:
                getlogger().warning(
                    f'variable {var!r} not defined for test '
                    f'{check.display_name!r}; ignoring parameterization'
                )
                continue

            body[f'.{var}'] = builtins.parameter(values, loggable=False)

        def _set_vars(self):
            for var in body.keys():
                setattr(self, var[1:],
                        make_convertible(getattr(self, f'{var}')))

        return make_test(
            cls.__name__, (cls,),
            body=body,
            methods=[builtins.run_after('init')(_set_vars)]
        ), body.keys()

    return _generate_tests(testcases, _make_parameterized_test)
