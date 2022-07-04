# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause


import reframe.core.builtins as builtins
import reframe.core.runtime as runtime
import reframe.utility as util

from reframe.core.decorators import TestRegistry
from reframe.core.logging import getlogger, time_function
from reframe.core.meta import make_test
from reframe.core.schedulers import Job
from reframe.frontend.executors import generate_testcases


@time_function
def getallnodes(state='all', jobs_cli_options=None):
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

        if state.casefold() != 'all':
            available_nodes = {n for n in available_nodes
                               if n.in_state(state)}
            getlogger().debug(
                f'[F] Selecting nodes in state {state!r}: '
                f'available nodes now: {len(available_nodes)}'
            )

        nodes[part.fullname] = [n.name for n in available_nodes]

    return nodes


def _rfm_pin_run_nodes(obj):
    nodelist = getattr(obj, '$nid')
    if not obj.local:
        obj.job.pin_nodes = nodelist


def _rfm_pin_build_nodes(obj):
    pin_nodes = getattr(obj, '$nid')
    if not obj.local and not obj.build_locally:
        obj.build_job.pin_nodes = pin_nodes


def make_valid_systems_hook(systems):
    '''Returns a function to be used as a hook that sets the
    valid systems.

    Since valid_systems change for each generated test, we need to
    generate different post-init hooks for each one of them.
    '''
    def _rfm_set_valid_systems(obj):
        obj.valid_systems = systems

    return _rfm_set_valid_systems


@time_function
def distribute_tests(testcases, node_map):
    '''Returns new testcases that will be parameterized to run in node of
    their partitions based on the nodemap
    '''
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
        nc = make_test(
            f'{cls.__name__}_{partition.fullname.replace(":", "_")}',
            (cls,),
            {
                'valid_systems': [partition.fullname],
                '$nid': builtins.parameter(
                    [[n] for n in node_map[partition.fullname]],
                    fmt=util.nodelist_abbrev
                )
            },
            methods=[
                builtins.run_before('run')(_rfm_pin_run_nodes),
                builtins.run_before('compile')(_rfm_pin_build_nodes),
                # We re-set the valid system in a hook to make sure that it
                # will not be overwriten by a parent post-init hook
                builtins.run_after('init')(
                    make_valid_systems_hook([partition.fullname])
                ),
            ]
        )

        # We have to set the prefix manually
        nc._rfm_custom_prefix = check.prefix
        for i in range(nc.num_variants):
            # Check if this variant should be instantiated
            vinfo = nc.get_variant_info(i, recurse=True)
            vinfo['params'].pop('$nid')
            if vinfo == variant_info:
                tmp_registry.add(nc, variant_num=i)

    new_checks = tmp_registry.instantiate_all()
    return generate_testcases(new_checks)


@time_function
def repeat_tests(testcases, num_repeats):
    '''Returns new test cases parameterized over their repetition number'''

    tmp_registry = TestRegistry()
    unique_checks = set()
    for tc in testcases:
        check = tc.check
        if check.is_fixture() or check in unique_checks:
            continue

        unique_checks.add(check)
        cls = type(check)
        variant_info = cls.get_variant_info(
            check.variant_num, recurse=True
        )
        nc = make_test(
            f'{cls.__name__}', (cls,),
            {
                '$repeat_no': builtins.parameter(range(num_repeats))
            }
        )
        nc._rfm_custom_prefix = check.prefix
        for i in range(nc.num_variants):
            # Check if this variant should be instantiated
            vinfo = nc.get_variant_info(i, recurse=True)
            vinfo['params'].pop('$repeat_no')
            if vinfo == variant_info:
                tmp_registry.add(nc, variant_num=i)

    new_checks = tmp_registry.instantiate_all()
    return generate_testcases(new_checks)
