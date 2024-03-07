# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
import os
import yaml

import reframe as rfm
import reframe.core.builtins as builtins
from reframe.core.meta import make_test

from stream_variables import build_stream, stream_test


def load_specs():
    spec_file = os.getenv('STREAM_SPEC_FILE')
    if spec_file is None:
        raise ValueError('no spec file specified')

    with open(spec_file) as fp:
        try:
            specs = yaml.safe_load(fp)
        except yaml.YAMLError as err:
            raise ValueError(f'could not parse spec file: {err}') from err

    return specs


def generate_tests(specs):
    tests = []
    for i, spec in enumerate(specs['stream_workflows']):
        num_threads = spec.pop('num_threads', None)
        thread_scaling = spec.pop('thread_scaling', None)
        test_body = {
            'stream_binary': builtins.fixture(build_stream,
                                              scope='environment',
                                              variables=spec)
        }
        methods = []
        if thread_scaling:
            def _set_num_threads(test):
                test.num_threads = test.nthr

            test_body['nthr'] = builtins.parameter(thread_scaling)
            methods.append(
                builtins.run_after('init')(_set_num_threads)
            )
        elif num_threads:
            test_body['num_threads'] = num_threads

        tests.append(make_test(
            f'stream_test_{i}', (stream_test,),
            test_body,
            methods
        ))

    return tests


# Register the tests with the framework
for t in generate_tests(load_specs()):
    rfm.simple_test(t)
