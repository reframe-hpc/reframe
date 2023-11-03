#!/usr/bin/env python3
#
# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import platform
import sys

prefix = os.path.abspath(os.path.dirname(__file__))
external = os.path.join(prefix, 'external', platform.machine())
sys.path = [prefix, external] + sys.path

import argparse                         # noqa: F401, F403
import pytest                           # noqa: F401, F403
import unittests.utility as test_util   # noqa: F401, F403


if __name__ == '__main__':
    # Unset any ReFrame environment variable; unit tests must start in a clean
    # environment
    for var in list(os.environ.keys()):
        if var.startswith('RFM_') and var != 'RFM_INSTALL_PREFIX':
            del os.environ[var]

    parser = argparse.ArgumentParser(
        add_help=False,
        usage='%(prog)s [REFRAME_OPTIONS...] [NOSE_OPTIONS...]')
    parser.add_argument(
        '--rfm-user-config', action='store', metavar='FILE',
        help='Config file to use for native unit tests.'
    )
    parser.add_argument(
        '--rfm-user-system', action='store', metavar='NAME',
        help="Specific system to use from user's configuration"
    )
    parser.add_argument(
        '--rfm-help', action='help', help='Print this help message and exit.'
    )
    options, rem_args = parser.parse_known_args()

    user_config = options.rfm_user_config
    if user_config is not None:
        user_config = os.path.abspath(user_config)

    test_util.USER_CONFIG_FILE = user_config
    test_util.USER_SYSTEM = options.rfm_user_system
    test_util.init_runtime()

    # If no positional argument is specified, use the `unittests` directory,
    # so as to avoid any automatic discovery of random unit tests from the
    # external dependencies.
    if all(arg.startswith('-') for arg in rem_args):
        rem_args.append('unittests')

    sys.argv = [sys.argv[0], *rem_args]
    sys.exit(pytest.main())
