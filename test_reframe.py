#!/usr/bin/env python3

import argparse
import pytest
import sys

import unittests.fixtures as fixtures


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        add_help=False,
        usage='%(prog)s [REFRAME_OPTIONS...] [NOSE_OPTIONS...]')
    parser.add_argument('--rfm-user-config', action='store', metavar='FILE',
                        help='Config file to use for native unit tests.')
    parser.add_argument('--rfm-help', action='help',
                        help='Print this help message and exit.')

    options, rem_args = parser.parse_known_args()
    if options.rfm_user_config:
        fixtures.set_user_config(options.rfm_user_config)

    fixtures.init_runtime()

    sys.argv = [sys.argv[0], *rem_args]
    sys.exit(pytest.main())
