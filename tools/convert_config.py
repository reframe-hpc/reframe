#!/usr/bin/env python3
#
# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import reframe.core.config as config    # noqa: F401, F403


if __name__ == '__main__':
    try:
        old_config = sys.argv[1]
    except IndexError:
        print(f'{sys.argv[0]}: too few arguments', file=sys.stderr)
        print(f'Usage: {sys.argv[0]} OLD_CONFIG_FILE [NEW_CONFIG_FILE]',
              file=sys.stderr)
        sys.exit(1)

    try:
        new_config = sys.argv[2]
    except IndexError:
        new_config = None

    try:
        new_config = config.convert_old_config(old_config, new_config)
    except Exception as e:
        print(f'{sys.argv[0]}: could not convert file: {e}',
              file=sys.stderr)
        sys.exit(1)

    print(
        f"Conversion successful! "
        f"The converted file can be found at '{new_config}'."
    )
