#!/usr/bin/env python3
#
# Copyright 2016-2026 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

# This script prepares the README file for publication to PyPI. It essentially
# removes all badges and replaces the dynamic logo selection with a static logo
# for light backgrounds.
#
# It should be run before build the distribution package:
#
#   ./ci-scripts/prepare-readme.py README.md
#   uv build
#   uv publish --token <PYPI_TOKEN>

import sys


def print_usage():
    print(f'Usage: {sys.argv[0]} <README>')


def main():
    try:
        readme_file = sys.argv[1]
    except IndexError:
        print(f'{sys.argv[0]}: too few arguments')
        print_usage()
        return 1

    new_contents = [
        '[![ReFrame Logo](https://raw.githubusercontent.com/reframe-hpc/reframe/master/docs/_static/img/reframe_logo-width400p.png)](https://github.com/reframe-hpc/reframe)<br/>\n\n'  # noqa: E501
    ]

    skip_lines = True
    with open(readme_file) as fp:
        # Remove everything up to the first section
        for line in fp:
            if line.startswith('#'):
                skip_lines = False

            if skip_lines:
                continue

            new_contents.append(line)

    with open(readme_file, 'w') as fp:
        fp.write(''.join(new_contents))

    return 0


if __name__ == '__main__':
    sys.exit(main())
