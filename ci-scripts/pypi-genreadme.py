#!/usr/bin/env python3
#
# Copyright 2016-2026 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

# This script prepares the README file for publication to PyPI. It essentially
# removes all badges, replaces the dynamic logo selection with a static logo
# for light backgrounds, and rewrites GitHub-only alert blocks (e.g.
# `> [!NOTE]`) into classic blockquotes, since PyPI's Markdown renderer does
# not support GitHub's alert syntax.
#
# It should be run before build the distribution package:
#
#   ./ci-scripts/prepare-readme.py README.md
#   uv build
#   uv publish --token <PYPI_TOKEN>

import re
import sys

ALERT_RE = re.compile(r'^>\s*\[!(\w+)\]\s*$')


def unalert(lines):
    '''Rewrite `> [!TYPE]` GitHub alert blocks as classic blockquotes.'''

    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = ALERT_RE.match(line.rstrip('\n'))
        if m and i + 1 < len(lines) and lines[i + 1].startswith('>'):
            alert_type = m.group(1).capitalize()
            next_line = lines[i + 1].removeprefix('>').strip()
            new_lines.append(f'> **{alert_type}:** {next_line}\n')
            i += 2
        else:
            new_lines.append(line)
            i += 1

    return new_lines


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

    new_contents = unalert(new_contents)
    with open(readme_file, 'w') as fp:
        fp.write(''.join(new_contents))

    return 0


if __name__ == '__main__':
    sys.exit(main())
