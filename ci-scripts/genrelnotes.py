#!/usr/bin/env python3
#
# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import re
import sys
import subprocess


def usage():
    sys.stderr.write('Usage: %s PREV_RELEASE CURR_RELEASE\n' % sys.argv[0])


def extract_release_notes(git_output):
    return re.findall(r'pull request (#\d+).*\s*\[(\S+)\] (.*)', git_output)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.stderr.write('%s: too few arguments\n' % sys.argv[0])
        usage()
        sys.exit(1)

    prev_release, curr_release, *_ = sys.argv[1:]
    try:
        git_cmd = 'git log --merges %s..%s' % (prev_release, curr_release)
        completed = subprocess.run(git_cmd.split(),
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   universal_newlines=True,
                                   check=True)
    except subprocess.CalledProcessError as e:
        sys.stderr.write('%s: git command failed: %s\n' %
                         (sys.argv[0], ' '.join(e.cmd)))
        sys.stderr.write(e.stdout)
        sys.exit(1)

    tag_mapping = {
        'feat': '## New features',
        'enhancement': '## Enhancements',
        'doc': '## Enhancements',
        'bugfix': '## Bug fixes',
        'testlib': '## Test library'
    }
    sections = {
        '## New features': [],
        '## Enhancements': [],
        '## Bug fixes': [],
        '## Test library': [],
        '## Other': []
    }
    for pr, tag, descr in extract_release_notes(completed.stdout):
        title_line = tag_mapping.get(tag, '## Other')
        descr_line = f'- {descr} ({pr})'
        sections[title_line].append(descr_line)

    print('# Release Notes')
    for sec_title, sec_lines in sections.items():
        if not sec_lines:
            continue

        print()
        print(sec_title)
        print()
        print('\n'.join(sec_lines))
