#!/usr/bin/env python3

import collections
import functools
import os
import re
import socket
import sys
import reframe.utility.osext as osext


def print_usage():
    print(f'Usage: {sys.argv[0]} [local|remote|all|<listing>]')


ListingInfo = collections.namedtuple(
    'ListingInfo',
    ['command', 'filename', 'tags', 'filters', 'env', 'xfail'],
    defaults=[None, None, False]
)


def remove_nocolor_opt(s):
    return s.replace(' --nocolor', '')


def remove_system_opt(s):
    return s.replace(' --system=catalina', '')


def replace_paths(s):
    cwd = os.getcwd()
    return s.replace(cwd, '/path/to/reframe').replace(os.getenv('HOME'), '/home/user')


def replace_user(s):
    user = osext.osuser()
    return s.replace(user, 'user')


def replace_hostname(s):
    host = socket.getfqdn()
    return s.replace(host, 'host')


DEFAULT_FILTERS = [remove_nocolor_opt, remove_system_opt,
                   replace_paths, replace_user, replace_hostname]


LISTINGS = {
    'hello1': ListingInfo(
        './bin/reframe --nocolor -c tutorials/basics/hello/hello1.py -r',
        'docs/listings/hello1.txt',
        {'local'},
        DEFAULT_FILTERS
    ),
    'run-report': ListingInfo(
        f'cat {os.getenv("HOME")}/.reframe/reports/run-report.json',
        'docs/listings/run-report.json',
        {'local'},
        DEFAULT_FILTERS
    ),
    'hello2': ListingInfo(
        './bin/reframe --nocolor -c tutorials/basics/hello/hello2.py -r',
        'docs/listings/hello2.txt',
        {'local'},
        DEFAULT_FILTERS,
        xfail=True
    ),
    'hello2_catalina': ListingInfo(
        './bin/reframe -C tutorials/config/settings.py --system=catalina --nocolor -c tutorials/basics/hello/hello2.py -r',
        'docs/listings/hello2_catalina.txt',
        {'local'},
        DEFAULT_FILTERS
    ),
    'hellomp1': ListingInfo(
        './bin/reframe --system=catalina --nocolor -c tutorials/basics/hellomp/hellomp1.py -r',
        'docs/listings/hellomp1.txt',
        {'local'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILE': os.path.join(os.getcwd(), 'tutorials/config/settings.py')
        }
    ),
    'hellomp2': ListingInfo(
        './bin/reframe --system=catalina --nocolor -c tutorials/basics/hellomp/hellomp2.py -r',
        'docs/listings/hellomp2.txt',
        {'local'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILE': os.path.join(os.getcwd(), 'tutorials/config/settings.py')
        },
        xfail=True
    )
}


runcmd = functools.partial(osext.run_command, log=False)

if __name__ == '__main__':
    try:
        choice = sys.argv[1]
    except IndexError:
        choice = 'all'

    for name, info in LISTINGS.items():
        if choice != 'all' and choice != name:
            continue

        print(f'Generating listing {name}...')

        # Set up the environment
        if info.env:
            for k, v in info.env.items():
                os.environ[k] = v

        completed = runcmd(info.command, check=not info.xfail)
        if info.xfail and completed.returncode == 0:
            print(f'{info.command} should have failed, but it did not; skipping...')
            continue

        # Apply filters
        output = completed.stdout
        for f in info.filters:
            output = f(output)

        # Write the listing
        with open(info.filename, 'w') as fp:
            fp.write(output)
