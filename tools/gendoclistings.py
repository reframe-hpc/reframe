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
    ['command', 'filename', 'tags', 'filters', 'env', 'xfail']
)


def remove_nocolor_opt(s):
    return s.replace(' --nocolor', '')


def remove_system_opt(s):
    return s.replace(' --system=catalina', '')


def replace_home(s):
    return s.replace(os.getenv('HOME'), '/home/user')


def replace_user(s):
    user = osext.osuser()
    return s.replace(user, 'user')


def replace_hostname(s):
    host = socket.getfqdn()
    return s.replace(host, 'host')


DEFAULT_FILTERS = [remove_nocolor_opt, remove_system_opt,
                   replace_home, replace_user, replace_hostname]


LISTINGS = {
    'hello1': ListingInfo(
        './bin/reframe -c tutorials/basics/hello/hello1.py -r',
        'docs/listings/hello1.txt',
        {'local', 'tutorial-basics'},
        DEFAULT_FILTERS,
        env={'RFM_COLORIZE': 'n'},
        xfail=False
    ),
    'run-report': ListingInfo(
        f'cat {os.getenv("HOME")}/.reframe/reports/run-report.json',
        'docs/listings/run-report.json',
        {'local', 'tutorial-basics'},
        DEFAULT_FILTERS,
        env=None,
        xfail=False
    ),
    'hello2': ListingInfo(
        './bin/reframe -c tutorials/basics/hello/hello2.py -r',
        'docs/listings/hello2.txt',
        {'local', 'tutorial-basics'},
        DEFAULT_FILTERS,
        env={'RFM_COLORIZE': 'n'},
        xfail=True
    ),
    'hello2_catalina': ListingInfo(
        './bin/reframe -C tutorials/config/settings.py --system=catalina -c tutorials/basics/hello/hello2.py -r',
        'docs/listings/hello2_catalina.txt',
        {'local', 'tutorial-basics'},
        DEFAULT_FILTERS,
        env={'RFM_COLORIZE': 'n'},
        xfail=False
    ),
    'hellomp1': ListingInfo(
        './bin/reframe --system=catalina -c tutorials/basics/hellomp/hellomp1.py -r',
        'docs/listings/hellomp1.txt',
        {'local', 'tutorial-basics'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILE': os.path.join(os.getcwd(), 'tutorials/config/settings.py'),
            'RFM_COLORIZE': 'n'
        },
        xfail=False
    ),
    'hellomp2': ListingInfo(
        './bin/reframe --system=catalina -c tutorials/basics/hellomp/hellomp2.py -r',
        'docs/listings/hellomp2.txt',
        {'local', 'tutorial-basics'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILE': os.path.join(os.getcwd(), 'tutorials/config/settings.py'),
            'RFM_COLORIZE': 'n'
        },
        xfail=True
    ),
    'alltests_daint': ListingInfo(
        './bin/reframe -c tutorials/basics/ -R -n "HelloMultiLangTest|HelloThreadedExtended2Test|StreamWithRefTest" --performance-report -r',
        'docs/listings/alltests_daint.txt',
        {'remote', 'tutorial-basics'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILE': os.path.join(os.getcwd(), 'tutorials/config/settings.py'),
            'RFM_COLORIZE': 'n'
        },
        xfail=False
    ),
    'stream4_daint': ListingInfo(
        './bin/reframe -c tutorials/basics/stream/stream4.py -r --performance-report',
        'docs/listings/stream4_daint.txt',
        {'remote', 'tutorial-basics'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILE': os.path.join(os.getcwd(), 'tutorials/config/settings.py'),
            'RFM_COLORIZE': 'n'
        },
        xfail=False
    ),
    'osu_bench_deps': ListingInfo(
        './bin/reframe -c tutorials/deps/osu_benchmarks.py -r',
        'docs/listings/osu_bench_deps.txt',
        {'remote', 'tutorial-deps'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILE': os.path.join(os.getcwd(), 'tutorials/config/settings.py'),
            'RFM_COLORIZE': 'n'
        },
        xfail=False
    ),
    'osu_latency_list': ListingInfo(
        './bin/reframe -c tutorials/deps/osu_benchmarks.py -n OSULatencyTest -l',
        'docs/listings/osu_latency_list.txt',
        {'remote', 'tutorial-deps'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILE': os.path.join(os.getcwd(), 'tutorials/config/settings.py'),
            'RFM_COLORIZE': 'n'
        },
        xfail=False
    ),
    'osu_latency_unresolved_deps': ListingInfo(
        './bin/reframe -c tutorials/deps/osu_benchmarks.py -n OSULatencyTest --system=daint:gpu -l',
        'docs/listings/osu_latency_unresolved_deps.txt',
        {'remote', 'tutorial-deps'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILE': os.path.join(os.getcwd(), 'tutorials/config/settings.py'),
            'RFM_COLORIZE': 'n'
        },
        xfail=False
    ),
    'osu_bench_list_concretized': ListingInfo(
        './bin/reframe -c tutorials/deps/osu_benchmarks.py -lC',
        'docs/listings/osu_bench_list_concretized.txt',
        {'remote', 'tutorial-deps'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILE': os.path.join(os.getcwd(), 'tutorials/config/settings.py'),
            'RFM_COLORIZE': 'n'
        },
        xfail=False
    ),
    'osu_bench_list_concretized_gnu': ListingInfo(
        './bin/reframe -c tutorials/deps/osu_benchmarks.py -n OSULatencyTest -L -p builtin -p gnu',
        'docs/listings/osu_bench_list_concretized_gnu.txt',
        {'remote', 'tutorial-deps'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILE': os.path.join(os.getcwd(), 'tutorials/config/settings.py'),
            'RFM_COLORIZE': 'n'
        },
        xfail=False
    ),
    'param_deps_list': ListingInfo(
        './bin/reframe -c tutorials/deps/parameterized.py -l',
        'docs/listings/param_deps_list.txt',
        {'local', 'tutorial-deps'},
        DEFAULT_FILTERS,
        env=None,
        xfail=False
    ),
    'osu_bench_fixtures_list': ListingInfo(
        './bin/reframe -c tutorials/fixtures/osu_benchmarks.py -l',
        'docs/listings/osu_bench_fixtures_list.txt',
        {'remote', 'tutorial-fixtures'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILE': os.path.join(os.getcwd(), 'tutorials/config/settings.py')
        },
        xfail=False
    ),
    'osu_bandwidth_concretized_daint': ListingInfo(
        './bin/reframe -c tutorials/fixtures/osu_benchmarks.py -n osu_bandwidth_test -lC',
        'docs/listings/osu_bandwidth_concretized_daint.txt',
        {'remote', 'tutorial-fixtures'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILE': os.path.join(os.getcwd(), 'tutorials/config/settings.py'),
            'RFM_COLORIZE': 'n'
        },
        xfail=False
    ),
    'osu_bandwidth_concretized_daint_pgi': ListingInfo(
        './bin/reframe -c tutorials/fixtures/osu_benchmarks.py -n osu_bandwidth_test -lC -p pgi',
        'docs/listings/osu_bandwidth_concretized_daint_pgi.txt',
        {'remote', 'tutorial-fixtures'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILE': os.path.join(os.getcwd(), 'tutorials/config/settings.py'),
            'RFM_COLORIZE': 'n'
        },
        xfail=False
    ),
    'osu_bench_fixtures_run': ListingInfo(
        './bin/reframe -c tutorials/fixtures/osu_benchmarks.py -r',
        'docs/listings/osu_bench_fixtures_run.txt',
        {'remote', 'tutorial-fixtures'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILE': os.path.join(os.getcwd(), 'tutorials/config/settings.py'),
            'RFM_COLORIZE': 'n'
        },
        xfail=False
    )
}


runcmd = functools.partial(osext.run_command, log=False)

if __name__ == '__main__':
    try:
        choice = sys.argv[1]
    except IndexError:
        choice = 'all'

    for name, info in LISTINGS.items():
        if (choice != 'all' and choice not in info.tags and choice != name):
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
