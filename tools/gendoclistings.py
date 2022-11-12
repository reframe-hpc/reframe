#!/usr/bin/env python3

import collections
import functools
import os
import socket
import sys
import reframe.utility.osext as osext


def print_usage():
    print(f'Usage: {sys.argv[0]} [all|<tag>|<listing>]')


ListingInfo = collections.namedtuple(
    'ListingInfo',
    ['command', 'tags', 'filters', 'env', 'xfail']
)


def remove_nocolor_opt(s):
    return s.replace(' --nocolor', '')


def remove_system_opt(s):
    return s.replace(' --system=tresa', '')


def replace_home(s):
    return s.replace(os.getenv('HOME'), '/home/user')


def replace_user(s):
    user = osext.osuser()
    return s.replace(user, 'user')


def replace_hostname(s):
    host = socket.gethostname()
    return s.replace(host, 'host')


DEFAULT_FILTERS = [remove_nocolor_opt, remove_system_opt,
                   replace_home, replace_user, replace_hostname]


LISTINGS = {
    'hello1': ListingInfo(
        './bin/reframe -c tutorials/basics/hello/hello1.py -r',
        {'local', 'tutorial-basics'},
        DEFAULT_FILTERS,
        env={'RFM_COLORIZE': 'n'},
        xfail=False
    ),
    'run-report': ListingInfo(
        f'cat {os.getenv("HOME")}/.reframe/reports/run-report.json',
        {'local', 'tutorial-basics'},
        DEFAULT_FILTERS,
        env=None,
        xfail=False
    ),
    'hello2': ListingInfo(
        './bin/reframe -c tutorials/basics/hello/hello2.py -r',
        {'local', 'tutorial-basics'},
        DEFAULT_FILTERS,
        env={'RFM_COLORIZE': 'n'},
        xfail=True
    ),
    'hello2_tresa': ListingInfo(
        './bin/reframe -C tutorials/config/tresa.py -c tutorials/basics/hello/hello2.py -r',   # noqa: E501
        {'local', 'tutorial-basics'},
        DEFAULT_FILTERS,
        env={'RFM_COLORIZE': 'n'},
        xfail=False
    ),
    'hellomp1': ListingInfo(
        './bin/reframe -c tutorials/basics/hellomp/hellomp1.py -r',   # noqa: E501
        {'local', 'tutorial-basics'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILES': os.path.join(os.getcwd(),
                                             'tutorials/config/tresa.py'),
            'RFM_COLORIZE': 'n'
        },
        xfail=False
    ),
    'hellomp2': ListingInfo(
        './bin/reframe -c tutorials/basics/hellomp/hellomp2.py -r',   # noqa: E501
        {'local', 'tutorial-basics'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILES': os.path.join(os.getcwd(),
                                             'tutorials/config/tresa.py'),
            'RFM_COLORIZE': 'n'
        },
        xfail=True
    ),
    'stream1': ListingInfo(
        './bin/reframe -c tutorials/basics/stream/stream1.py -r --performance-report',  # noqa: E501
        {'local', 'tutorial-basics'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILES': os.path.join(os.getcwd(),
                                             'tutorials/config/tresa.py'),
            'RFM_COLORIZE': 'n'
        },
        xfail=False
    ),
    'alltests_daint': ListingInfo(
        './bin/reframe -c tutorials/basics/ -R -n "HelloMultiLangTest|HelloThreadedExtended2Test|StreamWithRefTest" --performance-report -r',   # noqa: E501
        {'remote', 'tutorial-basics'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILES': os.path.join(os.getcwd(),
                                             'tutorials/config/daint.py'),
            'RFM_COLORIZE': 'n'
        },
        xfail=False
    ),
    'stream4_daint': ListingInfo(
        './bin/reframe -c tutorials/basics/stream/stream4.py -r --performance-report',  # noqa: E501
        {'remote', 'tutorial-basics'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILES': os.path.join(os.getcwd(),
                                             'tutorials/config/daint.py'),
            'RFM_COLORIZE': 'n'
        },
        xfail=False
    ),
    'osu_bench_deps': ListingInfo(
        './bin/reframe -c tutorials/deps/osu_benchmarks.py -r',
        {'remote', 'tutorial-deps'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILES': os.path.join(os.getcwd(),
                                             'tutorials/config/daint.py'),
            'RFM_COLORIZE': 'n'
        },
        xfail=False
    ),
    'osu_latency_list': ListingInfo(
        './bin/reframe -c tutorials/deps/osu_benchmarks.py -n OSULatencyTest -l',   # noqa: E501
        {'remote', 'tutorial-deps'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILES': os.path.join(os.getcwd(),
                                             'tutorials/config/daint.py'),
            'RFM_COLORIZE': 'n'
        },
        xfail=False
    ),
    'osu_latency_unresolved_deps': ListingInfo(
        './bin/reframe -c tutorials/deps/osu_benchmarks.py -n OSULatencyTest --system=daint:gpu -l',    # noqa: E501
        {'remote', 'tutorial-deps'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILES': os.path.join(os.getcwd(),
                                             'tutorials/config/daint.py'),
            'RFM_COLORIZE': 'n'
        },
        xfail=False
    ),
    'osu_bench_list_concretized': ListingInfo(
        './bin/reframe -c tutorials/deps/osu_benchmarks.py -lC',
        {'remote', 'tutorial-deps'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILES': os.path.join(os.getcwd(),
                                             'tutorials/config/daint.py'),
            'RFM_COLORIZE': 'n'
        },
        xfail=False
    ),
    'osu_bench_list_concretized_gnu': ListingInfo(
        './bin/reframe -c tutorials/deps/osu_benchmarks.py -n OSULatencyTest -L -p builtin -p gnu',  # noqa: E501
        {'remote', 'tutorial-deps'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILES': os.path.join(os.getcwd(),
                                             'tutorials/config/daint.py'),
            'RFM_COLORIZE': 'n'
        },
        xfail=False
    ),
    'param_deps_list': ListingInfo(
        './bin/reframe -c tutorials/deps/parameterized.py -l',
        {'local', 'tutorial-deps'},
        DEFAULT_FILTERS,
        env=None,
        xfail=False
    ),
    'osu_bench_fixtures_list': ListingInfo(
        './bin/reframe -c tutorials/fixtures/osu_benchmarks.py -l',
        {'remote', 'tutorial-fixtures'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILES': os.path.join(os.getcwd(),
                                             'tutorials/config/daint.py')
        },
        xfail=False
    ),
    'osu_bandwidth_concretized_daint': ListingInfo(
        './bin/reframe -c tutorials/fixtures/osu_benchmarks.py -n osu_bandwidth_test -lC',  # noqa: E501
        {'remote', 'tutorial-fixtures'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILES': os.path.join(os.getcwd(),
                                             'tutorials/config/daint.py'),
            'RFM_COLORIZE': 'n'
        },
        xfail=False
    ),
    'osu_bandwidth_concretized_daint_pgi': ListingInfo(
        './bin/reframe -c tutorials/fixtures/osu_benchmarks.py -n osu_bandwidth_test -lC -p pgi',   # noqa: E501
        {'remote', 'tutorial-fixtures'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILES': os.path.join(os.getcwd(),
                                             'tutorials/config/daint.py'),
            'RFM_COLORIZE': 'n'
        },
        xfail=False
    ),
    'osu_bench_fixtures_run': ListingInfo(
        './bin/reframe -c tutorials/fixtures/osu_benchmarks.py -r',
        {'remote', 'tutorial-fixtures'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILES': os.path.join(os.getcwd(),
                                             'tutorials/config/daint.py'),
            'RFM_COLORIZE': 'n'
        },
        xfail=False
    ),
    'stream_params': ListingInfo(
        './bin/reframe -c tutorials/advanced/parameterized/stream.py -l',  # noqa: E501
        {'local', 'tutorial-advanced'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILES': os.path.join(os.getcwd(),
                                             'tutorials/config/tresa.py'),
            'RFM_COLORIZE': 'n'
        },
        xfail=False
    ),
    'maketest_mixin': ListingInfo(
        './bin/reframe -c tutorials/advanced/makefiles/maketest_mixin.py -l',  # noqa: E501
        {'local', 'tutorial-advanced'},
        DEFAULT_FILTERS,
        env={
            'RFM_CONFIG_FILES': os.path.join(os.getcwd(),
                                             'tutorials/config/tresa.py'),
            'RFM_COLORIZE': 'n'
        },
        xfail=False
    ),
    'hello2_typo': ListingInfo(
        'sed -ie "s/parameter/paramter/g" tutorials/basics/hello/hello2.py && '
        './bin/reframe -c tutorials/basics/hello -R -l && '
        'mv tutorials/basics/hello/hello2.pye tutorials/basics/hello/hello2.py',    # noqa: E501
        {'local', 'tutorial-tips-n-tricks'},
        DEFAULT_FILTERS,
        env={'RFM_COLORIZE': 'n'},
        xfail=False
    ),
    'hello2_typo_stacktrace': ListingInfo(
        'sed -ie "s/parameter/paramter/g" tutorials/basics/hello/hello2.py && '
        './bin/reframe -c tutorials/basics/hello -R -l -v && '
        'mv tutorials/basics/hello/hello2.pye tutorials/basics/hello/hello2.py',    # noqa: E501
        {'local', 'tutorial-tips-n-tricks'},
        DEFAULT_FILTERS,
        env={'RFM_COLORIZE': 'n'},
        xfail=False
    ),
    'hello2_print_stdout': ListingInfo(
        'sed -ie "s/self\.stdout/sn.print(self.stdout)/g" tutorials/basics/hello/hello2.py && '  # noqa: E501
        './bin/reframe -C tutorials/config/tresa.py -c tutorials/basics/hello/hello2.py -r && '    # noqa: E501
        'mv tutorials/basics/hello/hello2.pye tutorials/basics/hello/hello2.py',    # noqa: E501
        {'local', 'tutorial-tips-n-tricks'},
        DEFAULT_FILTERS,
        env={'RFM_COLORIZE': 'n'},
        xfail=False
    ),
    'hello2_list_verbose': ListingInfo(
        './bin/reframe -C tutorials/config/tresa.py -c tutorials/basics/hello/hello2.py -l -vv',  # noqa: E501
        {'local', 'tutorial-tips-n-tricks'},
        DEFAULT_FILTERS,
        env={'RFM_COLORIZE': 'n'},
        xfail=False
    ),
    'deps_complex_run': ListingInfo(
        './bin/reframe -c unittests/resources/checks_unlisted/deps_complex.py -r',  # noqa: E501
        {'local', 'tutorial-tips-n-tricks'},
        DEFAULT_FILTERS,
        env={'RFM_COLORIZE': 'n'},
        xfail=True
    ),
    'deps_rerun_t6': ListingInfo(
        './bin/reframe -c unittests/resources/checks_unlisted/deps_complex.py --keep-stage-files -r > /dev/null || '    # noqa: E501
        './bin/reframe --restore-session --keep-stage-files -n T6 -r',
        {'local', 'tutorial-tips-n-tricks'},
        DEFAULT_FILTERS,
        env={'RFM_COLORIZE': 'n'},
        xfail=False
    ),
    'deps_run_t6': ListingInfo(
        './bin/reframe -c unittests/resources/checks_unlisted/deps_complex.py -n T6 -r',    # noqa: E501
        {'local', 'tutorial-tips-n-tricks'},
        DEFAULT_FILTERS,
        env={'RFM_COLORIZE': 'n'},
        xfail=False
    )
}


runcmd = functools.partial(osext.run_command, log=False, shell=True)

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
            print(f'{info.command} should have failed, but it did not; '
                  f'skipping...')
            continue

        # Apply filters
        output = completed.stdout
        for f in info.filters:
            output = f(output)

        # Write the listing
        filename = os.path.join('docs/listings', f'{name}.txt')
        with open(filename, 'w') as fp:
            fp.write(output)
