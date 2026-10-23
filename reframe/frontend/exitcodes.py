# Copyright 2016-2026 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

'''Exit codes of the ReFrame command line.

The codes follow the GNU convention, where 1 is a generic error and 2 denotes
command-line misuse (this is also what :mod:`argparse` exits with on invalid
command-line arguments).

This module must remain trivial, since it is also imported by
:mod:`reframe` before the Python version check completes.
'''

#: Everything succeeded.
EXIT_SUCCESS = 0

#: Any generic error, e.g., invalid configuration, failures in pipeline stages
#: other than sanity and performance, unexpected exceptions.
EXIT_ERROR = 1

#: Command-line misuse, e.g., unknown or invalid options.
EXIT_ERROR_USAGE = 2

#: At least one test that was run failed in its sanity checking phase.
EXIT_FAILURE_SANITY = 3

#: All the failed tests failed in their performance checking phase.
EXIT_FAILURE_PERFORMANCE = 4
