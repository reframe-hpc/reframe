# Copyright 2016-2023 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# module to test reframe's loader with relative imports
#
from .. import hellocheck           # noqa: F401
from ..hellocheck import HelloTest  # noqa: F401
