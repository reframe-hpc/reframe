#!/bin/bash
#
# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Bootstrap script for running ReFrame from source
#
# Run once before the first run.

pip3 install --upgrade pip
pip3 install -r requirements.txt --prefix=external/

if [ x"$1" == x"+docs" ]; then
    pip3 install -r docs/requirements.txt --prefix=external/
    make -C docs
fi
