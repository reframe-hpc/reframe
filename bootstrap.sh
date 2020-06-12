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

CMD()
{
    echo '==>' $* && $*
}

usage()
{
    echo "Usage: $0 [-h] [+docs]"
    echo "Bootstrap ReFrame; \
run once before invoking ReFrame for the first time"
    echo "  -h      Print this help message and exit"
    echo "  +docs   Build also the documentation"
}


while getopts "h" opt; do
    case $opt in
        "h") usage && exit 0 ;;
        "?") usage && exit 0 ;;
    esac
done

shift $((OPTIND - 1))

pyver=$(python3 -V | sed -n 's/Python \([0-9]\+\)\.\([0-9]\+\)\..*/\1.\2/p')

# Install pip for Python 3
CMD python3 -m ensurepip --root external/ --default-pip

export PATH=external/usr/bin:$PATH
export PYTHONPATH=external/usr/lib/python$pyver/site-packages:$PYTHONPATH

CMD pip install --upgrade pip --target=external/
CMD pip install -r requirements.txt --target=external/

if [ x"$1" == x"+docs" ]; then
    CMD pip install -r docs/requirements.txt --target=external/
fi
