#!/bin/bash
#
# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Bootstrap script for running ReFrame from source
#

if [ -t 1 ]; then
    BLUE='\033[0;34m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    NC='\033[0m'
fi

CMD()
{
    echo -e "${BLUE}==>${NC}" ${YELLOW}$*${NC} && $*
}

CMD_M()
{
    msg=$1
    shift && echo -e "${BLUE}==> [$msg]${NC}" ${YELLOW}$*${NC} && $*
}

usage()
{
    echo "Usage: $0 [-h] [+docs] [+pygelf]"
    echo "Bootstrap ReFrame by pulling all its dependencies"
    echo "  -P EXEC     Use EXEC as Python interpreter"
    echo "  -h          Print this help message and exit"
    echo "  --ignore-errors Ignore installation errors"
    echo "  --pip-opts  Pass additional options to pip."
    echo "  +docs       Build also the documentation"
    echo "  +pygelf     Install also the pygelf Python package"
}


while getopts "hP:-:"  opt; do
    case $opt in
        "P") python=$OPTARG ;;
        "h") usage && exit 0 ;;
	    "-")
	        case "${OPTARG}" in
                "ignore-errors") ignore_errors=1 ;;
                pip-opts)
	                PIPOPTS="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 )) ;;
		        pip-opts=*)
                    PIPOPTS=${OPTARG#*=} ;;
            esac;;
        "?") usage && exit 0 ;;
    esac
done

if [ -z $ignore_errors ]; then
    set -e
fi

shift $((OPTIND - 1))
if [ -z $python ]; then
    python=python3
fi

while [ -n "$1" ]; do
    case "$1" in
        "+docs") MAKEDOCS="true" && shift ;;
        "+pygelf") PYGELF="true" && shift ;;
        *) usage && exit 1 ;;
    esac
done

pyver=$($python -V | sed -n 's/Python \([0-9]\+\)\.\([0-9]\+\)\..*/\1.\2/p')

# We need to exit with a zero code if the Python version is the correct
# one, so we invert the comparison

if $python -c 'import sys; sys.exit(sys.version_info[:2] >= (3, 6))'; then
    echo -e "ReFrame requires Python >= 3.6 (found $($python -V 2>&1))"
    exit 1
fi

# Disable the user installation scheme which is the default for Debian and
# cannot be combined with `--target`
export PIP_USER=0

# Check if ensurepip is installed
$python -m ensurepip --version &> /dev/null
epip=$?

export PATH=$(pwd)/external/usr/bin:$PATH

# Install pip for Python 3
if [ $epip -eq 0 ]; then
    CMD $python -m ensurepip --root $(pwd)/external/ --default-pip
fi

# ensurepip installs pip in `external/usr/` whereas the `--root` option installs
# everything under `external/`. That's why we include both in the PYTHONPATH

export PYTHONPATH=$(pwd)/external:$(pwd)/external/usr/lib/python$pyver/site-packages:$PYTHONPATH

CMD $python -m pip install --no-cache-dir -q --upgrade pip --target=external/

if [ -n "$PYGELF" ]; then
    tmp_requirements=$(mktemp)
    sed -e 's/^#+pygelf%//g' requirements.txt > $tmp_requirements
    CMD_M +pygelf $python -m pip install --no-cache-dir -q -r $tmp_requirements --target=external/ --upgrade $PIPOPTS && rm $tmp_requirements
else
    CMD $python -m pip install --no-cache-dir -q -r requirements.txt --target=external/ --upgrade $PIPOPTS
fi

if [ -n "$MAKEDOCS" ]; then
    CMD_M +docs $python -m pip install --no-cache-dir -q -r docs/requirements.txt --target=external/ --upgrade $PIPOPTS
    make -C docs PYTHON=$python
fi
