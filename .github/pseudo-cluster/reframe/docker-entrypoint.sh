#!/bin/bash

trap exit 0 INT

sudo service munge start

# Needs to be copied in the shared home directory
cp -r /usr/local/share/reframe .
cd reframe
./bootstrap.sh
pip install pytest-cov

echo "Running unittests with backend scheduler: ${BACKEND}"

tempdir=$(mktemp -d -p /scratch)
TMPDIR=$tempdir ./test_reframe.py --cov=reframe --cov-report=xml \
    --rfm-user-config=ci-scripts/configs/ci-cluster.py \
    --rfm-user-system=pseudo-cluster:compute-${BACKEND:-squeue}
