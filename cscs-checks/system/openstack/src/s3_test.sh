#!/bin/sh

virtualenv s3virtenv
source s3virtenv/bin/activate
pip install boto

which python

python $1
