#!/bin/sh

python3 -m venv s3virtenv
source s3virtenv/bin/activate
pip install boto

which python

python $1 $2 $3

deactivate
/bin/rm -fr s3virtenv
