#!/bin/bash

sudo /sbin/create-munge-key -f
cp /etc/munge/munge.key /scratch

tail -f /dev/null
