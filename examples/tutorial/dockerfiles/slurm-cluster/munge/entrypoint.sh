#!/bin/bash

sudo /sbin/mungekey -c -f
cp /etc/munge/munge.key /scratch

tail -f /dev/null
