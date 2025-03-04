#!/bin/bash

service munge start
service slurmctld start

tail -f /dev/null
