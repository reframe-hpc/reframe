#!/bin/bash

trap -- '' TERM
sleep 30 &
echo $!
wait
