#!/bin/bash

trap -- '' TERM
sleep 5 &
echo $!
wait
