#!/usr/bin/env bash

for i in {1..100}
do
    echo Random: $((RANDOM%($UPPER+1-$LOWER)+$LOWER))
done
