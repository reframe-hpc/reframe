#!/usr/bin/env python3

import sys
import os
if sys.stdin.isatty():
    print("stdin is a tty")
    # do some tty-only thing
    print(os.tcgetpgrp(sys.stdin.fileno()))
else:
    print("stdin is not a tty")
