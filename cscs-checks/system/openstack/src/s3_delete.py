#!/usr/bin/env python3

import time
import sys
import tools

system = sys.argv[1]
username = sys.argv[2]
conn = tools.get_connection()
start = time.time()
tools.delete_reframe_buckets(conn, system, username)
end = time.time()
nobjects = 30  # 10 buckets + 10 small + 10 large objects
elapsed_secs = end - start
avg_deletion_time = elapsed_secs / nobjects
print(f'Average deletion time (s): {avg_deletion_time}')
