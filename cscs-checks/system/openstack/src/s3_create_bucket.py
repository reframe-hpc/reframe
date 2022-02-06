#!/usr/bin/env python3

import time
import sys
import tools

system = sys.argv[1]
username = sys.argv[2]

conn = tools.get_connection()

# Initial cleanup
tools.delete_reframe_buckets(conn, system, username)

nbuckets = 10
start = time.time()

for count in range(nbuckets):
    bkt_name = f'{system}_{username}_reframe_s3_bucket_{count}'
    print(f'Creating bucket {bkt_name}')
    conn.create_bucket(bkt_name)

end = time.time()
elapsed_secs = end - start
avg_creation_time = elapsed_secs / nbuckets
print(f'Average bucket creation time (s): {avg_creation_time}')
