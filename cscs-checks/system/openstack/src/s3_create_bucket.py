#!/usr/bin/env python3

import time
import sys
import tools

system = sys.argv[1]

conn = tools.get_connection()

# Initial cleanup
tools.delete_reframe_buckets(conn, system)

nbuckets = 10
start = time.time()

for count in range(nbuckets):
    bkt_name = '%s_reframe_s3_bucket_%d' % (system, count)
    print('Creating bucket %s' % bkt_name)
    conn.create_bucket(bkt_name)

end = time.time()

elapsed_secs = end - start
avg_creation_time = float(elapsed_secs)/nbuckets
print('Average bucket creation time (s): %f' % avg_creation_time)
