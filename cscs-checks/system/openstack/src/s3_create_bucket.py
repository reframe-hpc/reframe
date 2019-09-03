#!/usr/bin/python

import time
import tools

conn = tools.get_connection()

# Initial cleanup
tools.delete_reframe_buckets(conn)

nbuckets = 10
start = time.time()

for count in range(nbuckets):
    bkt_name = 'reframe_s3_bucket_%d' % count
    print('Creating bucket %s' % bkt_name)
    conn.create_bucket(bkt_name)

end = time.time()

elapsed_secs = end - start
avg_creation_time = float(elapsed_secs)/nbuckets
print('Average bucket creation time (s): %f' % avg_creation_time)
