#!/usr/bin/python

import time
import tools

conn = tools.get_connection()

start = time.time()
tools.delete_reframe_buckets(conn)
end = time.time()
nbuckets = 30  # 10 buckets + 10 small + 10 large objects
elapsed_secs = end - start
avg_deletion_time = float(elapsed_secs)/nbuckets
print('Average deletion time (s): %f' % avg_deletion_time)
