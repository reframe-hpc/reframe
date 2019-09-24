#!/usr/bin/env python3

import time
import sys
import subprocess
import tempfile
import tools

system = sys.argv[1]
username = sys.argv[2]

conn = tools.get_connection()

# Wait until the create small object test is done
tools.wait_for_state(conn, system, username, 'upload_large_object_done')

nobjects = 10

print(conn.get_all_buckets())

bkt_name = '%s_%s_reframe_s3_bucket_0' % (system, username)
bkt = conn.get_bucket(bkt_name)

print('Working in bucket: %s' % bkt.name)
print('Content of this bucket: %s' % bkt.list())

test_file = tempfile.NamedTemporaryFile(dir='/tmp', delete=False)

start = time.time()

for count in range(nobjects):
    obj_name = 'obj_large_%d' % count
    print('Downloading object %s from bucket %s to file %s' % (obj_name, bkt.name, test_file.name))
    obj = bkt.new_key(obj_name)
    obj.get_contents_to_filename(test_file.name)

end = time.time()

elapsed_secs = end - start
size_mb = 1024 * nobjects
avg_download_rate = float(size_mb/elapsed_secs)
print('Average download rate (MiB/s): %f' % avg_download_rate)

state = 'download_large_object_done'
tools.set_state(conn, system, username, state)
