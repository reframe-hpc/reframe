#!/usr/bin/env python3

import time
import sys
import subprocess
import tempfile
import tools

system = sys.argv[1]

conn = tools.get_connection()

nobjects = 10

print(conn.get_all_buckets())

bkt_name = '%s_reframe_s3_bucket_0' % system
bkt = conn.get_bucket(bkt_name)

print(bkt.list())

test_file = tempfile.NamedTemporaryFile(dir='/tmp', delete=False)

start = time.time()

for count in range(nobjects):
    obj_name = 'obj_large_%d' % count
    print('Downloading object %s to file %s' % (obj_name, test_file.name))
    obj = bkt.new_key(obj_name)
    obj.get_contents_to_filename(test_file.name)

end = time.time()

elapsed_secs = end - start
size_bytes = pow(2, 30) * nobjects
avg_download_rate = float(size_bytes/elapsed_secs)
print('Average download rate (bytes/s): %f' % avg_download_rate)
