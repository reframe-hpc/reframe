#!/usr/bin/env python3

import time
import sys
import subprocess
import tempfile
import tools

system = sys.argv[1]

conn = tools.get_connection()

nobjects = 10

bkt_name = '%s_reframe_s3_bucket_0' % system
bkt = conn.get_bucket(bkt_name)

test_file = tempfile.NamedTemporaryFile(dir='/tmp', delete=False)
cmd = 'dd if=/dev/zero of=%s bs=1M count=1024' % test_file.name
p = subprocess.Popen(cmd.split(),
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE)
out, err = p.communicate()

start = time.time()

for count in range(nobjects):
    obj_name = 'obj_large_%d' % count
    print('Creating object %s' % obj_name)
    obj = bkt.new_key(obj_name)
    obj.set_contents_from_filename(test_file.name)

end = time.time()

elapsed_secs = end - start
size_bytes = pow(2, 30) * nobjects
avg_upload_rate = float(size_bytes/elapsed_secs)
print('Average upload rate (bytes/s): %f' % avg_upload_rate)
