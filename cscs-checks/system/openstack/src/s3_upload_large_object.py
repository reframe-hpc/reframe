#!/usr/bin/python

import time
import subprocess
import tools

conn = tools.get_connection()

nobjects = 10

bkt_name = 'reframe_s3_bucket_0'
bkt = conn.get_bucket(bkt_name)

test_file = 'testfile.txt'
cmd = 'dd if=/dev/zero of=%s bs=1M count=1024' % test_file
p = subprocess.Popen(cmd.split(),
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE)
out, err = p.communicate()

start = time.time()

for count in range(nobjects):
    obj_name = 'obj_large_%d' % count
    print('Creating object %s' % obj_name)
    obj = bkt.new_key(obj_name)
    obj.set_contents_from_filename(test_file)

end = time.time()

elapsed_secs = end - start
size_bytes = pow(2, 30) * nobjects
avg_upload_rate = float(size_bytes/elapsed_secs)
print('Average upload rate (bytes/s): %f' % avg_upload_rate)
