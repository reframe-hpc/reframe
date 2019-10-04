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
tools.wait_for_state(conn, system, username, 'create_small_object_done')

nobjects = 10

bkt_name = '%s_%s_reframe_s3_bucket_0' % (system, username)
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
size_mb = 1024 * nobjects
avg_upload_rate = float(size_mb/elapsed_secs)
print('Average upload rate (MiB/s): %f' % avg_upload_rate)

state = 'upload_large_object_done'
tools.set_state(conn, system, username, state)
