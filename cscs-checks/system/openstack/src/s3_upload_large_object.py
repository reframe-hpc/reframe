#!/usr/bin/env python3

import time
import sys
import subprocess
import tempfile
import tools

system = sys.argv[1]
username = sys.argv[2]
conn = tools.get_connection()
nobjects = 10
bkt_name = f'{system}_{username}_reframe_s3_bucket_0'
bkt = conn.get_bucket(bkt_name)
test_file = tempfile.NamedTemporaryFile(dir='/tmp', delete=False)
cmd = f'dd if=/dev/zero of={test_file.name} bs=1M count=1024'
p = subprocess.Popen(
    cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = p.communicate()
start = time.time()

for count in range(nobjects):
    obj_name = f'obj_large_{count}'
    print(f'Creating object {obj_name}')
    obj = bkt.new_key(obj_name)
    obj.set_contents_from_filename(test_file.name)

end = time.time()
elapsed_secs = end - start
size_mb = 1024 * nobjects
avg_upload_rate = size_mb / elapsed_secs
print(f'Average upload rate (MiB/s): {avg_upload_rate}')
