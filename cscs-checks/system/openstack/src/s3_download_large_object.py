#!/usr/bin/env python3

import time
import sys
import tempfile
import tools

system = sys.argv[1]
username = sys.argv[2]
conn = tools.get_connection()
nobjects = 10
print(conn.get_all_buckets())
bkt_name = f'{system}_{username}_reframe_s3_bucket_0'
bkt = conn.get_bucket(bkt_name)
print(f'Working in bucket: {bkt_name}')
print(f'Content of this bucket: {bkt.list()}')
test_file = tempfile.NamedTemporaryFile(dir='/tmp', delete=False)
start = time.time()

for count in range(nobjects):
    obj_name = f'obj_large_{count}'
    print(f'Downloading object {obj_name} from bucket {bkt.name} '
          f'to file {test_file.name}')
    obj = bkt.new_key(obj_name)
    obj.get_contents_to_filename(test_file.name)

end = time.time()
elapsed_secs = end - start
size_mb = 1024 * nobjects
avg_download_rate = size_mb / elapsed_secs
print(f'Average download rate (MiB/s): {avg_download_rate}')
