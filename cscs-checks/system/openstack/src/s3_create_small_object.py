#!/usr/bin/env python3

import time
import sys
import tools

system = sys.argv[1]
username = sys.argv[2]
conn = tools.get_connection()
nobjects = 10
print('All buckets: ', conn.get_all_buckets())
bkt_name = f'{system}_{username}_reframe_s3_bucket_0'
bkt = conn.get_bucket(bkt_name)
start = time.time()

for count in range(nobjects):
    obj_name = 'obj_small_{count}'
    print(f'Creating object {obj_name}')
    obj = bkt.new_key(obj_name)
    obj.set_contents_from_string('Test!')

end = time.time()
elapsed_secs = end - start
avg_creation_time = elapsed_secs / nobjects
print(f'Average object creation time (s): {avg_creation_time}')
