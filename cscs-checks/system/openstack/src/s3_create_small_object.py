#!/usr/bin/python

import time
import sys
import tools

system = sys.argv[1]

conn = tools.get_connection()

nobjects = 10

bkt_name = '%s_reframe_s3_bucket_0' % system
bkt = conn.get_bucket(bkt_name)

start = time.time()

for count in range(nobjects):
    obj_name = 'obj_small_%d' % count
    print('Creating object %s' % obj_name)
    obj = bkt.new_key(obj_name)
    obj.set_contents_from_string('Test!')

end = time.time()

elapsed_secs = end - start
avg_creation_time = float(elapsed_secs)/nobjects
print('Average object creation time (s): %f' % avg_creation_time)
