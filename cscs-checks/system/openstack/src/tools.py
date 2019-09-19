import os
import re
import sys
import boto.s3.connection


def get_s3_credentials():

    home = os.environ['HOME']
    credentials_file = "%s/.reframe_openstack" % home
    f = open(credentials_file, 'r')
    for line in f:
        linesplit = line.split('=')
        if linesplit[0] == 's3_access_key':
            access_key = linesplit[1].rstrip()
        if linesplit[0] == 's3_secret_key':
            secret_key = linesplit[1].rstrip()
    return (access_key, secret_key)


def get_connection():
    (access_key, secret_key) = get_s3_credentials()
    conn = boto.connect_s3(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        host='object.cscs.ch',
        port=443,
        calling_format=boto.s3.connection.OrdinaryCallingFormat()
    )
    return conn


def delete_reframe_buckets(conn, system):
    print('Removing Reframe test buckets')
    buckets = conn.get_all_buckets()
    print('All buckets: ', buckets)
    for bkt in buckets:
        if re.search(system, bkt.name):
            print('Removing bucket %s and its objects' % bkt.name)
            for obj in bkt.list():
                print('Removing object %s' % obj.name)
                obj.delete()
            #for debugging, listing again the bucket content after
            # having deleted its objects
            for obj in bkt.list():
                print('This object is still in the bucket: %s' % obj.name)
            print('Deleting bucket %s.' % bkt.name)
            conn.delete_bucket(bkt.name)
