import os
import re
import sys
import time
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


def delete_reframe_buckets(conn, system, username):
    print('Removing Reframe test buckets')
    buckets = conn.get_all_buckets()
    bkts = []
    # Remove objects
    for bkt in buckets:
        if not re.search(system, bkt.name):
            continue
        if not re.search(username, bkt.name):
            continue
        objs = [obj.name for obj in bkt.list()]
        bkts.append(bkt.name)
        print('Deleting objects from bucket %s: %s' % (bkt.name, objs))
        bkt.delete_keys(objs)
    # Remove buckets
    for bkt in bkts:
        print('Deleting bucket %s' % bkt)
        conn.delete_bucket(bkt)
