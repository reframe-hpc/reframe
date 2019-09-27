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
    # Remove objects/buckets
    for bkt in buckets:
        if not re.search(system, bkt.name):
            continue
        if not re.search(username, bkt.name):
            continue
        for obj in bkt.list():
            print('Deleting object %s/%s' % (bkt.name, obj.name))
            obj.delete()
        print('Deleting bucket %s' % bkt.name)
        bkt.delete()


def wait_for_state(conn, system, username, state):
    bkt_name = '%s_%s_reframe_s3' % (system, username)
    obj_name = 'state'
    while True:
        print('Waiting <%s> status' % state)
        time.sleep(1)
        if conn.lookup(bkt_name):
            bkt = conn.get_bucket(bkt_name)
            if bkt.get_key(obj_name):
                obj = bkt.get_key(obj_name)
                content = obj.get_contents_as_string(encoding='utf-8')
                if content == state:
                    break


def set_state(conn, system, username, state):
    print('Setting state to <%s>.' % state)
    bkt_name = '%s_%s_reframe_s3' % (system, username)
    obj_name = 'state'
    bkt = conn.lookup(bkt_name)
    if bkt is None:
        bkt = conn.create_bucket(bkt_name)
    obj = bkt.new_key(obj_name)
    obj.set_contents_from_string(state)
