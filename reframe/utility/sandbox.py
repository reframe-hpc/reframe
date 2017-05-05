from reframe.core.fields import CopyOnWriteField

class Sandbox(object):
    """
    Sandbox class for manipulating shared resources
    """
    environ   = CopyOnWriteField('environ')
    system    = CopyOnWriteField('system')
