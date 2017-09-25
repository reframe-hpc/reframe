from reframe.core.fields import CopyOnWriteField


class Sandbox:
    """Sandbox class for manipulating shared resources."""
    environ = CopyOnWriteField('environ')
    system  = CopyOnWriteField('system')
    check   = CopyOnWriteField('check')
