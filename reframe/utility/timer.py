def seconds_to_hms(seconds):
    '''Convert time in seconds to a tuple (hour, minutes, seconds)'''
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s
