import reframe.core.config as config

site_configuration = config.detect_config(
    exclude_feats=['row*', 'c*-*', 'group*',
                   'contbuild', 'startx', 'perf', 'cvmfs', 'gpumodedefault',
                   'gpu'],
    detect_containers=True,
    detect_devices=False,
    sched_options=['-A csstaff'],
    filename='daint_config'
)
