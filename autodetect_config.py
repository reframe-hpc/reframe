import reframe.core.config as config

site_configuration = config.detect_config(
    exclude_feats=['colum*'],
    detect_containers=False,
    sched_options=[],
    time_limit=200,
    filename='system_config'
)
