#!/usr/bin/env python3
#
# Utility script to plot ReFrame's polling rate.
#
# Usage:
#   ./plot_poll_rate.py LOGFILE
#
#   This produces a diagram of the polling rate.
#
#   ./plot_poll_rate.py LOGFILE...
#
#   This produces a histogram of the polling counts from different ReFrame
#   processes.
#
# The log files must contain `debug2` information.

import io
import re
import sys

import polars as pl
import plotly.express as px


def read_logfile(logfile):
    regex = re.compile(r"\[(\S+)\] debug2:.*sleep_time=(\S+), "
                       r"pr_desired=(\S+), pr_current=(\S+), pr_global=(\S+)")
    csv_data = ''
    with open(logfile) as fp:
        for line in fp:
            if m := regex.match(line):
                csv_data += ",".join(m.groups()) + "\n"

    if not csv_data:
        return pl.DataFrame()

    df = pl.read_csv(
        io.StringIO(csv_data),
        has_header=False,
        new_columns=['timestamp', 'sleep_time', 'pr_desired',
                     'pr_current', 'pr_global']
    ).with_columns(
        pl.col('timestamp').str.to_datetime()
    )
    return df


def plot_poll_rates(logfile):
    px.line(read_logfile(logfile),
            x='timestamp', y=['pr_desired', 'pr_current', 'pr_global']).show()


def plot_poll_histogram(logfiles):
    dataframes = []
    for filename in logfiles:
        if not (df := read_logfile(filename)).is_empty():
            dataframes.append(
                df.with_columns(pl.lit(filename).alias('filename'))
            )

    fig = px.histogram(pl.concat(dataframes).sort('timestamp'), x='timestamp',
                       color='filename', nbins=100)
    fig.show()
    # fig.write_image('hist.svg')


def main():
    if len(sys.argv[1:]) == 1:
        plot_poll_rates(sys.argv[1])
    else:
        plot_poll_histogram(sys.argv[1:])

    return 0


if __name__ == '__main__':
    sys.exit(main())
