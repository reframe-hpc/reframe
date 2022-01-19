import json
import matplotlib.pyplot as plt
import os
import sys


if __name__ == '__main__':
    with open(sys.argv[1]) as fp:
        raw_data = json.load(fp)

    for state, steps in raw_data.items():
        print(state, len(steps))

    try:
        mode = sys.argv[2]
        if mode not in ('steps', 'time'):
            print(f'unknown mode: {mode}')
            sys.exit(1)
    except IndexError:
        mode = 'steps'

    if mode == 'steps':
        x_label = '# Steps'
        x_values = range(len(raw_data['startup']))
    else:
        x_label = 'Time (s)'
        x_values = [x[1] for x in raw_data['startup']]

    y_values = []
    for x in raw_data.values():
        step_values = [s[0] for s in x]
        y_values.append(step_values)

    fig, ax = plt.subplots()
    ax.stackplot(x_values, y_values, labels=raw_data.keys(), alpha=1)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('Pipeline progress')
    ax.set_xlabel(x_label)
    ax.set_ylabel('Number of tasks')
    figname = os.path.splitext(sys.argv[1])[0] + '_' + mode + '.png'
    plt.savefig(figname, bbox_inches='tight')
    plt.show()
