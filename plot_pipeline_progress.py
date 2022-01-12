import json
import matplotlib.pyplot as plt
import sys


if __name__ == '__main__':
    with open(sys.argv[1]) as fp:
        raw_data = json.load(fp)

    for state, steps in raw_data.items():
        print(state, len(steps))

    fig, ax = plt.subplots()
    steps = range(len(raw_data['startup']))
    ax.stackplot(steps, raw_data.values(), labels=raw_data.keys(), alpha=0.8)
    ax.legend(loc='upper left')
    ax.set_title('Pipeline progress')
    ax.set_xlabel('Step')
    ax.set_ylabel('Number of tasks')
    plt.show()
