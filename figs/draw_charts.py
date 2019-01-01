import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import csv
from matplotlib.ticker import FuncFormatter

x_axis = ["4GB", "8GB", "16GB", "32GB", "64GB"]
workloads = ['S', 'P', 'A', 'B', 'C', 'E-', 'E', 'E+', 'D']


def draw_line_chart(chart_name, lines):

    fig, ax = plt.subplots()
    for line in lines:
        ax.plot(x_axis, line['data'], label=line['label'])

    ax.set(xlabel='', ylabel='Throughput',
           title=chart_name)
    ax.grid()
    ax.autoscale(enable=True, axis='x', tight=True)

    plt.legend(loc=1)
    plt.savefig(chart_name.replace(' ', '_') + '_line.pdf', bbox_inches='tight')


def draw_speedup_chart(chart_name, distributions):
    fig, ax = plt.subplots()
    i = 0
    bar_width = 0.35/len(distributions)
    index = np.arange(len(x_axis))
    for dist in distributions:
        ax.bar(index + i, dist['data'], bar_width, label=dist['label'])
        i = i + bar_width

    ax.set(xlabel='', ylabel='Speedup %',
           title=chart_name)
    ax.grid()
    plt.ylim(-100, 200)
    ax.set_xticks(index + bar_width*(len(distributions)-1)/2)
    ax.set_xticklabels(x_axis)
    ax.legend(loc=1)
    plt.savefig(chart_name.replace(' ', '_') + '_speedup.pdf', bbox_inches='tight')


def read_csv(path="./Pewee - _golden_ benchmark set - csv_for_figs.csv"):

    experiments = dict()
    for workload in workloads:
        experiments[workload] = dict()

    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if '' not in row:
                workload = row[0].split()[-1]
                run = ' '.join(row[0].split()[:-1])
                experiments[workload][run] = list(map(int, row[1:]))
    return experiments


def calculate_speedups(workload, distributions):
    result = []
    for dist in distributions:
        rocks = [v for k, v in workload.items() if
                 'Rocks' in k and dist in k]
        piwi = [v for k, v in workload.items() if
                'Piwi' in k and dist in k]
        if len(rocks) == 0:
            continue
        speedup = [int((-i/j+1)*100) if i > j else int((j/i-1)*100)
                   for i, j in zip(rocks[0], piwi[0])]

        result.append({'label': dist, 'data': speedup})
    return result


def main():
    experiments = read_csv()

    # draw line charts:
    for workload in workloads:
        draw_line_chart('Workload ' + workload,
                        [{'label': k, 'data': v}
                         for (k, v) in experiments[workload].items()])
    # draw speedups
    for workload in workloads:
        distributions = calculate_speedups(experiments[workload],
                                           ['Flurry', 'Zipf', 'Latest'])
        draw_speedup_chart('workload ' + workload, distributions)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
