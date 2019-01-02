import numpy as np
import matplotlib.pyplot as plt
import csv


x_axis = ["4GB", "8GB", "16GB", "32GB", "64GB"]
workloads = ['S', 'P', 'A', 'B', 'C', 'E-', 'E', 'E+', 'D',
             'P_writeamplification_disk', 'A_writeamplification_disk']

# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

linewidth = 5
marksize = 10
rocks_linestyle = ':'
rocks_marker = 'v'
piwi_linestyle = '-'
piwi_marker = 'o'
flurry_linecolor = tableau20[0]
zip_linecolor = tableau20[6]
latest_linecolor = tableau20[0]

line_color = {'Rocks Flurry': {'color': flurry_linecolor, 'linestyle': rocks_linestyle, 'linewidth':linewidth, 'marker': rocks_marker},
              'Rocks Zipf': {'color': zip_linecolor, 'linestyle': rocks_linestyle, 'linewidth':linewidth, 'marker': rocks_marker},
              'Rocks Latest': {'color': latest_linecolor, 'linestyle': rocks_linestyle, 'linewidth':linewidth, 'marker': rocks_marker},
              'Piwi Flurry': {'color': flurry_linecolor, 'linestyle': piwi_linestyle, 'linewidth':linewidth, 'marker': piwi_marker},
              'Piwi Zipf': {'color': zip_linecolor, 'linestyle': piwi_linestyle, 'linewidth':linewidth, 'marker': piwi_marker},
              'Piwi Latest': {'color': latest_linecolor, 'linestyle': piwi_linestyle, 'linewidth':linewidth, 'marker': piwi_marker}}


def draw_line_chart(chart_name, lines):
    fig, ax = plt.subplots()
    for line in lines:
        ax.plot(x_axis, line['data'], label=line['label'],
                color=line_color[line['label']]['color'],
                linestyle=line_color[line['label']]['linestyle'],
                linewidth=line_color[line['label']]['linewidth'],
                marker=line_color[line['label']]['marker'],
                markersize=marksize)

    ax.set(xlabel='', ylabel='Throughput',
           title=chart_name)
    ax.grid()
    ax.autoscale(enable=True, axis='x', tight=True)
    plt.ylim(0)
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
                label = ' '.join(row[0].split()[:-1])
                if workload not in experiments:
                    continue
                experiments[workload][label] = list(map(float, row[1:]))
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
        if 'writeamplification' in workload:
            continue
        distributions = calculate_speedups(experiments[workload],
                                           ['Flurry', 'Zipf', 'Latest'])
        draw_speedup_chart('workload ' + workload, distributions)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
