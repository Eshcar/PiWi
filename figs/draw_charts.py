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
myfontsize = 15

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

    latency_breakdown = {'C': {'flurry': dict(), 'zipfian': dict()},
                         'A': {'flurry': dict(), 'zipfian': dict()}}

    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0] == 'Workload C latency breakdown':
                break
            if row[0] is not '':
                workload = row[0].split()[-1]
                label = ' '.join(row[0].split()[:-1])
                if workload not in experiments:
                    continue
                experiments[workload][label] = list(map(float, [i for i in row[1:] if i is not '']))

        # read latency breakdown part
        workload = 'C'
        for row in csv_reader:
            if row[0] == 'Workload A latency breakdown':
                workload = 'A'
            if row[0] not in latency_breakdown[workload]:
                continue
            distribution = row[0]
            memory = row[1]
            latency_breakdown[workload][distribution][memory] = [float(val.strip('%').replace(',','')) for val in row[2:]]

    return {'experiments': experiments, 'latency': latency_breakdown}


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


def draw_percentage_breakdown(chart_name, latency):
    MUNKS_INDEX = 2
    RC_INDEX = 6
    WB_INDEX = 11
    KS_INDEX = 15

    munks_percentage_flurry = [v[MUNKS_INDEX]
                               for (k, v) in latency['flurry'].items()]
    rawcache_percentage_flurry = [v[RC_INDEX]
                                  for (k, v) in latency['flurry'].items()]
    wb_percentage_flurry = [v[WB_INDEX]
                            for (k, v) in latency['flurry'].items()]
    keystore_percentage_flurry = [v[KS_INDEX]
                                  for (k, v) in latency['flurry'].items()]

    bar_width = 0.5
    munks_color = tableau20[0]
    rc_color = tableau20[2]
    wb_color = tableau20[6]
    ks_color = tableau20[4]
    flurry_hatch = ''
    zipf_hatch = '/'
    munks_label = 'Munk'
    rc_label = 'Rowcache'
    wb_label = 'Writebuffer'
    ks_label = 'Keystore'

    fig, ax = plt.subplots()
    
    flurry_indices = [0, 2, 4, 6, 8]
    ax.bar(flurry_indices, munks_percentage_flurry, bar_width,
           color=munks_color, edgecolor='black', hatch=flurry_hatch,
           label=munks_label)
    ax.bar(flurry_indices, rawcache_percentage_flurry, bar_width,
           bottom=munks_percentage_flurry, color=rc_color, edgecolor='black',
           hatch=flurry_hatch, label=rc_label)
    ax.bar(flurry_indices, wb_percentage_flurry, bar_width,
           bottom=[sum(x) for x in zip(munks_percentage_flurry,
                                       rawcache_percentage_flurry)],
           color=wb_color, edgecolor='black', hatch=flurry_hatch,
           label=wb_label)
    ax.bar(flurry_indices, keystore_percentage_flurry, bar_width,
           bottom=[sum(x) for x in zip(munks_percentage_flurry,
                                       rawcache_percentage_flurry,
                                       wb_percentage_flurry)],
           color=ks_color, edgecolor='black', hatch=flurry_hatch,
           label=ks_label)

    munks_percentage_zipf = [v[MUNKS_INDEX]
                             for (k, v) in latency['zipfian'].items()]
    rawcache_percentage_zipf = [v[RC_INDEX]
                                for (k, v) in latency['zipfian'].items()]
    wb_percentage_zipf = [v[WB_INDEX]
                            for (k, v) in latency['zipfian'].items()]
    keystore_percentage_zipf = [v[KS_INDEX]
                                for (k, v) in latency['zipfian'].items()]

    zipf_indices = [v+bar_width + 0.1 for v in flurry_indices]

    ax.bar(zipf_indices, munks_percentage_zipf, bar_width,
           color=munks_color, edgecolor='black', hatch=zipf_hatch)
    ax.bar(zipf_indices, rawcache_percentage_zipf, bar_width,
           bottom=munks_percentage_zipf, color=rc_color, edgecolor='black',
           hatch=zipf_hatch)
    ax.bar(zipf_indices, wb_percentage_zipf, bar_width,
           bottom=[sum(x) for x in zip(munks_percentage_zipf,
                                       rawcache_percentage_zipf)],
           color=wb_color, edgecolor='black', hatch=zipf_hatch)
    ax.bar(zipf_indices, keystore_percentage_zipf, bar_width,
           bottom=[sum(x) for x in zip(munks_percentage_zipf,
                                       rawcache_percentage_zipf,
                                       wb_percentage_zipf)],
           color=ks_color, edgecolor='black', hatch=zipf_hatch)

    ax.legend(loc=3, fontsize=myfontsize)

    ax.set_xticks(flurry_indices+zipf_indices)
    ax.set_xticklabels(['Flurry' for i in range(5)]
                       + ['Zipf' for i in range(5)], rotation=45)

    ax.set(xlabel='', ylabel='%',
           title=chart_name)
    
    ax2 = ax.twiny()
    ax2.bar(flurry_indices, [0, 0, 0, 0, 0], bar_width)
    ax2.bar(zipf_indices, [0, 0, 0, 0, 0], bar_width)
    ax2.set_xticks([v+bar_width/2 for v in flurry_indices], False)
    ax2.set_xticklabels(x_axis)
    plt.savefig(chart_name.replace(' ', '_') + '.pdf', bbox_inches='tight')


def draw_latency_breakdown(chart_name, latency):
    MUNKS_INDEX = 3
    RC_INDEX = 7
    WB_INDEX = 12
    KS_INDEX = 16

    munks_percentage_flurry = [v[MUNKS_INDEX]
                               for (k, v) in latency['flurry'].items()]
    rawcache_percentage_flurry = [v[RC_INDEX]
                                  for (k, v) in latency['flurry'].items()]
    wb_percentage_flurry = [v[WB_INDEX]
                            for (k, v) in latency['flurry'].items()]
    keystore_percentage_flurry = [v[KS_INDEX]
                                  for (k, v) in latency['flurry'].items()]

    bar_width = 0.5
    munks_color = tableau20[0]
    rc_color = tableau20[2]
    wb_color = tableau20[6]
    ks_color = tableau20[4]
    flurry_hatch = ''
    zipf_hatch = '/'
    munks_label = 'Munk'
    rc_label = 'Rowcache'
    wb_label = 'Writebuffer'
    ks_label = 'Keystore'

    fig, ax = plt.subplots()
    
    flurry_indices = [0, 2, 4, 6, 8]
    ax.bar(flurry_indices, munks_percentage_flurry, bar_width,
           color=munks_color, edgecolor='black', hatch=flurry_hatch,
           label=munks_label)
    ax.bar(flurry_indices, rawcache_percentage_flurry, bar_width,
           bottom=munks_percentage_flurry, color=rc_color, edgecolor='black',
           hatch=flurry_hatch, label=rc_label)
    ax.bar(flurry_indices, wb_percentage_flurry, bar_width,
           bottom=[sum(x) for x in zip(munks_percentage_flurry,
                                       rawcache_percentage_flurry)],
           color=wb_color, edgecolor='black', hatch=flurry_hatch,
           label=wb_label)
    ax.bar(flurry_indices, keystore_percentage_flurry, bar_width,
           bottom=[sum(x) for x in zip(munks_percentage_flurry,
                                       rawcache_percentage_flurry,
                                       wb_percentage_flurry)],
           color=ks_color, edgecolor='black', hatch=flurry_hatch,
           label=ks_label)

    munks_percentage_zipf = [v[MUNKS_INDEX]
                             for (k, v) in latency['zipfian'].items()]
    rawcache_percentage_zipf = [v[RC_INDEX]
                                for (k, v) in latency['zipfian'].items()]
    wb_percentage_zipf = [v[WB_INDEX]
                            for (k, v) in latency['zipfian'].items()]
    keystore_percentage_zipf = [v[KS_INDEX]
                                for (k, v) in latency['zipfian'].items()]

    zipf_indices = [v+bar_width + 0.1 for v in flurry_indices]

    ax.bar(zipf_indices, munks_percentage_zipf, bar_width,
           color=munks_color, edgecolor='black', hatch=zipf_hatch)
    ax.bar(zipf_indices, rawcache_percentage_zipf, bar_width,
           bottom=munks_percentage_zipf, color=rc_color, edgecolor='black',
           hatch=zipf_hatch)
    ax.bar(zipf_indices, wb_percentage_zipf, bar_width,
           bottom=[sum(x) for x in zip(munks_percentage_zipf,
                                       rawcache_percentage_zipf)],
           color=wb_color, edgecolor='black', hatch=zipf_hatch)
    ax.bar(zipf_indices, keystore_percentage_zipf, bar_width,
           bottom=[sum(x) for x in zip(munks_percentage_zipf,
                                       rawcache_percentage_zipf,
                                       wb_percentage_zipf)],
           color=ks_color, edgecolor='black', hatch=zipf_hatch)

    ax.legend(loc=2, fontsize=myfontsize)

    ax.set_xticks(flurry_indices+zipf_indices)
    ax.set_xticklabels(['Flurry' for i in range(5)]
                       + ['Zipf' for i in range(5)], rotation=45)
    ax.set_yscale("log", nonposy='clip')
    ax.set(xlabel='', ylabel='[usec]',
           title=chart_name)

    ax2 = ax.twiny()
    ax2.bar(flurry_indices, [0.1, 0.1, 0.1, 0.1, 0.1], bar_width)
    ax2.bar(zipf_indices, [0.1, 0.1, 0.1, 0.1, 0.1], bar_width)
    ax2.set_xticks([v+bar_width/2 for v in flurry_indices], False)
    ax2.set_xticklabels(x_axis)
    plt.savefig(chart_name.replace(' ', '_')+'.pdf', bbox_inches='tight')
    
    
    
def main():
    data = read_csv()
    
    experiments = data['experiments']
    # draw line charts
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

    latency = data['latency']

    draw_percentage_breakdown('Time percentage C', latency['C'])
    draw_percentage_breakdown('Time percentage A', latency['A'])
    
    draw_latency_breakdown('Latency C', latency['C'])
    draw_latency_breakdown('Latency A', latency['A'])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

