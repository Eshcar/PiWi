import numpy as np
import matplotlib.pyplot as plt
import csv


x_axis = ["4GB", "8GB", "16GB", "32GB", "64GB"]
workloads = ['S', 'P', 'A', 'B', 'C', 'E-', 'E', 'E+', 'D']

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
zip_linecolor = tableau20[2]
latest_linecolor = tableau20[0]
myfontsize = 15

munks_label = 'Munk Cache'
rc_label = 'Row Cache'
wb_label = 'Log'
ks_label = 'SSTable'

line_color = {'Rocks Flurry': {'color': flurry_linecolor, 'linestyle': rocks_linestyle, 'linewidth':linewidth, 'marker': rocks_marker},
              'Rocks Zipf': {'color': zip_linecolor, 'linestyle': rocks_linestyle, 'linewidth':linewidth, 'marker': rocks_marker},
              'Rocks Latest': {'color': latest_linecolor, 'linestyle': rocks_linestyle, 'linewidth':linewidth, 'marker': rocks_marker},
              'Piwi Flurry': {'color': flurry_linecolor, 'linestyle': piwi_linestyle, 'linewidth':linewidth, 'marker': piwi_marker},
              'Piwi Zipf': {'color': zip_linecolor, 'linestyle': piwi_linestyle, 'linewidth':linewidth, 'marker': piwi_marker},
              'Piwi Latest': {'color': latest_linecolor, 'linestyle': piwi_linestyle, 'linewidth':linewidth, 'marker': piwi_marker}}


def renamings(label):

    # order is important
    rename = {'Zipfian': 'Zipf-simple', 'Zipf': 'Zipf-simple',
              'Flurry': 'Zipf-composite', 'Latest': 'Latest-simple', 'flurry': 'Zipf-composite'}
    for key in rename.keys():
        if key in label:
            return label.replace(key, rename[key]).replace('Piwi', 'YoDB').replace('Rocks', 'RocksDB')

    return label



def draw_line_chart(file_name, lines, chart_name='', yaxis='', legend=1,
                    y_upper=None, x=x_axis, x_label='Dataset size', x_bottom=None, legend_image=False):
    fig, ax = plt.subplots()
    for line in lines:
        ax.plot(x, line['data'], label=line['label'],
                color=line['style']['color'],
                linestyle=line['style']['linestyle'],
                linewidth=line['style']['linewidth'],
                marker=line['style']['marker'],
                markersize=marksize)

    ax.set_xlabel(x_label, fontsize=myfontsize)
    ax.set_ylabel(yaxis, fontsize=myfontsize)
    ax.grid()
    ax.autoscale(enable=True, axis='x', tight=True)

    y_bottom = min(0, ax.get_ylim()[0])
    y_top = ax.get_ylim()[1]
    if y_upper is not None:
        y_top = y_upper
    ax.set_ylim(y_bottom, y_top)

    for label in ax.get_yticklabels() + ax.get_xticklabels():
        label.set_fontsize(myfontsize)

    if x_bottom is not None:
        ax.set_xlim(x_bottom, ax.get_xlim()[1])

    if legend_image:
        figsize = (0.1, 0.1)
        fig_leg = plt.figure(figsize=figsize)
        ax_leg = fig_leg.add_subplot(111)
        # add the legend from the previous axes
        ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', ncol=4)
        # hide the axes frame and the x/y labels
        ax_leg.axis('off')
        fig_leg.savefig('legend.pdf', bbox_inches='tight')
    else:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc=legend, fontsize=myfontsize)

    fig.savefig(file_name.replace(' ', '_') + '_line.pdf', bbox_inches='tight')


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
    ax.set_xticklabels([renamings('Flurry') for i in range(5)]
                       + [renamings('Zipf') for i in range(5)], rotation=80, fontsize=myfontsize+5)

    ax.set_xlabel('Distribution', fontsize=myfontsize+5)
    ax.set_ylabel('% Accesses', fontsize=myfontsize+5)

    for label in ax.get_yticklabels() + ax.get_xticklabels():
        label.set_fontsize(myfontsize+5)
    
    ax2 = ax.twiny()
    ax2.set_xlabel('Dataset Size', fontsize=myfontsize+5)
    ax2.bar(flurry_indices, [0, 0, 0, 0, 0], bar_width)
    ax2.bar(zipf_indices, [0, 0, 0, 0, 0], bar_width)
    ax2.set_xticks([v+bar_width/2 for v in flurry_indices], False)
    ax2.set_xticklabels(x_axis, fontsize=myfontsize+5)
    plt.savefig(chart_name.replace(' ', '_') + '.pdf', bbox_inches='tight')


def draw_latency_breakdown(chart_name, latency):
    MUNKS_INDEX = 3
    RC_INDEX = 7
    WB_INDEX = 12
    KS_INDEX = 16

    munks_percentage_flurry = [v[MUNKS_INDEX] / 1000
                               for (k, v) in latency['flurry'].items()][2:]
    rawcache_percentage_flurry = [v[RC_INDEX] / 1000
                                  for (k, v) in latency['flurry'].items()][2:]
    wb_percentage_flurry = [v[WB_INDEX] / 1000
                            for (k, v) in latency['flurry'].items()][2:]
    keystore_percentage_flurry = [v[KS_INDEX] / 1000
                                  for (k, v) in latency['flurry'].items()][2:]

    bar_width = 0.5
    munks_color = tableau20[0]
    rc_color = tableau20[2]
    wb_color = tableau20[6]
    ks_color = tableau20[4]
    flurry_hatch = ''
    zipf_hatch = '/'


    fig, ax = plt.subplots()
    
    flurry_indices = [0, 2, 4, 7, 10][2:]
    # ax.bar(flurry_indices, munks_percentage_flurry, bar_width,
    #        color=munks_color, edgecolor='black', hatch=flurry_hatch,
    #        label=munks_label)
    # ax.bar(flurry_indices, rawcache_percentage_flurry, bar_width,
    #        bottom=munks_percentage_flurry, color=rc_color, edgecolor='black',
    #        hatch=flurry_hatch, label=rc_label)
    ax.bar(flurry_indices, wb_percentage_flurry, bar_width,
           bottom=[sum(x) for x in zip(munks_percentage_flurry,
                                       rawcache_percentage_flurry)],
           color=wb_color, edgecolor='black', hatch=flurry_hatch,
           label=wb_label)
    ax.bar([index+bar_width for index in flurry_indices], keystore_percentage_flurry, bar_width,
           color=ks_color, edgecolor='black', hatch=flurry_hatch,
           label=ks_label)

    munks_percentage_zipf = [v[MUNKS_INDEX] / 1000
                             for (k, v) in latency['zipfian'].items()][2:]
    rawcache_percentage_zipf = [v[RC_INDEX] / 1000
                                for (k, v) in latency['zipfian'].items()][2:]
    wb_percentage_zipf = [v[WB_INDEX] / 1000
                          for (k, v) in latency['zipfian'].items()][2:]
    keystore_percentage_zipf = [v[KS_INDEX] / 1000
                                for (k, v) in latency['zipfian'].items()][2:]

    zipf_indices = [v+bar_width*2 + 0.1 for v in flurry_indices]

    # ax.bar(zipf_indices, munks_percentage_zipf, bar_width,
    #        color=munks_color, edgecolor='black', hatch=zipf_hatch)
    # ax.bar(zipf_indices, rawcache_percentage_zipf, bar_width,
    #        bottom=munks_percentage_zipf, color=rc_color, edgecolor='black',
    #        hatch=zipf_hatch)
    ax.bar(zipf_indices, wb_percentage_zipf, bar_width,
           bottom=[sum(x) for x in zip(munks_percentage_zipf,
                                       rawcache_percentage_zipf)],
           color=wb_color, edgecolor='black', hatch=zipf_hatch)
    ax.bar([index + bar_width for index in zipf_indices], keystore_percentage_zipf, bar_width,
           color=ks_color, edgecolor='black', hatch=zipf_hatch)

    ax.legend(loc=2, fontsize=myfontsize)

    ax.set_xticks([v+bar_width/2 for v in flurry_indices]+[v+bar_width/2 for v in zipf_indices])
    ax.set_xticklabels([renamings('Flurry') for i in range(3)]
                       + [renamings('Zipf') for i in range(3)], rotation=80, fontsize=myfontsize+5)

    for label in ax.get_yticklabels():
        label.set_fontsize(myfontsize+5)
    
    ax.set_xlabel('Distribution', fontsize=myfontsize+5)

    ax.set_ylabel('[msec]', fontsize=myfontsize+5)
    ax2 = ax.twiny()
    ax2.set_xlabel('Dataset Size', fontsize=myfontsize+5)
    ax2.bar(flurry_indices, [0.00001, 0.00001, 0.00001, 0.00001, 0.00001][2:], bar_width)
    ax2.bar(zipf_indices, [0.00001, 0.00001, 0.00001, 0.00001, 0.00001][2:], bar_width)

    ax2.set_xticks([4.75, 7.25, 10.5], False)
    ax2.set_xticklabels(x_axis[2:], fontsize=myfontsize+5)
    plt.savefig(chart_name.replace(' ', '_')+'.pdf', bbox_inches='tight')


def read_csv(path="./Pewee - _golden_ benchmark set - csv_for_figs.csv"):

    experiments = dict()

    for workload in workloads:
        experiments[workload] = dict()

    latency_breakdown = {'C': {'flurry': dict(), 'zipfian': dict()},
                         'A': {'flurry': dict(), 'zipfian': dict()}}

    bloom_filter_partitioning = {'A_32': {'flurry': [], 'zipfian': []}}

    amplifications = {'P': dict(),
                      'A': dict(),
                      'C': dict()}

    scalability = {}

    caching = {}

    tail = {}
    
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0] == 'read_write_amplification':
                break
            if row[0] is not '':
                workload = row[0].split()[-1]
                label = ' '.join(row[0].split()[:-1])
                if workload not in experiments:
                    continue
                experiments[workload][label] = \
                    list(map(float, [i for i in row[1:] if i is not '']))

        # read/write amplification
        for row in csv_reader:
            if row[0] == 'Workload C latency breakdown':
                break
            if row[0] is '':
                continue
            workload = row[0].split()[2]
            if workload not in amplifications:
                continue
            amplifications[workload][row[0]] = list(map(float, [i for i in row[1:] if i is not '']))
        # read latency breakdown part
        workload = 'C'
        for row in csv_reader:
            if row[0] == 'bloom filter':
                break
            if row[0] == 'Workload A latency breakdown':
                workload = 'A'
            if row[0] not in latency_breakdown[workload]:
                continue
            distribution = row[0]
            memory = row[1]
            latency_breakdown[workload][distribution][memory] = \
                [float(val.strip('%').replace(',', '')) for val in row[2:] if val is not '']

        workload = 'A_32'
        for row in csv_reader:
            if row[0] == 'scalability':
                break
            distribution = row[1]
            if distribution not in bloom_filter_partitioning[workload]:
                continue

            bloom_filter_partitioning[workload][distribution].append(row)

        for row in csv_reader:
            if row[0] == 'caching':
                break
            if (row[0] != ''):
                scalability[row[0]] = list(map(float, [i for i in row[1:] if i is not '']))

        for row in csv_reader:
            if row[0] == 'tail latency':
                break
            if (row[0] == 'Munks RAM'):
                continue
            caching[row[0]] = [v/1000 for v in list(map(float, [i for i in row[1:] if i is not '']))]

        for row in csv_reader:
            if (row[0] == ''):
                continue
            tail[row[0]] =  list(map(float, [i for i in row[1:] if i is not '']))

    return {'experiments': experiments, 'latency': latency_breakdown,
            'bloom_filter_partitioning': bloom_filter_partitioning,
            'amplifications': amplifications,
            'scalability': scalability,
            'caching': caching,
            'tail':tail}


def draw_bloom_filter_partitions(chart_name, data):

    partitions = [int(val[0]) for val in data['flurry']]
    flurry_throughput = [int(val[2]) for val in data['flurry']]
    zipf_throughput = [int(val[2]) for val in data['zipfian']]

    fig, ax = plt.subplots()

    ax.plot(partitions, flurry_throughput, label='Flurry', color=flurry_linecolor, linestyle='-', linewidth=linewidth, marker='o', markersize=marksize)
    ax.plot(partitions, zipf_throughput, label='Zipfian', color=zip_linecolor, linestyle='-', linewidth=linewidth, marker='o', markersize=marksize)

    ax.set(xlabel='Partitions', ylabel='Throughput',
           title=chart_name)
    ax.grid()
    ax.autoscale(enable=True, axis='x', tight=True)

    plt.ylim(0)
    plt.legend(loc=1)
    plt.savefig(chart_name.replace(' ', '_') + '.pdf', bbox_inches='tight')


def draw_line_charts(data):

    # ylim = {'P': 380, 'A': 580, 'C': 1280, 'E-': 580, 'E': 260, 'E+': 30, 'S': None, 'B': None, 'D': None}

    experiments = data['experiments']
    # draw line charts
    for workload in workloads:
        lines = [{'label': renamings(k), 'data': v, 'style': line_color[k]}
                 for (k, v) in experiments[workload].items()]
        if len(lines) == 4:
            lines = [lines[1], lines[3], lines[0], lines[2]]
            legend_image=True
        else:
            legend_image=False
        draw_line_chart('Workload ' + workload,
                        lines,
                        yaxis='Kops', legend_image=legend_image)


def draw_speedup_charts(data):
    experiments = data['experiments']
    for workload in workloads:
        if 'writeamplification' in workload:
            continue
        distributions = calculate_speedups(experiments[workload],
                                           ['Flurry', 'Zipf', 'Latest'])
        draw_speedup_chart('workload ' + workload, distributions)


def draw_latency_charts(data):
    latency = data['latency']

    draw_percentage_breakdown('Time percentage C', latency['C'])
    draw_percentage_breakdown('Time percentage A', latency['A'])

    draw_latency_breakdown('Latency C', latency['C'])
    draw_latency_breakdown('Latency A', latency['A'])


def draw_bloom_filter_charts(data):
    draw_bloom_filter_partitions('Bloom filter', data['bloom_filter_partitioning']['A_32'])


def draw_ampl_charts(data):
    amplifications = data['amplifications']
    # P write amplification disk:
    draw_line_chart('P_write_amplification_disk',
                    [{'label': ' '.join(k.split()[0:2]), 'data': v, 'style': line_color[' '.join(k.split()[0:2])]}
                     for (k, v) in amplifications['P'].items() if 'disk' in k],
                    yaxis='Amplification', legend=3)

    # C read amplification disk:
    draw_line_chart('C_read_amplification_disk',
                    [{'label': ' '.join(k.split()[0:2]), 'data': v, 'style': line_color[' '.join(k.split()[0:2])]}
                     for (k, v) in amplifications['C'].items() if 'disk' in k],
                    yaxis='Amplification', legend=3)

    draw_line_chart('C_read_amplification_kernel',
                    [{'label': ' '.join(k.split()[0:2]), 'data': v, 'style': line_color[' '.join(k.split()[0:2])]}
                     for (k, v) in amplifications['C'].items() if 'kernel' in k],
                    yaxis='Amplification', legend=3)
 

def draw_scalability_charts(data):
    line_color = {'P Flurry': {'color': tableau20[0], 'linestyle': '-', 'linewidth':linewidth, 'marker': rocks_marker},
                  'P Zipfian': {'color': tableau20[0], 'linestyle': ':', 'linewidth':linewidth, 'marker': rocks_marker},
                  'A Flurry': {'color': tableau20[2], 'linestyle': '-', 'linewidth':linewidth, 'marker': piwi_marker},
                  'A Zipfian': {'color': tableau20[2], 'linestyle': ':', 'linewidth':linewidth, 'marker': piwi_marker},
                  'C Flurry': {'color': tableau20[4], 'linestyle': '-', 'linewidth':linewidth, 'marker': piwi_marker},
                  'C Zipfian': {'color': tableau20[4], 'linestyle': ':', 'linewidth':linewidth, 'marker': piwi_marker}}
    
    lines = [{'label': renamings(k), 'data': v, 'style': line_color[k]} for (k, v) in data['scalability'].items()]
    draw_line_chart(file_name='scalability', lines=lines, chart_name='', yaxis='Kops', legend=2, y_upper=450, x=[1,2,4,8,12], x_label='Threads', x_bottom=0)


def draw_caching_effect(data):
    line_color = {'P flurry': {'color': tableau20[0], 'linestyle': '-', 'linewidth':linewidth, 'marker': rocks_marker},
                  'P zipf': {'color': tableau20[0], 'linestyle': ':', 'linewidth':linewidth, 'marker': rocks_marker},
                  'A flurry': {'color': tableau20[2], 'linestyle': '-', 'linewidth':linewidth, 'marker': piwi_marker},
                  'A zipf': {'color': tableau20[2], 'linestyle': ':', 'linewidth':linewidth, 'marker': piwi_marker},
                  'C flurry': {'color': tableau20[4], 'linestyle': '-', 'linewidth':linewidth, 'marker': piwi_marker},
                  'C zipf': {'color': tableau20[4], 'linestyle': ':', 'linewidth':linewidth, 'marker': piwi_marker}}
    lines = [{'label': renamings(k), 'data': v, 'style': line_color[k]} for (k, v) in data['caching'].items() if 'flurry' in k]
    draw_line_chart(file_name='cache', lines=lines, chart_name='', yaxis='Kops', legend=3, y_upper=None, x=[0,2,4,6,8,10], x_label='Munk cache GB', x_bottom=0)


def draw_95(data):

    line_color = {'Rocks 95% Get': {'color': tableau20[0], 'linestyle': ':', 'linewidth':linewidth, 'marker': rocks_marker},
                  'Rocks 95% Put': {'color': tableau20[2], 'linestyle': ':', 'linewidth':linewidth, 'marker': rocks_marker},
                  'Piwi 95% Get': {'color': tableau20[0], 'linestyle': '-', 'linewidth':linewidth, 'marker': piwi_marker},
                  'Piwi 95% Put': {'color': tableau20[2], 'linestyle': '-', 'linewidth':linewidth, 'marker': piwi_marker}}

    lines = [{'label': renamings(k), 'data': v, 'style': line_color[k]} for (k, v) in data['tail'].items()]

    draw_line_chart(file_name='tail', lines=[lines[0], lines[2], lines[1], lines[3]], chart_name='', yaxis='[msec]', legend=2, y_upper=0.4, x=[4,8,16,32,64], x_bottom=0)    
    
def main():
    data = read_csv()

    # draw_line_charts(data)
    # draw_speedup_charts(data)
    # draw_latency_charts(data)
    # draw_bloom_filter_charts(data)
    # draw_ampl_charts(data)
    # draw_scalability_charts(data)
    # draw_caching_effect(data)
    draw_95(data)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

