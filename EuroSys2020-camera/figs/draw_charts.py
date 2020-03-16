import numpy as np
import matplotlib.pyplot as plt
import csv

x_axis = ["4GB", "8GB", "16GB", "32GB", "64GB", "128GB", "256GB"]
workloads = ['S', 'P', 'A', 'B', 'C', 'E-', 'E', 'E+', 'F', 'D']

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
latest_linecolor = tableau20[6]
uniform_linecolor = tableau20[4]
myfontsize = 15

munks_label = 'Munk Cache'
rc_label = 'Row Cache'
wb_label = 'Log'
ks_label = 'SSTable'

line_color = {'Rocks Flurry': {'color': flurry_linecolor, 'linestyle': rocks_linestyle, 'linewidth': linewidth,
                               'marker': rocks_marker},
              'Rocks Zipf': {'color': zip_linecolor, 'linestyle': rocks_linestyle, 'linewidth': linewidth,
                             'marker': rocks_marker},
              'Rocks Latest': {'color': latest_linecolor, 'linestyle': rocks_linestyle, 'linewidth': linewidth,
                               'marker': rocks_marker},
              'Piwi Flurry': {'color': flurry_linecolor, 'linestyle': piwi_linestyle, 'linewidth': linewidth,
                              'marker': piwi_marker},
              'Piwi Zipf': {'color': zip_linecolor, 'linestyle': piwi_linestyle, 'linewidth': linewidth,
                            'marker': piwi_marker},
              'Piwi Latest': {'color': latest_linecolor, 'linestyle': piwi_linestyle, 'linewidth': linewidth,
                              'marker': piwi_marker},
              'Rocks Uniform': {'color': uniform_linecolor, 'linestyle': rocks_linestyle, 'linewidth': linewidth,
                                'marker': rocks_marker},
              'Piwi Uniform': {'color': uniform_linecolor, 'linestyle': piwi_linestyle, 'linewidth': linewidth,
                               'marker': piwi_marker}}


def renamings(label):
    if label == 'RocksDB' or label == 'EvenDB':
        return label
    # order is important
    rename = {'Zipfian': 'Zipf-simple', 'Zipf': 'Zipf-simple', 'zipf': 'Zipf-simple',
              'Flurry': 'Zipf-composite', 'Latest': 'Latest-simple', 'flurry': 'Zipf-composite',
              'flurry2': 'Zipf-composite', '95% ': ''}
    new_label = label.replace('Piwi', 'EvenDB').replace('Rocks', 'RocksDB')
    for key in rename.keys():
        if key in label:
            return new_label.replace(key, rename[key])

    return new_label


def draw_line_chart(file_name, lines, chart_name='', yaxis='', legend=1,
                    y_upper=None, x=x_axis, x_label='Dataset size',
                    x_bottom=None, legend_image=False, fontsize=myfontsize, ncol=1, x_upper=None, bbox_to_anchor=None):
    fig, ax = plt.subplots()
    for line in lines:
        ax.plot(x, line['data'], label=line['label'],
                color=line['style']['color'],
                linestyle=line['style']['linestyle'],
                linewidth=line['style']['linewidth'],
                marker=line['style']['marker'],
                markersize=marksize)

    ax.set_xlabel(x_label, fontsize=fontsize + 5)
    ax.set_ylabel(yaxis, fontsize=fontsize + 5)
    ax.grid()
    ax.autoscale(enable=True, axis='x', tight=True)

    y_bottom = min(0, ax.get_ylim()[0])
    y_top = ax.get_ylim()[1]
    if y_upper is not None:
        y_top = y_upper
    ax.set_ylim(y_bottom, y_top)

    for label in ax.get_yticklabels() + ax.get_xticklabels():
        label.set_fontsize(fontsize)

    if x_bottom is not None:
        ax.set_xlim(x_bottom, ax.get_xlim()[1])
    if x_upper is not None:
        ax.set_xlim(ax.get_xlim()[0], x_upper)

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
        h, l = ax.get_legend_handles_labels()
        if file_name == 'Workload P':
            # omit the common lines from per-graph legends to save space
            h = h[4:]
            l = l[4:]
        ax.legend(h, l, loc=legend, fontsize=fontsize, ncol=ncol, bbox_to_anchor=bbox_to_anchor)

    fig.savefig(file_name.replace(' ', '_') + '_line.pdf', bbox_inches='tight')


def draw_speedup_chart(chart_name, distributions):
    fig, ax = plt.subplots()
    i = 0
    bar_width = 0.35 / len(distributions)
    index = np.arange(len(x_axis))
    for dist in distributions:
        ax.bar(index + i, dist['data'], bar_width, label=dist['label'])
        i = i + bar_width

    ax.set(xlabel='', ylabel='Speedup %',
           title=chart_name)
    ax.grid()
    plt.ylim(-100, 200)
    ax.set_xticks(index + bar_width * (len(distributions) - 1) / 2)
    ax.set_xticklabels(x_axis)
    ax.legend(loc=1)
    plt.savefig(chart_name.replace(' ', '_') + '_speedup.pdf', bbox_inches='tight')


def draw_munk_size_chart(file_name, bars):
    print(bars)
    # fig, ax = plt.subplots()
    # i = 0
    # bar_width = 0.35/len(distributions)
    # index = np.arange(len(x_axis))
    # for dist in distributions:
    #     ax.bar(index + i, dist['data'], bar_width, label=dist['label'])
    #     i = i + bar_width

    # ax.set(xlabel='', ylabel='Speedup %',
    #        title=chart_name)
    # ax.grid()
    # plt.ylim(-100, 200)
    # ax.set_xticks(index + bar_width*(len(distributions)-1)/2)
    # ax.set_xticklabels(x_axis)
    # ax.legend(loc=1)
    # plt.savefig(chart_name.replace(' ', '_') + '_speedup.pdf', bbox_inches='tight')


def calculate_speedups(workload, distributions):
    result = []
    for dist in distributions:
        rocks = [v for k, v in workload.items() if
                 'Rocks' in k and dist in k]
        piwi = [v for k, v in workload.items() if
                'Piwi' in k and dist in k]
        if len(rocks) == 0:
            continue
        speedup = [int((-i / j + 1) * 100) if i > j else int((j / i - 1) * 100)
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

    zipf_indices = [v + bar_width + 0.1 for v in flurry_indices]

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

    ax.legend(loc=3, fontsize=myfontsize + 5)

    ax.set_xticks(flurry_indices + zipf_indices)
    ax.set_xticklabels([renamings('Flurry') for i in range(5)]
                       + [renamings('Zipf') for i in range(5)], rotation=80, fontsize=myfontsize + 5)

    ax.set_xlabel('Distribution', fontsize=myfontsize + 10)
    ax.set_ylabel('% Accesses', fontsize=myfontsize + 10)

    for label in ax.get_yticklabels() + ax.get_xticklabels():
        label.set_fontsize(myfontsize + 5)

    ax2 = ax.twiny()
    ax2.set_xlabel('Dataset Size', fontsize=myfontsize + 10)
    ax2.bar(flurry_indices, [0, 0, 0, 0, 0], bar_width)
    ax2.bar(zipf_indices, [0, 0, 0, 0, 0], bar_width)
    ax2.set_xticks([v + bar_width / 2 for v in flurry_indices], False)
    ax2.set_xticklabels(x_axis, fontsize=myfontsize + 5)
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
    ax.bar([index + bar_width for index in flurry_indices], keystore_percentage_flurry, bar_width,
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

    zipf_indices = [v + bar_width * 2 + 0.1 for v in flurry_indices]

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

    ax.legend(loc=2, fontsize=myfontsize + 5)

    ax.set_xticks([v + bar_width / 2 for v in flurry_indices] + [v + bar_width / 2 for v in zipf_indices])
    ax.set_xticklabels([renamings('Flurry') for i in range(3)]
                       + [renamings('Zipf') for i in range(3)], rotation=80, fontsize=myfontsize + 5)

    for label in ax.get_yticklabels():
        label.set_fontsize(myfontsize + 5)

    ax.set_xlabel('Distribution', fontsize=myfontsize + 10)

    ax.set_ylabel('Latency, [ms]', fontsize=myfontsize + 10)
    ax2 = ax.twiny()
    ax2.set_xlabel('Dataset Size', fontsize=myfontsize + 10)
    ax2.bar(flurry_indices, [0.00001, 0.00001, 0.00001, 0.00001, 0.00001][2:], bar_width)
    ax2.bar(zipf_indices, [0.00001, 0.00001, 0.00001, 0.00001, 0.00001][2:], bar_width)

    ax2.set_xticks([4.75, 7.25, 10.5], False)
    ax2.set_xticklabels(x_axis[2:], fontsize=myfontsize + 5)
    plt.savefig(chart_name.replace(' ', '_') + '.pdf', bbox_inches='tight')


def draw_piwi_vs_rocks_bars(chart_name, data, xlabel, ylabel):
    fig, ax = plt.subplots()
    index = np.arange(len(data.keys()) - 1)
    bar_width = 0.35
    piwi_hatch = ''
    rocks_hatch = '/'
    import matplotlib as mpl
    org_width = mpl.rcParams['hatch.linewidth']
    mpl.rcParams['hatch.linewidth'] = 3

    data_values = list(data.values())
    ax.bar(index, [r[0] for r in data_values[1:]], bar_width, label=renamings(data_values[0][0]),
           color=flurry_linecolor, edgecolor='black', hatch=piwi_hatch)
    ax.bar(index + bar_width, [r[1] for r in data_values[1:]], bar_width, label=renamings(data_values[0][1]),
           color=flurry_linecolor, edgecolor='black', hatch=rocks_hatch)

    ax.legend(loc=2, fontsize=myfontsize + 3, ncol=2)

    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(list(data.keys())[1:], rotation=0, fontsize=myfontsize + 5)

    for label in ax.get_yticklabels():
        label.set_fontsize(myfontsize + 5)

    ax.set_xlabel(xlabel, fontsize=myfontsize + 10)
    ax.set_ylabel(ylabel, fontsize=myfontsize + 10)

    y_bottom = ax.get_ylim()[0]
    y_top = ax.get_ylim()[1]
    ax.set_ylim(y_bottom, y_top * 1.15)  # make room for the legend

    plt.savefig(chart_name.replace(' ', '_') + '.pdf', bbox_inches='tight')
    mpl.rcParams['hatch.linewidth'] = org_width


def draw_rocks_block_cache_chart(chart_name, data, xlabel, ylabel):
    fig, ax = plt.subplots()
    index = np.arange(len(data.keys()) - 1)
    bar_width = 0.15

    data_values = list(data.values())
    for i in range(len(data_values[1])):
        ax.bar(index + bar_width * i, [r[i] for r in data_values[1:]], bar_width, zorder=3,
               label=renamings(data_values[0][i]))

    ax.legend(loc=9, fontsize=myfontsize, ncol=2)

    ax.set_xticks(index + bar_width / 2)
    xticklabels = []
    dist_used = []
    label_offsets = [21, 23]
    for k in list(data.keys())[1:]:
        label = k.split(' ')[0]
        dist = k.split(' ')[1]
        if dist not in dist_used:
            label += '\n' + ' ' * label_offsets[len(dist_used)] + renamings(dist)
            dist_used.append(dist)
        xticklabels.append(label)
    ax.set_xticklabels(xticklabels, rotation=0, fontsize=myfontsize + 5)
    # ax.set_xticklabels(list(data.keys())[1:], rotation=0, fontsize=myfontsize + 5)

    for label in ax.get_yticklabels():
        label.set_fontsize(myfontsize + 5)

    ax.set_xlabel(xlabel, fontsize=myfontsize + 5)
    ax.set_ylabel(ylabel, fontsize=myfontsize + 5)
    ax.grid(axis='y', zorder=0)

    y_bottom = ax.get_ylim()[0]
    y_top = ax.get_ylim()[1]
    ax.set_ylim(y_bottom, y_top * 1.15)  # make room for the legend

    plt.savefig(chart_name.replace(' ', '_') + '.pdf', bbox_inches='tight')


def draw_timeline_chart(file_name, data, y_label='Throughput, Kops', x_label='Execution time, minutes', legend=1,
                        y_upper=None, x_bottom=None, fontsize=myfontsize, ncol=1, x_upper=None, bbox_to_anchor=None,
                        tick_sec=30):
    labels_row = list(data.values())[0]
    values_rows = list(data.values())[1:]
    lines = {}
    for label in labels_row:
        lines[label] = []
    for row in values_rows:
        for z in zip(labels_row, row):
            lines[z[0]].append(z[1] / 1000 if z[1] is not None else None)  # turn to kops
    timeline = [0] + list(data.keys())[1:]

    fig, ax = plt.subplots()
    for key, value in lines.items():
        if len(value) < len(timeline):
            value += [None] * (len(timeline) - len(value))
        style = piwi_linestyle if key == 'EvenDB' else rocks_linestyle
        ax.plot(timeline, value, label=key, linewidth=linewidth * 0.6, color=flurry_linecolor, linestyle=style)

    ax.set_xlabel(x_label, fontsize=fontsize + 5)
    ax.set_ylabel(y_label, fontsize=fontsize + 5)
    ax.grid()
    ax.autoscale(enable=True, axis='x', tight=True)

    y_top = ax.get_ylim()[1]
    if y_upper is not None:
        y_top = y_upper
    ax.set_ylim(bottom=0, top=y_top)
    ax.set_xticks(ax.get_xticks()[::tick_sec])  # tick every tick_sec seconds

    for label in ax.get_yticklabels() + ax.get_xticklabels():
        label.set_fontsize(fontsize)

    if len(ax.get_xticklabels()) > 7:
        for label in ax.get_xticklabels():
            label.set_rotation(80)

    if ax.get_yticks()[1] - ax.get_yticks()[0] == 0.25:
        ax.set_yticks(ax.get_yticks()[::2])  # skip numbers with too many decimal digits...

    if x_bottom is not None:
        ax.set_xlim(x_bottom, ax.get_xlim()[1])
    if x_upper is not None:
        ax.set_xlim(ax.get_xlim()[0], x_upper)

    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, loc=legend, fontsize=fontsize, ncol=ncol, bbox_to_anchor=bbox_to_anchor)

    fig.savefig(file_name.replace(' ', '_') + '_line.pdf', bbox_inches='tight')


def draw_space_timeline_chart(file_name, data, y_label='Disk space (GB)', x_label='Execution percent', legend=1,
                        y_upper=None, x_bottom=None, fontsize=myfontsize, ncol=1, x_upper=None, bbox_to_anchor=None,
                        tick_percent=30):
    labels_row = list(data.values())[0]
    values_rows = list(data.values())[1:]
    lines = {}
    for label in labels_row:
        lines[label] = []
    for row in values_rows:
        for z in zip(labels_row, row):
            lines[z[0]].append(z[1] / 2**30)
    timeline = [0] + list(data.keys())[1:]

    '''
    Even - solid blue
    Even log - solid gray, legend - "EvenDB Log"
    Rocks - dotted
    "Raw Data" - solid black
    Also, reorder lines in order of appearance
    '''
    ordered_lines = [None] * 4
    for key, value in lines.items():
        i = 0 if key == 'RocksDB' else 1 if key == 'EvenDB' else 2 if key == 'Input size' else 3
        ordered_lines[i] = (key, value)
    fig, ax = plt.subplots()
    for key, value in ordered_lines:
        if len(value) < len(timeline):
            value += [None] * (len(timeline) - len(value))
        style, color, label, linew = ('-', flurry_linecolor, 'EvenDB', linewidth * 0.6) if key == 'EvenDB' \
            else ('--', flurry_linecolor, 'RocksDB', linewidth * 0.6) if key == 'RocksDB' \
                else (':', tableau20[14], 'EvenDB Log', linewidth * 0.6) if key == 'Log space' \
                    else ('-', (0, 0, 0), 'Raw Data', linewidth * 0.4) if key == 'Input size' \
                        else (None, None)
        ax.plot(timeline, value, label=label, linewidth=linew, color=color, linestyle=style)

    ax.set_xlabel(x_label, fontsize=fontsize + 5)
    ax.set_ylabel(y_label, fontsize=fontsize + 5)
    ax.grid()
    ax.autoscale(enable=True, axis='x', tight=True)

    y_top = ax.get_ylim()[1]
    if y_upper is not None:
        y_top = y_upper
    ax.set_ylim(bottom=0, top=y_top)
    ax.set_xticks(ax.get_xticks()[::tick_percent])  # tick every tick_percent

    for label in ax.get_yticklabels() + ax.get_xticklabels():
        label.set_fontsize(fontsize)

    if len(ax.get_xticklabels()) > 7:
        for label in ax.get_xticklabels():
            label.set_rotation(80)

    if ax.get_yticks()[1] - ax.get_yticks()[0] == 0.25:
        ax.set_yticks(ax.get_yticks()[::2])  # skip numbers with too many decimal digits...

    if x_bottom is not None:
        ax.set_xlim(x_bottom, ax.get_xlim()[1])
    if x_upper is not None:
        ax.set_xlim(ax.get_xlim()[0], x_upper)

    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, loc=legend, fontsize=fontsize, ncol=ncol, bbox_to_anchor=bbox_to_anchor, labelspacing=0.3)

    fig.savefig(file_name.replace(' ', '_') + '_line.pdf', bbox_inches='tight')


def draw_dist_chart(file_name, data, x_label=None, y_label=None, x_scale='linear', y_scale='linear',
                    y_upper=None, x_bottom=None, fontsize=myfontsize, ncol=1, x_upper=None, bbox_to_anchor=None):
    fig, ax = plt.subplots()
    values = list(data.values())
    keys = list(data.keys())
    ax.plot(values[0], values[1], label=keys[1],
            linestyle=piwi_linestyle,
            linewidth=linewidth)

    ax.set_xlabel(x_label, fontsize=fontsize + 5)
    ax.set_ylabel(y_label, fontsize=fontsize + 5)
    ax.grid()
    ax.autoscale(enable=True, axis='x', tight=True)

    # y_top = ax.get_ylim()[1]
    # if y_upper is not None:
    #     y_top = y_upper
    # ax.set_ylim(bottom=0, top=y_top)
    # ax.set_xticks(list(map(lambda t: round(t, 3), ax.get_xticks())))
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)

    for label in ax.get_yticklabels() + ax.get_xticklabels():
        label.set_fontsize(fontsize)

    if len(ax.get_xticklabels()) > 7:
        for label in ax.get_xticklabels():
            label.set_rotation(80)

    if x_bottom is not None:
        ax.set_xlim(x_bottom, ax.get_xlim()[1])
    if x_upper is not None:
        ax.set_xlim(ax.get_xlim()[0], x_upper)

    # h, l = ax.get_legend_handles_labels()
    # ax.legend(h, l, loc=legend, fontsize=fontsize, ncol=ncol, bbox_to_anchor=bbox_to_anchor)

    fig.savefig(file_name.replace(' ', '_') + '_line.pdf', bbox_inches='tight')


def read_csv(path="./Pewee - _golden_ benchmark set - csv_for_figs - EuroSys.csv"):
    experiments = dict()

    for workload in workloads:
        experiments[workload] = dict()

    latency_breakdown = {'C': {'flurry': dict(), 'zipfian': dict()},
                         'A': {'flurry': dict(), 'zipfian': dict()}}

    bloom_filter_partitioning = {'A_32': {'flurry': [], 'zipfian': []}}

    write_amp_p = {}
    amplifications = {'P': dict(),
                      'A': dict(),
                      'C': dict()}
    scalability = {}
    caching = {}
    tail = {'flurry': {}, 'zipfian': {}}
    max_log = {}
    p_uniform = {}
    rocks_block_cache = {}
    space_timeline_real = {}
    ingestion = {}
    write_amp_256 = {}
    throughput_256_ingestions = {}
    puts_only_skew = {}
    gets_only_skew = {}
    throughput_64_scans_10s = {}
    throughput_128_scans_10s = {}
    throughput_256_scans_10s = {}
    throughput_64_scans_1s = {}
    throughput_128_scans_1s = {}
    throughput_256_scans_1s = {}
    throughput_64_scans_1m = {}
    throughput_128_scans_1m = {}
    throughput_256_scans_1m = {}
    app_names_loglog = {}
    app_names_cdf = {}

    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0] == 'write_amp_p':
                break
            if row[0] is not '':
                workload = row[0].split()[-1]
                label = ' '.join(row[0].split()[:-1])
                if workload not in experiments:
                    continue
                experiments[workload][label] = \
                    list(map(float, [i for i in row[1:] if i is not '']))

        # P write amplification (new, including uniform)
        for row in csv_reader:
            if row[0] == 'read_write_amplification':
                break
            if row[0] is '':
                continue
            write_amp_p[row[0]] = list(map(float, [i for i in row[1:] if i is not '']))
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
            if row[0] != '':
                scalability[row[0]] = list(map(float, [i for i in row[1:] if i is not '']))

        for row in csv_reader:
            if row[0] == 'tail latency':
                break
            if row[0] == 'Munks RAM':
                continue
            caching[row[0]] = [v / 1000 for v in list(map(float, [i for i in row[1:] if i is not '']))]

        dist = ''
        for row in csv_reader:
            if row[0] == 'max_log':
                break
            if row[0] == '':
                continue
            if row[0] == 'zipfian' or row[0] == 'flurry':
                dist = row[0]
                continue
            tail[dist][row[0]] = list(map(float, [i for i in row[1:] if i is not '']))

        for row in csv_reader:
            if row[0] == 'P uniform':
                break
            if row[0] == 'Throughput' or row[0] == '':
                continue
            max_log[row[0]] = list(map(float, [i for i in row[1:] if i is not '']))

        for row in csv_reader:
            if row[0] == 'RocksDB block cache size':
                break
            if row[0] == '':
                continue
            p_uniform[row[0]] = list(map(float, [i for i in row[1:] if i is not '']))

        for row in csv_reader:
            if row[0] == '256GB real DB ingestion':
                break
            if row[0] == '':
                continue
            if len(rocks_block_cache.keys()) == 0:
                rocks_block_cache[row[0]] = row[1:]
            else:
                rocks_block_cache[row[0]] = list(map(float, [i for i in row[1:] if i is not '']))

        columns_num = 0
        for row in csv_reader:
            if row[0] == 'Puts only':
                break
            if row[0] == '':
                continue
            if len(space_timeline_real.keys()) == 0:
                space_timeline_real[row[0]] = [i for i in row[1:] if i is not '']
                columns_num = len(space_timeline_real[row[0]])
            else:
                space_timeline_real[row[0]] = list(
                    map(lambda v: float(v) if v != '' else None, [i for i in row[1:columns_num + 1]]))

        for row in csv_reader:
            if row[0] == 'Workload C':
                break
            if row[0] == '':
                continue
            # if len(puts_only_skew.keys()) == 0:
            #     puts_only_skew[row[0]] = row[1:]
            # else:
            puts_only_skew[row[0]] = list(map(float, [i for i in row[1:] if i is not '']))

        for row in csv_reader:
            if row[0] == '':
                continue
            # if len(gets_only_skew.keys()) == 0:
            #     gets_only_skew[row[0]] = row[1:]
            # else:
            gets_only_skew[row[0]] = list(map(float, [i for i in row[1:] if i is not '']))

        # no need to redraw real workload graphs, disabled to avoid having to merge input files

        # for row in csv_reader:
        #     if row[0] == 'Write amplification comparison':
        #         break
        #     if row[0] == '':
        #         continue
        #     if len(ingestion.keys()) == 0:
        #         ingestion[row[0]] = row[1:]
        #     else:
        #         ingestion[row[0]] = list(map(float, [i for i in row[1:] if i is not '']))

        # for row in csv_reader:
        #     if row[0] == 'Throughput over time - ingestions':
        #         break
        #     if row[0] == '':
        #         continue
        #     if len(write_amp_256.keys()) == 0:
        #         write_amp_256[row[0]] = row[1:]
        #     else:
        #         write_amp_256[row[0]] = list(map(float, [i for i in row[1:] if i is not '']))

        # columns_num = 0
        # for row in csv_reader:
        #     if row[0] == 'Throughput over time - scans 64 10s':
        #         break
        #     if row[0] == '':
        #         continue
        #     if len(throughput_256_ingestions.keys()) == 0:
        #         throughput_256_ingestions[row[0]] = [i for i in row[1:] if i is not '']
        #         columns_num = len(throughput_256_ingestions[row[0]])
        #     else:
        #         throughput_256_ingestions[row[0]] = list(
        #             map(lambda v: float(v) if v != '' else None, [i for i in row[1:columns_num + 1]]))

        # columns_num = 0
        # for row in csv_reader:
        #     if row[0] == 'Throughput over time - scans 128 10s':
        #         break
        #     if row[0] == '':
        #         continue
        #     if len(throughput_64_scans_10s.keys()) == 0:
        #         throughput_64_scans_10s[row[0]] = [i for i in row[1:] if i is not '']
        #         columns_num = len(throughput_64_scans_10s[row[0]])
        #     else:
        #         throughput_64_scans_10s[row[0]] = list(
        #             map(lambda v: float(v) if v != '' else None, [i for i in row[1:columns_num + 1]]))

        # columns_num = 0
        # for row in csv_reader:
        #     if row[0] == 'Throughput over time - scans 256 10s':
        #         break
        #     if row[0] == '':
        #         continue
        #     if len(throughput_128_scans_10s.keys()) == 0:
        #         throughput_128_scans_10s[row[0]] = [i for i in row[1:] if i is not '']
        #         columns_num = len(throughput_128_scans_10s[row[0]])
        #     else:
        #         throughput_128_scans_10s[row[0]] = list(
        #             map(lambda v: float(v) if v != '' else None, [i for i in row[1:columns_num + 1]]))

        # columns_num = 0
        # for row in csv_reader:
        #     if row[0] == 'Throughput over time - scans 64 1s':
        #         break
        #     if row[0] == '':
        #         continue
        #     if len(throughput_256_scans_10s.keys()) == 0:
        #         throughput_256_scans_10s[row[0]] = [i for i in row[1:] if i is not '']
        #         columns_num = len(throughput_256_scans_10s[row[0]])
        #     else:
        #         throughput_256_scans_10s[row[0]] = list(
        #             map(lambda v: float(v) if v != '' else None, [i for i in row[1:columns_num + 1]]))

        # columns_num = 0
        # for row in csv_reader:
        #     if row[0] == 'Throughput over time - scans 128 1s':
        #         break
        #     if row[0] == '':
        #         continue
        #     if len(throughput_64_scans_1s.keys()) == 0:
        #         throughput_64_scans_1s[row[0]] = [i for i in row[1:] if i is not '']
        #         columns_num = len(throughput_64_scans_1s[row[0]])
        #     else:
        #         throughput_64_scans_1s[row[0]] = list(
        #             map(lambda v: float(v) if v != '' else None, [i for i in row[1:columns_num + 1]]))

        # columns_num = 0
        # for row in csv_reader:
        #     if row[0] == 'Throughput over time - scans 256 1s':
        #         break
        #     if row[0] == '':
        #         continue
        #     if len(throughput_128_scans_1s.keys()) == 0:
        #         throughput_128_scans_1s[row[0]] = [i for i in row[1:] if i is not '']
        #         columns_num = len(throughput_128_scans_1s[row[0]])
        #     else:
        #         throughput_128_scans_1s[row[0]] = list(
        #             map(lambda v: float(v) if v != '' else None, [i for i in row[1:columns_num + 1]]))

        # columns_num = 0
        # for row in csv_reader:
        #     if row[0] == 'Throughput over time - scans 64 1m':
        #         break
        #     if row[0] == '':
        #         continue
        #     if len(throughput_256_scans_1s.keys()) == 0:
        #         throughput_256_scans_1s[row[0]] = [i for i in row[1:] if i is not '']
        #         columns_num = len(throughput_256_scans_1s[row[0]])
        #     else:
        #         throughput_256_scans_1s[row[0]] = list(
        #             map(lambda v: float(v) if v != '' else None, [i for i in row[1:columns_num + 1]]))

        # columns_num = 0
        # for row in csv_reader:
        #     if row[0] == 'Throughput over time - scans 128 1m':
        #         break
        #     if row[0] == '':
        #         continue
        #     if len(throughput_64_scans_1m.keys()) == 0:
        #         throughput_64_scans_1m[row[0]] = [i for i in row[1:] if i is not '']
        #         columns_num = len(throughput_64_scans_1m[row[0]])
        #     else:
        #         throughput_64_scans_1m[row[0]] = list(
        #             map(lambda v: float(v) if v != '' else None, [i for i in row[1:columns_num + 1]]))

        # columns_num = 0
        # for row in csv_reader:
        #     if row[0] == 'Throughput over time - scans 256 1m':
        #         break
        #     if row[0] == '':
        #         continue
        #     if len(throughput_128_scans_1m.keys()) == 0:
        #         throughput_128_scans_1m[row[0]] = [i for i in row[1:] if i is not '']
        #         columns_num = len(throughput_128_scans_1m[row[0]])
        #     else:
        #         throughput_128_scans_1m[row[0]] = list(
        #             map(lambda v: float(v) if v != '' else None, [i for i in row[1:columns_num + 1]]))

        # columns_num = 0
        # for row in csv_reader:
        #     if row[0] == 'apps freq dist':
        #         break
        #     if row[0] == '':
        #         continue
        #     if len(throughput_256_scans_1m.keys()) == 0:
        #         throughput_256_scans_1m[row[0]] = [i for i in row[1:] if i is not '']
        #         columns_num = len(throughput_256_scans_1m[row[0]])
        #     else:
        #         throughput_256_scans_1m[row[0]] = list(
        #             map(lambda v: float(v) if v != '' else None, [i for i in row[1:columns_num + 1]]))

        # for row in csv_reader:
        #     if row[0] == 'app names cdf':
        #         break
        #     if row[0] == '':
        #         continue
        #     if len(app_names_loglog.keys()) == 0:
        #         for key in row:
        #             app_names_loglog[key] = []
        #     else:
        #         for key_val in zip(app_names_loglog.keys(), [i for i in row if i is not '']):
        #             app_names_loglog[key_val[0]].append(float(key_val[1]))

        # for row in csv_reader:
        #     if row[0] == '':
        #         continue
        #     if len(app_names_cdf.keys()) == 0:
        #         for key in row:
        #             app_names_cdf[key] = []
        #     else:
        #         for key_val in zip(app_names_cdf.keys(), [i for i in row if i is not '']):
        #             app_names_cdf[key_val[0]].append(float(key_val[1]))

    return {'experiments': experiments,
            'latency': latency_breakdown,
            'bloom_filter_partitioning': bloom_filter_partitioning,
            'write_amp_p': write_amp_p,
            'amplifications': amplifications,
            'scalability': scalability,
            'caching': caching,
            'tail': tail,
            'max_log': max_log,
            'p_uniform': p_uniform,
            'rocks_block_cache': rocks_block_cache,
            'space_timeline_real': space_timeline_real,
            'puts_only_skew': puts_only_skew,
            'gets_only_skew': gets_only_skew,
            'ingestion': ingestion,
            'write_amp_256': write_amp_256,
            'throughput_256_ingestions': throughput_256_ingestions,
            'throughput_64_scans_10s': throughput_64_scans_10s,
            'throughput_128_scans_10s': throughput_128_scans_10s,
            'throughput_256_scans_10s': throughput_256_scans_10s,
            'throughput_64_scans_1s': throughput_64_scans_1s,
            'throughput_128_scans_1s': throughput_128_scans_1s,
            'throughput_256_scans_1s': throughput_256_scans_1s,
            'throughput_64_scans_1m': throughput_64_scans_1m,
            'throughput_128_scans_1m': throughput_128_scans_1m,
            'throughput_256_scans_1m': throughput_256_scans_1m,
            'app_names_loglog': app_names_loglog,
            'app_names_cdf': app_names_cdf}


def draw_bloom_filter_partitions(chart_name, data):
    partitions = [(val[0]) for val in data['flurry']]
    flurry_throughput = [int(val[2]) / 1000 for val in data['flurry']]
    zipf_throughput = [int(val[2]) / 1000 for val in data['zipfian']]

    fig, ax = plt.subplots()

    ax.plot(partitions, flurry_throughput, label='Flurry', color=flurry_linecolor, linestyle='-', linewidth=linewidth,
            marker='o', markersize=marksize)
    ax.plot(partitions, zipf_throughput, label='Zipfian', color=zip_linecolor, linestyle='-', linewidth=linewidth,
            marker='o', markersize=marksize)

    ax.set(xlabel='Partitions', ylabel='Throughput, Kops',
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
        if workload == 'P':
            lines += [{'label': renamings(k), 'data': v, 'style': line_color[k]}
                      for (k, v) in data['p_uniform'].items()]
        if len(lines) == 4:
            lines = [lines[1], lines[3], lines[0], lines[2]]
            legend_image = True
        else:
            legend_image = False
        draw_line_chart('Workload ' + workload,
                        lines,
                        yaxis='Throughput, Kops', legend_image=legend_image)


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
    line_color = {'Flurry': {'color': tableau20[0], 'linestyle': '-', 'linewidth': linewidth, 'marker': rocks_marker},
                  'Zipfian': {'color': tableau20[2], 'linestyle': '-', 'linewidth': linewidth, 'marker': rocks_marker}}

    flurry_line = {'label': renamings('flurry'),
                   'data': [int(val[2]) / 1000 for val in data['bloom_filter_partitioning']['A_32']['flurry']],
                   'style': line_color['Flurry']}
    zipf_line = {'label': renamings('Zipfian'),
                 'data': [int(val[2]) / 1000 for val in data['bloom_filter_partitioning']['A_32']['zipfian']],
                 'style': line_color['Zipfian']}

    draw_line_chart(file_name='Bloom filter', lines=[flurry_line, zipf_line], yaxis='Throughput, Kops', legend=2,
                    y_upper=190, x=['1', '2', '4', '8', '16'], x_label='Split factor', x_bottom=0)


def draw_ampl_charts(data):
    write_amp_p = data['write_amp_p']
    lines = [{'label': renamings(k), 'data': v, 'style': line_color[k]}
             for (k, v) in write_amp_p.items()]
    draw_line_chart('write_amp_p', lines,
                    yaxis='Write amplification', legend_image=False, legend='upper left', y_upper=12)

    # amplifications = data['amplifications']
    # # P write amplification disk:
    # draw_line_chart('P_write_amplification_disk',
    #                 [{'label': ' '.join(k.split()[0:2]), 'data': v, 'style': line_color[' '.join(k.split()[0:2])]}
    #                  for (k, v) in amplifications['P'].items() if 'disk' in k],
    #                 yaxis='Amplification', legend=3)

    # # C read amplification disk:
    # draw_line_chart('C_read_amplification_disk',
    #                 [{'label': ' '.join(k.split()[0:2]), 'data': v, 'style': line_color[' '.join(k.split()[0:2])]}
    #                  for (k, v) in amplifications['C'].items() if 'disk' in k],
    #                 yaxis='Amplification', legend=3)

    # draw_line_chart('C_read_amplification_kernel',
    #                 [{'label': ' '.join(k.split()[0:2]), 'data': v, 'style': line_color[' '.join(k.split()[0:2])]}
    #                  for (k, v) in amplifications['C'].items() if 'kernel' in k],
    #                 yaxis='Amplification', legend=3)


def draw_scalability_charts(data):
    line_color = {'P Flurry': {'color': tableau20[0], 'linestyle': '-', 'linewidth': linewidth, 'marker': rocks_marker},
                  'P Zipfian': {'color': tableau20[0], 'linestyle': ':', 'linewidth': linewidth,
                                'marker': rocks_marker},
                  'A Flurry': {'color': tableau20[2], 'linestyle': '-', 'linewidth': linewidth, 'marker': piwi_marker},
                  'A Zipfian': {'color': tableau20[2], 'linestyle': ':', 'linewidth': linewidth, 'marker': piwi_marker},
                  'C Flurry': {'color': tableau20[4], 'linestyle': '-', 'linewidth': linewidth, 'marker': piwi_marker},
                  'C Zipfian': {'color': tableau20[4], 'linestyle': ':', 'linewidth': linewidth, 'marker': piwi_marker}}

    lines = [{'label': renamings(k), 'data': v, 'style': line_color[k]} for (k, v) in data['scalability'].items()]
    draw_line_chart(file_name='scalability', lines=lines, chart_name='', yaxis='Throughput, Kops', legend=2,
                    y_upper=200,
                    x=[1, 2, 4, 8, 12], x_label='Threads', x_bottom=0, bbox_to_anchor=(1, 1.03))


def draw_caching_effect(data):
    line_color = {'P flurry': {'color': tableau20[0], 'linestyle': '-', 'linewidth': linewidth, 'marker': rocks_marker},
                  'P zipf': {'color': tableau20[0], 'linestyle': ':', 'linewidth': linewidth, 'marker': rocks_marker},
                  'A flurry': {'color': tableau20[2], 'linestyle': '-', 'linewidth': linewidth, 'marker': piwi_marker},
                  'A zipf': {'color': tableau20[2], 'linestyle': ':', 'linewidth': linewidth, 'marker': piwi_marker},
                  'C flurry': {'color': tableau20[4], 'linestyle': '-', 'linewidth': linewidth, 'marker': piwi_marker},
                  'C zipf': {'color': tableau20[4], 'linestyle': ':', 'linewidth': linewidth, 'marker': piwi_marker}}

    lines = [{'label': renamings(k),
              'data': [int((-i / v[4] + 1) * 100) if i < v[4] else 5 - int((v[4] / i - 1) * 100) for i in v],
              'style': line_color[k]} for (k, v) in data['caching'].items() if 'flurry' in k]
    draw_munk_size_chart(file_name='cache', bars=lines)
    # draw_line_chart(file_name='cache', lines=lines, chart_name='', yaxis='Throughput, Kops', legend=3, y_upper=None, x=[0,2,4,6,8,10], x_label='Munk cache GB', x_bottom=0)


def draw_95(data):
    line_color = {
        'Rocks 95% Get': {'color': tableau20[0], 'linestyle': ':', 'linewidth': linewidth, 'marker': rocks_marker},
        'Rocks 95% Put': {'color': tableau20[2], 'linestyle': ':', 'linewidth': linewidth, 'marker': rocks_marker},
        'Piwi 95% Get': {'color': tableau20[0], 'linestyle': '-', 'linewidth': linewidth, 'marker': piwi_marker},
        'Piwi 95% Put': {'color': tableau20[2], 'linestyle': '-', 'linewidth': linewidth, 'marker': piwi_marker}}

    x = [l[0:-2] for l in x_axis] # remove the GB
    lines = [{'label': renamings(k), 'data': v, 'style': line_color[k]} for (k, v) in data['tail']['flurry'].items()]
    draw_line_chart(file_name='tail_flurry', lines=[lines[0], lines[2], lines[1], lines[3]], chart_name='', y_upper=1.5,
                    yaxis='Latency, [ms]', x_label='Dataset size (GB)', x=x, legend=2, x_bottom=0, fontsize=myfontsize + 5)

    lines = [{'label': renamings(k), 'data': v, 'style': line_color[k]} for (k, v) in data['tail']['zipfian'].items()]
    draw_line_chart(file_name='tail_zipf', lines=[lines[0], lines[2], lines[1], lines[3]], chart_name='', y_upper=2.5,
                    yaxis='Latency, [ms]', x_label='Dataset size (GB)', x=x, legend=2, x_bottom=0, fontsize=myfontsize + 5)


def draw_log_size_charts(data):
    line_color = {
        'E100 Flurry': {'color': tableau20[0], 'linestyle': '-', 'linewidth': linewidth, 'marker': rocks_marker},
        'E100 Zipfian': {'color': tableau20[0], 'linestyle': ':', 'linewidth': linewidth, 'marker': rocks_marker},
        'A Flurry': {'color': tableau20[2], 'linestyle': '-', 'linewidth': linewidth, 'marker': piwi_marker},
        'A Zipfian': {'color': tableau20[2], 'linestyle': ':', 'linewidth': linewidth, 'marker': piwi_marker}}
    lines = [{'label': renamings(k), 'data': [i / 1000 for i in v], 'style': line_color[k]} for (k, v) in
             data['max_log'].items()]
    draw_line_chart(file_name='max_log_size', lines=lines, chart_name='', yaxis='Throughput, Kops', legend=2,
                    x=['128K', '256K', '512K', '1M', '2M', '4M'], x_label='Maximum log size', x_bottom=0, y_upper=225)


def draw_real_data_bar_charts(data):
    draw_piwi_vs_rocks_bars('ingestion', data['ingestion'], 'Dataset size', 'Throughput, Kops')
    draw_piwi_vs_rocks_bars('write_amp_256', data['write_amp_256'], 'Dataset size', 'Write amplification')


def draw_timeline_charts(data):
    draw_timeline_chart('throughput_256_ingestions', data['throughput_256_ingestions'], fontsize=myfontsize + 5,
                        legend=1)
    draw_timeline_chart('throughput_64_scans_10s', data['throughput_64_scans_10s'], fontsize=myfontsize + 5, legend=4,
                        tick_sec=10)
    draw_timeline_chart('throughput_128_scans_10s', data['throughput_128_scans_10s'], fontsize=myfontsize + 5, legend=4,
                        tick_sec=15)
    draw_timeline_chart('throughput_256_scans_10s', data['throughput_256_scans_10s'], fontsize=myfontsize + 5, legend=4,
                        tick_sec=15)
    draw_timeline_chart('throughput_64_scans_1s', data['throughput_64_scans_1s'], fontsize=myfontsize + 5, legend=4,
                        tick_sec=5)
    draw_timeline_chart('throughput_128_scans_1s', data['throughput_128_scans_1s'], fontsize=myfontsize + 5, legend=2,
                        tick_sec=10)
    draw_timeline_chart('throughput_256_scans_1s', data['throughput_256_scans_1s'], fontsize=myfontsize + 5, legend=2,
                        tick_sec=10)
    draw_timeline_chart('throughput_64_scans_1m', data['throughput_64_scans_1m'], fontsize=myfontsize + 5, legend=4,
                        tick_sec=15)
    draw_timeline_chart('throughput_128_scans_1m', data['throughput_128_scans_1m'], fontsize=myfontsize + 5, legend=4)
    draw_timeline_chart('throughput_256_scans_1m', data['throughput_256_scans_1m'], fontsize=myfontsize + 5, legend=4)


def draw_space_timeline_charts(data):
    draw_space_timeline_chart('space_timeline_real', data['space_timeline_real'], fontsize=myfontsize + 5,
                        legend=2, y_upper=400, tick_percent=25)


def draw_skew_charts(data):
    puts_only_skew = data['puts_only_skew']
    lines = [{'label': renamings(k), 'data': v, 'style': line_color[k]} for (k, v) in puts_only_skew.items()]
    draw_line_chart(file_name='puts_only_skew', lines=lines, chart_name='', yaxis='Throughput, Kops', legend=1,
                    y_upper=250, x=['0.99', '0.95', '0.90', '0.85', '0.80'], x_label='Zipf Theta', x_bottom=0)

    gets_only_skew = data['gets_only_skew']
    lines = [{'label': renamings(k), 'data': v, 'style': line_color[k]} for (k, v) in gets_only_skew.items()]
    draw_line_chart(file_name='gets_only_skew', lines=lines, chart_name='', yaxis='Throughput, Kops', legend=1,
                    y_upper=250, x=['0.99', '0.95', '0.90', '0.85', '0.80'], x_label='Zipf Theta', x_bottom=0)


def draw_dist_charts(data):
    draw_dist_chart('app_names_loglog', data['app_names_loglog'], x_scale='log', y_scale='log',
                    y_label='Probability density', x_label='App popularity ranking', fontsize=myfontsize + 5)
    draw_dist_chart('app_names_cdf', data['app_names_cdf'], y_label='CDF', x_label='App name percentile',
                    fontsize=myfontsize + 5)


def draw_rocks_block_cache_charts(data):
    draw_rocks_block_cache_chart('rocks_block_cache', data['rocks_block_cache'], 'Block cache size', 'Ratio vs. default config')


def main():
    data = read_csv()

    # draw_line_charts(data)
    # draw_speedup_charts(data)
    # draw_latency_charts(data)
    # draw_bloom_filter_charts(data)
    # draw_ampl_charts(data)
    # draw_scalability_charts(data)
    ###draw_caching_effect(data)
    # draw_95(data)
    draw_space_timeline_charts(data)
    # draw_skew_charts(data)
    # draw_log_size_charts(data)
    # draw_real_data_bar_charts(data)
    # draw_timeline_charts(data)
    # draw_dist_charts(data)
    # draw_rocks_block_cache_charts(data)
    plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
