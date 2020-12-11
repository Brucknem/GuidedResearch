import os
from pathlib import Path

import pandas as pd
import numpy as np
from bokeh.io import show, output_file, save
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from os import listdir
from os.path import isfile, join
import sys

from bokeh.palettes import viridis, magma, linear_palette

from utils import get_maxima, get_frequencies

UNIX_TIMESTAMP = 'Unix Timestamp'

TIMESTAMP = 'Timestamp'

MILLISECONDS = 'Milliseconds'

TOOLTIPS = [
    ("index", "$index"),
    ("(x,y)", "($x, $y)"),
]
tools = "pan,wheel_zoom,box_zoom,reset,hover,save"


def get_frequencies_renderable(maxima: list, milliseconds: list, interval: int = -1):
    frequencies = get_frequencies(maxima, milliseconds, interval)
    frequency_x = np.array(list(frequencies.keys())) / 1000
    return frequency_x, list(frequencies.values())


def generate_plot(filename):
    df = pd.read_csv(filename)
    df = df.drop_duplicates()

    one_hour = 60 * 60
    start_time = 1607432525 - 1 + one_hour
    end_time = 1607432580 + 1 + one_hour

    start_time = 0
    end_time = 2607432580

    try:
        df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], errors='coerce')
        df[UNIX_TIMESTAMP] = df[TIMESTAMP].astype(int) / 10 ** 9
        df = df[df[UNIX_TIMESTAMP] >= start_time]
        df = df[df[UNIX_TIMESTAMP] <= end_time]
        df = df.drop(columns=[UNIX_TIMESTAMP])
    except Exception:
        print('EXCEPTION THROWN WHILE CONVERTING TIMESTAMPS')

    milliseconds = list(df[MILLISECONDS])
    df[MILLISECONDS] = df[MILLISECONDS] / 1000
    value_columns = list(df.columns)
    try:
        del value_columns[value_columns.index(MILLISECONDS)]
        del value_columns[value_columns.index(TIMESTAMP)]
    except:
        pass

    value_columns.sort()

    colors = linear_palette(('#962626', '#a818a8', '#1b1ba8', '#AAAAAA', '#168080', '#168c16'), len(value_columns))
    # colors = linear_palette(('#A7226E', '#F26B38', '#EC2049',  '#45ADA8', '#2F9599', '#F7DB4F'), len(value_columns))
    # colors = magma(len(value_columns))

    title = filename.split('/')[-1]
    width = 800
    height = 600
    fig = figure(title=title, tooltips=TOOLTIPS, tools=tools, active_drag="pan", plot_width=width, plot_height=height)

    milliseconds_x_axis = list(df[MILLISECONDS])

    for index, column in enumerate(value_columns):
        fig.line(milliseconds_x_axis, list(df[column]), color=colors[index], alpha=.8, legend_label=column)
        maxima = get_maxima(df[column], threshold=0.00)
        fig.circle(milliseconds_x_axis, maxima, color=colors[index], alpha=.4, legend_label=column)

        frequencies_x, frequencies_y = get_frequencies_renderable(maxima, milliseconds)
        fig.step(frequencies_x, frequencies_y, color=colors[index], alpha=.7, line_dash='dotted', mode="after",
                 legend_label='{} [1/s (avg.)]'.format(column))
        #
        # frequencies_x, frequencies_y = get_frequencies_renderable(maxima, milliseconds, 1000)
        # fig.step(frequencies_x, frequencies_y, color=colors[index], alpha=.7, line_dash='dashdot', mode="after",
        #          legend_label='{} [1/s (1s)]'.format(column))

    fig.legend.click_policy = "hide"

    return fig


def generate_plots(data_folder, output_filename, plots_per_line):
    csv_files = [join(data_folder, f) for f in listdir(data_folder) if isfile(join(data_folder, f))]
    csv_files = [f for f in csv_files if f.endswith('.csv')]
    csv_files.sort()

    plots = {}
    for file in csv_files:
        plots[file] = generate_plot(file)


    if plots_per_line > 0:
        while len(plots) % plots_per_line != 0:
            plots[str(len(plots))] = None
        output_html = '{}.html'.format(output_filename)
        output_file(os.path.join(data_folder, output_filename, output_html))
        plots = list(plots.values())
        plots = list(np.array(plots).reshape(-1, plots_per_line))
        final_plot = gridplot(plots)
        show(final_plot)
        save(final_plot)
    else:
        output_folder = os.path.join(data_folder, output_filename)
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        for filename, plot in plots.items():
            output_html = '{}.html'.format(os.path.basename(filename).split('.')[0])
            filename = os.path.join(output_folder, output_html)
            output_file(filename)
            show(plot)
            save(plot)


help_text = '''
Usage:   python {}  input_dir   [plots_per_line]

        input_dir:          The input directory containing some csv files
        [plots_per_line]:   An optional number of plots per line. 
                            > 0     - The plots are merged into a single output plot with the given plots per line.
                            <= 0    - One output file is generated per plot
'''.format(os.path.basename(__file__))

if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(help_text)
        sys.exit(1)

    data_folder = os.path.expanduser(sys.argv[1])

    plots_per_line = -1
    if len(sys.argv) == 3:
        plots_per_line = int(sys.argv[2])

    generate_plots(data_folder, 'plots', plots_per_line)

    sys.exit(0)
