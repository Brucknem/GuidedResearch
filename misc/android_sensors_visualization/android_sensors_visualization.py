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
    milliseconds = list(df[MILLISECONDS])
    df[MILLISECONDS] = df[MILLISECONDS] / 1000
    value_columns = list(df.columns)
    try:
        del value_columns[value_columns.index(MILLISECONDS)]
        del value_columns[value_columns.index(TIMESTAMP)]
    except:
        pass

    value_columns.sort()
    cds = ColumnDataSource(df)

    # colors = linear_palette(('#FF0000', '#0000FF'), len(value_columns))
    colors = viridis(len(value_columns))

    title = filename.split('/')[-1]
    fig = figure(title=title, tooltips=TOOLTIPS, tools=tools, active_drag="pan", plot_width=1920, plot_height=1200)

    for index, column in enumerate(value_columns):
        fig.line(MILLISECONDS, column, source=cds, color=colors[index], alpha=.8, legend_label=column)
        maxima = get_maxima(df[column], threshold=0.10)
        fig.circle(df[MILLISECONDS], maxima, color=colors[index], alpha=.8, legend_label=column)

        frequencies_x, frequencies_y = get_frequencies_renderable(maxima, milliseconds)
        fig.step(frequencies_x, frequencies_y, color=colors[index], alpha=.6, line_dash='dotted', mode="after",
                 legend_label='{} [Hz (avg.)]'.format(column))

        frequencies_x, frequencies_y = get_frequencies_renderable(maxima, milliseconds, 1000)
        fig.step(frequencies_x, frequencies_y, color=colors[index], alpha=.6, line_dash='dashdot', mode="after",
                 legend_label='{} [Hz (1s)]'.format(column))

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
        output_file('{}.html'.format(output_filename))
        plots = list(plots.values())
        plots = list(np.array(plots).reshape(-1, plots_per_line))
        final_plot = gridplot(plots)
        show(final_plot)
        save(final_plot)
    else:
        for filename, plot in plots.items():
            Path(output_filename).mkdir(parents=True, exist_ok=True)
            filename = os.path.basename(filename)
            output_file(os.path.join(output_filename, '{}.html'.format(filename.split('.')[0])))
            show(plot)
            save(plot)


help_text = '''
Usage:   python {}  input_dir   output  [plots_per_line]

        input_dir:          The input directory containing some csv files
        output:             The output folder   - If no optional or less or equal to 0 plots_per_line is given
                            The output filename - If plots_per_line is set greater than 0
        [plots_per_line]:   An optional number of plots per line. 
                            > 0     - The plots are merged into a single output plot with the given plots per line.
                            <= 0    - One output file is generated per plot
'''.format(os.path.basename(__file__))

if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print(help_text)
        sys.exit(1)

    data_folder = os.path.expanduser(sys.argv[1])
    output_filename = os.path.expanduser(sys.argv[2])
    if '.' in output_filename:
        output_filename = output_filename.split('.')[0]

    plots_per_line = -1
    if len(sys.argv) == 4:
        plots_per_line = int(sys.argv[3])

    generate_plots(data_folder, output_filename, plots_per_line)

    sys.exit(0)
