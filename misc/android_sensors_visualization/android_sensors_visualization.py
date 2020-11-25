import os

import pandas as pd
import numpy as np
from bokeh.io import show, output_file, save
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from os import listdir
from os.path import isfile, join
import sys

colors = ['red', 'blue', 'green', 'yellow', 'violet', 'black', 'magenta', 'cyan']
TOOLTIPS = [
    ("index", "$index"),
    ("(x,y)", "($x, $y)"),
]
tools="pan,wheel_zoom,box_zoom,reset,hover,save"

def generate_plot(filename):
    df = pd.read_csv(filename)
    df['Milliseconds'] = df['Milliseconds'] / 1000.0
    cds = ColumnDataSource(df)

    value_columns = list(df.columns)
    del value_columns[0]
    del value_columns[0]

    title = filename.split('/')[-1]

    fig = figure(title=title, tooltips=TOOLTIPS, tools=tools, active_drag="pan")

    for index, column in enumerate(value_columns):
        fig.line('Milliseconds', column, source=cds, color=colors[index], legend_label=column)
    fig.legend.click_policy="hide"

    return fig


def generate_plots(data_folder, output_filename, plots_per_line):
    csv_files = [join(data_folder, f) for f in listdir(data_folder) if isfile(join(data_folder, f))]
    csv_files = [f for f in csv_files if f.endswith('.csv')]

    output_file(output_filename)
    plots = []
    for file in csv_files:
        plots.append(generate_plot(file))

    while len(plots) % plots_per_line != 0:
        plots.append(None)

    plots = list(np.array(plots).reshape(-1, plots_per_line))
    final_plot = gridplot(plots)
    show(final_plot)
    save(final_plot)


if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print('Usage:\n\tpython {} {} {} {}'.format(os.path.basename(__file__), '<path_to_folder_with_csvs>',
                                                 '<output_filename>', '[opt_number_of_plots_per_line]'))
        sys.exit(1)
        
    data_folder = os.path.expanduser(sys.argv[1])
    output_filename = os.path.expanduser(sys.argv[2])
    if '.' in output_filename:
        output_filename = output_filename.split('.')[0]
    output_filename += '.html'

    plots_per_line = 3
    if len(sys.argv) == 4:
        plots_per_line = int(sys.argv[3])

    generate_plots(data_folder, output_filename, plots_per_line)
    
    sys.exit(0)

