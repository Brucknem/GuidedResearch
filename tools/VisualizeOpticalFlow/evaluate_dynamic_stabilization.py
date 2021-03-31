import sys
from os.path import join
from pathlib import Path
import inspect

import numpy as np
import pandas as pd
from bokeh.io import save, show
from bokeh.models import Legend
from bokeh.palettes import mpl
from bokeh.plotting import figure, output_file

from commons import *

display = True

filename = sys.argv[1]
if not filename or not filename.endswith('.csv'):
    print("Please specify .csv input file")
    exit(-1)

df = pd.read_csv(filename)

x = df['Frame']


def add_means(plot, means, color, legend_label):
    plot.line(x=x,
              y=means,
              color=color,
              legend_label=legend_label,
              line_width=2)


def add_stds(plot, means, stds, color, legend_label):
    plot.varea(x=x,
               y1=means - stds,
               y2=means + stds,
               color=color,
               alpha=0.2,
               legend_label=legend_label)


def get_statistics(window_size):
    columns_names = ['Original', 'FAST', 'ORB', 'SURF']
    columns = len(columns_names)
    means = [0.] * columns
    stds = [0.] * columns

    for i, column in enumerate(columns_names):
        means[i] = df[column].rolling(window_size).mean()
        stds[i] = df[column].rolling(window_size).std()

    return columns, columns_names, means, stds


def get_output_filename(foldername, window_size):
    folder = Path(join('evaluate_dynamic_stabilization', Path(filename).name, foldername))
    folder.mkdir(exist_ok=True, parents=True)
    return str(join(folder, "window_size_" + str(window_size) + ".html"))


def setup(window_size):
    output_file(get_output_filename(inspect.stack()[1].function, window_size))

    p = figure(plot_width=plot_width, plot_height=plot_height, tools=tools)
    p.title.text = 'Damping of mean pixel shifts after dynamic stabilization [' + title_suffix + ']'
    p.add_layout(Legend(), 'right')

    columns, columns_names, means, stds = get_statistics(window_size)

    return p, columns, columns_names, means, stds


def compare_of_mean_pixel_shift(window_size):
    p, columns, columns_names, means, stds = setup(window_size)
    colors = get_colors(len(columns_names))
    for i in range(columns):
        add_stds(p, means[i], stds[i], colors[i], columns_names[i])

    for i in range(columns):
        add_means(p, means[i], colors[i], columns_names[i])

    set_plot_settings(p)
    p.xaxis.axis_label = "Frame"
    p.yaxis.axis_label = "Mean pixel shift [px]"

    show_or_save(p, display)


def deltas_of_mean_pixel_shift(window_size):
    p, columns, columns_names, means, stds = setup(window_size)
    colors = get_colors(len(columns_names))

    deltas = [means[0] - means[i] for i in range(columns)]
    stds = [abs(stds[0] - stds[i]) for i in range(columns)]

    for i in range(columns):
        add_stds(p, deltas[i], stds[i], colors[i], columns_names[i])

    for i in range(columns):
        add_means(p, deltas[i], colors[i], columns_names[i])

    set_plot_settings(p)
    p.xaxis.axis_label = "Frame"
    p.yaxis.axis_label = "Mean pixel shift [px]"
    show_or_save(p, display)


def calculate_percentage_of_better_frames():
    columns, columns_names, means, stds = get_statistics(1)
    total_number_frames = len(means[0])

    deltas = [means[0] - means[i] for i in range(columns)]

    better_frames = [np.where(delta >= 0)[0].shape[0] for delta in deltas]
    better_frames_fractions = [better_frame / total_number_frames for better_frame in better_frames]
    better_frames_percentages = [better_frames_fraction * 100 for better_frames_fraction in better_frames_fractions]

    worse_frames = [total_number_frames - better_frame for better_frame in better_frames]
    worse_frames_fractions = [1 - better_frames_fraction for better_frames_fraction in better_frames_fractions]
    worse_frames_percentages = [100 - better_frames_percentage for better_frames_percentage in
                                better_frames_percentages]

    means_means = [np.mean(mean) for mean in means]

    data = {
        'Stabilizer': columns_names,
        'Mean pixel shift': means_means,
        'Better [abs]': better_frames,
        'Better [frac]': better_frames_fractions,
        'Better [%]': better_frames_percentages,
        'Worse [abs]': worse_frames,
        'Worse [frac]': worse_frames_fractions,
        'Worse [%]': worse_frames_percentages
    }

    result_df = pd.DataFrame(data=data)

    folder = Path(join('evaluation', Path(filename).name))
    folder.mkdir(exist_ok=True, parents=True)
    result_df.to_csv(join(str(folder), 'stats.csv'))


calculate_percentage_of_better_frames()

# create_plot(1)
for window_size in [1, int(25 / 8), int(25 / 4), int(25 / 2), 25, 50, 75, 100, 125]:
    deltas_of_mean_pixel_shift(window_size)
    compare_of_mean_pixel_shift(window_size)
