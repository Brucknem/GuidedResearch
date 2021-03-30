import sys
from math import sqrt
from os.path import join
from pathlib import Path
import inspect
import os

import numpy as np
import pandas as pd
from bokeh.io import save, show
from bokeh.models import Legend
from bokeh.palettes import mpl
from bokeh.plotting import figure, output_file

from main import *

display = True

folder = sys.argv[1]
if not folder or not Path(folder).is_dir():
    print("Please specify an input directory containing the pixel paths as .csv")
    exit(-1)


def get_output_filename(foldername, filename):
    directory = Path(join('evaluate_object_tracking', foldername))
    directory.mkdir(exist_ok=True, parents=True)
    return str(join(directory, filename + ".html"))


def setup(foldername, filename):
    output_file(get_output_filename(foldername, filename))

    p = figure(plot_width=plot_width, plot_height=plot_height, tools=tools)
    p.title.text = 'Tracking of ' + filename.replace('.csv', '') + ' [' + title_suffix + ']'
    # p.add_layout(Legend(), 'right')

    return p


def xy_plot(df, p, name, color):
    x = df[name + " [mx]"]
    y = df[name + " [my]"]

    indices = x >= int(sys.argv[3])
    x = x[indices]
    y = y[indices]

    indices = y >= int(sys.argv[4])
    x = x[indices]
    y = y[indices]

    indices = x < 1920 - int(sys.argv[5])
    x = x[indices]
    y = y[indices]

    indices = y < 1200 - int(sys.argv[6])
    x = x[indices]
    y = y[indices]

    x = x.rolling(window).mean()
    y = y.rolling(window).mean()

    # p.triangle_pin(x, y, size=10, color=color, alpha=0.5, legend_label=name)
    p.line(x, y, line_width=2, color=color, alpha=0.5, legend_label=name)


def xy_plots(df, foldername, filename):
    p = setup(foldername, filename)

    colors = get_colors(4 + 1)
    xy_plot(df, p, "Original", color=colors[0])
    xy_plot(df, p, "FAST", color=colors[1])
    xy_plot(df, p, "ORB", color=colors[2])
    xy_plot(df, p, "SURF", color=colors[3])

    p.y_range.flipped = True
    p.xaxis.axis_label = "X [px]"
    p.yaxis.axis_label = "Y [px]"
    # p.legend.location = "right"

    set_plot_settings(p)
    show_or_save(p, display)


window = 1

for csv in os.listdir(folder):
    if csv.endswith(".csv"):
        print(os.path.join(folder, csv))
        df = pd.read_csv(os.path.join(folder, csv))
        xy_plots(df, Path(folder).name, csv)
