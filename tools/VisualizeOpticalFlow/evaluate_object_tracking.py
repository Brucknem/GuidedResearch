import sys
from math import sqrt
from os.path import join
from pathlib import Path
import inspect

import numpy as np
import pandas as pd
from bokeh.io import save, show
from bokeh.palettes import mpl
from bokeh.plotting import figure, output_file

from main import *

display = True


def get_output_filename(foldername):
    folder = Path(join('evaluate_object_tracking', Path(filename).name, foldername))
    folder.mkdir(exist_ok=True, parents=True)
    return str(join(folder, "path.html"))


def setup():
    output_file(get_output_filename(inspect.stack()[1].function))

    p = figure(plot_width=1600, plot_height=1000, tools=tools)
    p.title.text = 'Path of object [' + title_suffix + ']'

    return p


def xy_plot(p, name, color):
    x = df[name + " [mx]"]
    y = df[name + " [my]"]

    indices = x > 10
    x = x[indices]
    y = y[indices]

    indices = y > 10
    x = x[indices]
    y = y[indices]

    indices = x < 1900
    x = x[indices]
    y = y[indices]

    indices = y < 1190
    x = x[indices]
    y = y[indices]

    x = x.rolling(window).mean()
    y = y.rolling(window).mean()

    # p.triangle_pin(x, y, size=10, color=color, alpha=0.5, legend_label=name)
    p.line(x, y, line_width=2, color=color, alpha=0.5, legend_label=name)


def xy_plots():
    p = setup()

    colors = get_colors(4 + 1)
    xy_plot(p, "Original", color=colors[0])
    xy_plot(p, "FAST", color=colors[1])
    xy_plot(p, "ORB", color=colors[2])
    xy_plot(p, "SURF", color=colors[3])

    p.y_range.flipped = True
    p.xaxis.axis_label = "X [px]"
    p.yaxis.axis_label = "Y [px]"

    set_plot_settings(p)
    show_or_save(p, display)


window = 1
xy_plots()
