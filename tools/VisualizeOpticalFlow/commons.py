import sys
from os.path import join
from pathlib import Path
import inspect

import numpy as np
import pandas as pd
from bokeh.io import save, show
from bokeh.palettes import mpl, Inferno, Cividis256, Turbo256
from bokeh.plotting import figure, output_file

font = "Times"

font_size = '28pt'
tick_font_size = '24pt'
title_font_size = '32pt'
title_suffix = sys.argv[2]

tools = "save,crosshair,pan,wheel_zoom,box_zoom,undo,redo,reset"
plot_width = 1600
plot_height = 800


def get_colors(amount):
    return Turbo256[::int(256 / amount)]


def set_plot_settings(p):
    p.legend.click_policy = "hide"
    p.legend.label_text_font = font
    p.legend.label_text_font_size = font_size

    p.title.text_font = font
    p.title.text_font_size = title_font_size

    p.xaxis.axis_label_text_font = font
    p.yaxis.axis_label_text_font = font

    p.xaxis.axis_label_text_font_size = font_size
    p.yaxis.axis_label_text_font_size = font_size

    p.xaxis.major_label_text_font = font
    p.yaxis.major_label_text_font = font

    p.xaxis.major_label_text_font_size = tick_font_size
    p.yaxis.major_label_text_font_size = tick_font_size

    p.xaxis.axis_label_text_font_style = "normal"
    p.yaxis.axis_label_text_font_style = "normal"

    p.legend.orientation = "horizontal"
    p.legend.location = "top_right"


def show_or_save(p, display):
    if display:
        show(p)
    else:
        save(p)
