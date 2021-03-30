import sys
from os.path import join
from pathlib import Path
import inspect

import numpy as np
import pandas as pd
from bokeh.io import save, show
from bokeh.palettes import mpl
from bokeh.plotting import figure, output_file

filename = sys.argv[1]
if not filename or not filename.endswith('.csv'):
    print("Please specify .csv input file")
    exit(-1)

df = pd.read_csv(filename)

font = "Times"

font_size = '28pt'
tick_font_size = '24pt'
title_font_size = '32pt'
title_suffix = sys.argv[2]

tools = "save,crosshair,pan,wheel_zoom,box_zoom,undo,redo,reset"


def get_colors(amount):
    return mpl['Inferno'][amount]


def set_plot_settings(p):
    p.legend.location = "top"
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


def show_or_save(p, display):
    if display:
        show(p)
    else:
        save(p)
