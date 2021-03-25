from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.io import save
from bokeh.palettes import mpl
from bokeh.plotting import figure, output_file

filename = 'evaluateDynamicStabilization'
Path(filename).mkdir(exist_ok=True)

df = pd.read_csv(filename + '.csv')

x = df['FrameId']
font = "Times"

font_size = '28pt'
tick_font_size = '24pt'
title_font_size = '32pt'


def calculate_statistics(original, window_size):
    window_size_half = int((window_size + 1) / 2)

    means = np.array([0.] * len(original))
    stds = np.array([0.] * len(original))

    for i in range(len(original)):
        window = [original[i - window_size_half + x] for x in range(window_size) if
                  0 <= i - window_size_half + x < len(original)]
        means[i] = np.mean(window)
        stds[i] = np.std(window)

    return means, stds


def add_means(plot, means, color, legend_label):
    plot.line(x=x,
              y=means,
              color=color,
              legend_label=legend_label)


def add_stds(plot, means, stds, color, legend_label):
    plot.varea(x=x,
               y1=means - stds,
               y2=means + stds,
               color=color,
               alpha=0.2,
               legend_label=legend_label)


def create_plot(window_size):
    output_file(join(filename, "moving_average_window_size_" + str(window_size) + ".html"))

    p = figure(plot_width=1600, plot_height=1000, tools="save,crosshair,pan,wheel_zoom,box_zoom,undo,redo,reset")

    p.title.text = 'Comparison of mean pixel shifts after dynamic stabilization'

    columns_names = ['Original', 'FAST', 'ORB', 'SURF']
    columns = len(columns_names)
    means = [0.] * columns
    stds = [0.] * columns
    # colors = ["#DA4167", "#F5853F", "#313D5A", "#87B38D"]
    colors = mpl['Inferno'][5]

    for i, column in enumerate(columns_names):
        means[i], stds[i] = calculate_statistics(df[column], window_size=window_size)

    for i in range(columns):
        add_stds(p, means[i], stds[i], colors[i], columns_names[i])

    for i in range(columns):
        add_means(p, means[i], colors[i], columns_names[i])

    p.legend.location = "top"
    p.legend.click_policy = "hide"
    p.legend.label_text_font = font
    p.legend.label_text_font_size = font_size

    p.title.text_font = font
    p.title.text_font_size = title_font_size

    p.xaxis.axis_label = "Frame"
    p.yaxis.axis_label = "Mean pixel shift [px]"

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

    # show(p)
    save(p)


create_plot(1)
for window_size in range(5, len(x), 5):
    create_plot(window_size)
