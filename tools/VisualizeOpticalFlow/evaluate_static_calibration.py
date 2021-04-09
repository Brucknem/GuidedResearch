import os
import sys
from math import ceil, floor

import numpy as np
import pandas as pd
import seaborn as sns
from bokeh.io import show, output_file
from bokeh.io import export_png, export_svg
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Legend, LabelSet, Label
from bokeh.palettes import Turbo256
from bokeh.plotting import figure

from sklearn.manifold import TSNE
from sklearn.cluster import OPTICS
from sklearn.mixture import BayesianGaussianMixture


def draw_clusters(p, values, clustering):
    colors = Turbo256
    labels = list(set(clustering))
    labels.sort()

    max_num_samples = len(clustering) / len(labels)

    for cluster_label in labels:
        value_indices = np.where(clustering == cluster_label)
        cluster = values[value_indices]
        size = len(cluster)
        if size < 0.1 * max_num_samples:
            continue

        if cluster_label == 0:
            color_index = 0
        else:
            color_index = int((cluster_label) * (255. / (max(clustering))))
        color = colors[color_index]

        cluster = np.transpose(cluster)
        label = str(cluster_label) + " [" + str(size) + "]"
        p.dot(x=cluster[0], y=cluster[1], size=15, alpha=0.5, color=color, legend_label=label)

        mean = np.mean(cluster, axis=1)
        p.cross(x=mean[0], y=mean[1], size=25, alpha=1, color=color, legend_label=label, line_width=5)
        mean = np.round(mean, decimals=3)
        labels = Label(x=mean[0], y=mean[1], text=str(mean[0]) + "," + str(mean[1]), y_offset=5, render_mode='canvas',
                       text_align='left')
        p.add_layout(labels)


def create_scatter_plot(df: pd.DataFrame):
    columns = list(df.columns)
    columns.remove("Run")
    weight_columns = [x for x in columns if "Weights" in str(x)]
    for weight_column in weight_columns:
        columns.remove(weight_column)
    print(columns)
    output_file("layout_grid.html")

    source = ColumnDataSource(df)

    loss_columns = [x for x in columns if str(x).startswith("Loss")]
    non_loss_columns = [x for x in columns if x not in loss_columns]

    num_columns = len(non_loss_columns)
    side_length = int(2000 / num_columns)

    # loss_columns = ["Loss"]
    # non_loss_columns = ["Rotation [z]"]

    plots = []
    for i, yaxis in enumerate(loss_columns):
        for j, xaxis in enumerate(non_loss_columns):
            if xaxis == yaxis:
                continue
            p = figure(tools="pan,wheel_zoom,box_zoom,reset,save")

            # x_values = [[x] for x in df[xaxis].to_numpy()]
            values = np.array([df[xaxis], df[yaxis]])
            values = np.transpose(values)
            x_values = values

            clustering = BayesianGaussianMixture(n_components=25, random_state=42).fit_predict(x_values)
            draw_clusters(p, values, clustering)

            p.xaxis.axis_label = xaxis
            p.yaxis.axis_label = yaxis
            p.xaxis.major_label_orientation = 0.8
            p.legend.click_policy = 'hide'
            plots.append(p)

    grid = gridplot(plots, ncols=num_columns, plot_width=400, plot_height=400, merge_tools=False)
    export_png(grid, filename=os.path.basename(__file__) + ".png")
    show(grid)


def tsne_plot(df: pd.DataFrame):
    X = df.to_numpy()
    X_embedded = TSNE(n_components=2).fit_transform(X)

    p = figure(tools="hover,pan,wheel_zoom,box_zoom,reset", plot_width=1000, plot_height=1000)

    clustering = OPTICS(min_samples=20).fit_predict(X_embedded)
    draw_clusters(p, X_embedded, clustering)

    show(p)


if __name__ == "__main__":
    filename = sys.argv[1]
    if not filename or not filename.endswith('.csv'):
        print("Please specify .csv input file")
        exit(-1)

    df = pd.read_csv(filename)
    df.dropna(axis=1, inplace=True)

    create_scatter_plot(df)
    # tsne_plot(df)
