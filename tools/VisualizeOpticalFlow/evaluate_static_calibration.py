import os
import sys
from math import ceil, floor

import numpy as np
import pandas as pd
import seaborn as sns
from bokeh.io import show, output_file
from bokeh.io import export_png, export_svg
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Legend, LabelSet, Label, Whisker, Span, Range1d, LinearAxis
from bokeh.palettes import Turbo256, Turbo3, Turbo4
from bokeh.plotting import figure
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

from sklearn.manifold import TSNE
from sklearn.cluster import OPTICS
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from commons import *


def get_output_filename(foldername, filename, suffix):
    directory = Path(join('evaluate_static_calibration', foldername))
    directory.mkdir(exist_ok=True, parents=True)
    return str(join(directory, filename + "." + suffix))


def create_outlier_mask(_values):
    values = np.array(_values)
    if len(values.shape) == 1:
        values = values.reshape(-1, 1)
    unique = np.unique(values)
    if len(unique) == 1:
        return [True] * len(values)
    # iso = LocalOutlierFactor()
    iso = OneClassSVM()
    yhat = iso.fit_predict(values)
    inliers = yhat != -1
    return inliers, np.sum(inliers) / len(values)


def add_label(p, mean):
    mean = np.round(mean, decimals=8)
    labels = Label(x=mean[0], y=mean[1], text=str(mean[0]), y_offset=-17, x_offset=0, render_mode='canvas',
                   text_align='left', background_fill_color='white', background_fill_alpha=0.8)
    p.add_layout(labels)
    labels = Label(x=mean[0], y=mean[1], text=str(mean[1]), y_offset=-34, x_offset=0, render_mode='canvas',
                   text_align='left', background_fill_color='white', background_fill_alpha=0.8)
    p.add_layout(labels)


def add_spans(p, mean, std, mins, maxs):
    for val in zip(
            [mins, maxs, mean, np.array(mean) - std, np.array(mean) + std],
            [1, 1, 3, 2, 2]):
        p.add_layout(
            Span(location=val[0][0], dimension='height', line_color='black', line_dash='dashed', line_width=val[1]))


def add_whisker(p, mean, std, label):
    # whisker_colors = ["green", "yellow", "red"]
    whisker_colors = ["black"]
    label_conf = label + " [conf]"
    p.cross(x=mean[0], y=mean[1], size=25, alpha=1, color=whisker_colors[0], legend_label=label_conf, line_width=5)
    # for i in range(3, 0, -1):
    deviations = 1
    for i in range(1, 0, -1):
        p.line(x=[mean[0] - i * std[0], mean[0] + i * std[0]], y=[mean[1], mean[1]], line_alpha=0.75,
               color=whisker_colors[i - 1], legend_label=label_conf, line_width=2 * (deviations + 1 - i))


def create_result(cluster_label, size, total_size, mean, std, mins, maxs):
    return {
        'Cluster/ID': str(cluster_label),
        'Cluster/Size': size,
        'Cluster/Fraction': total_size,
        'Cluster/Mean [x]': mean[0],
        'Cluster/Std [x]': std[0],
        'Cluster/Min [x]': mins[0],
        'Cluster/Max [x]': maxs[0],
        'Cluster/Mean [y]': mean[1],
        'Cluster/Std [y]': std[1],
        'Cluster/Min [y]': mins[1],
        'Cluster/Max [y]': maxs[1],
    }


def draw_clusters(p, values, clustering):
    colors = Turbo256
    labels = list(set(clustering))
    labels.sort()

    max_num_samples = len(clustering) / len(labels)
    unique, count = np.unique(clustering, return_counts=True)
    unique = unique[np.where(count >= max_num_samples)]
    labels = unique

    # order = np.argsort(count)
    # unique = unique[order]
    # labels = unique[-min(2, len(unique)):]

    result = []
    color_it = 0
    for cluster_label in labels:
        value_indices = np.where(clustering == cluster_label)
        cluster = values[value_indices]
        size = len(cluster)
        if size < 0.1 * max_num_samples:
            continue

        color_index = int(color_it * (255. / (len(labels))))
        color = colors[color_index]
        color_it += 1
        cluster = np.transpose(cluster)
        label = str(cluster_label) + " [" + str(size) + "]"
        p.dot(x=cluster[0], y=cluster[1], size=15, alpha=0.5, color=color, legend_label=label)

        mean = np.mean(cluster, axis=1)
        std = np.std(cluster, axis=1)
        mins = [min(cluster[0]), min(cluster[1])]
        maxs = [max(cluster[0]), max(cluster[1])]
        result.append(create_result(cluster_label, size, size / len(clustering), mean, std, mins, maxs))

        add_spans(p, mean, std, mins, maxs)
        add_whisker(p, mean, std, label)

    return result


def setup_pairplots(df):
    columns = list(df.columns)
    columns.remove("Run")
    columns.remove("Correspondences")
    columns.remove("Penalize Scale [Lambdas]")
    columns.remove("Penalize Scale [Rotation]")
    columns.remove("Loss [Rotations]")
    weight_columns = [x for x in columns if "Weights" in str(x)]
    for weight_column in weight_columns:
        columns.remove(weight_column)

    global filename
    foldername = Path(filename).parent.name

    source = ColumnDataSource(df)

    loss_columns = [x for x in columns if str(x).startswith("Loss")]
    non_loss_columns = [x for x in columns if x not in loss_columns]

    num_columns = len(non_loss_columns)
    side_length = int(2000 / num_columns)

    # loss_columns = ["Loss"]
    # non_loss_columns = ["Rotation [z]"]
    return loss_columns, non_loss_columns, foldername


def finalize_pairplot(plots, foldername, result_data_rows):
    output_file(get_output_filename(foldername, 'pairplot', 'html'))
    grid = gridplot(plots, ncols=len(result_data_rows[0]), plot_width=plot_height, plot_height=plot_height,
                    merge_tools=False)
    export_png(grid, filename=get_output_filename(foldername, 'pairplot', 'png'))
    result_df = pd.DataFrame(result_data_rows)
    result_df.to_csv(get_output_filename(foldername, 'clustering', 'csv'))
    show(grid)


def create_scatter_plot(df: pd.DataFrame):
    loss_columns, non_loss_columns, foldername = setup_pairplots(df)

    result_data_rows = []

    plots = []
    for i, yaxis in enumerate(loss_columns):
        for j, xaxis in enumerate(non_loss_columns):
            if xaxis == yaxis:
                continue
            output_filename = str(xaxis).strip().replace(" ", "") + "_vs_" + str(yaxis).strip().replace(" ", "")
            output_file(get_output_filename(foldername, output_filename, 'html'))
            p = figure(tools="pan,wheel_zoom,box_zoom,reset,save", plot_width=plot_height, plot_height=plot_height)

            values = np.array([df[xaxis], df[yaxis]])
            values = np.transpose(values)
            # mask, confidence = create_outlier_mask(df[yaxis])
            conf = 1
            for _ in range(1):
                mask, confidence = create_outlier_mask(values[:, 1].reshape(-1, 1))
                values = values[mask]
                conf *= confidence

            remove_distance = 20
            order = np.argsort(values[:, 0])
            values = values[order[remove_distance: -remove_distance]]

            # cluster_input = values
            # clustering = BayesianGaussianMixture(n_components=2, random_state=42).fit_predict(cluster_input)
            clustering = np.array([0] * len(values))
            cluster_results = draw_clusters(p, values, clustering)
            for cluster_result in cluster_results:
                result_data_rows.append({
                    'X': xaxis,
                    'Y': yaxis,
                    'Size Total': len(df),
                    'Inlier Fraction': conf,
                    **cluster_result
                })

            set_plot_settings(p)
            p.legend.items = []
            p.xaxis.formatter.use_scientific = False
            p.xaxis.axis_label = xaxis
            p.yaxis.axis_label = yaxis
            p.xaxis.major_label_orientation = math.pi / 8
            plots.append(p)
            show(p)
            export_png(p, filename=get_output_filename(foldername, output_filename, 'png'))
    finalize_pairplot(plots, foldername, result_data_rows)


def create_loss_compare_plot(df: pd.DataFrame):
    loss_columns, non_loss_columns, foldername = setup_pairplots(df)

    result_data_rows = []
    yaxis = ["Loss [Correspondences]", "Loss [Lambdas]"]

    plots = []
    for j, xaxis in enumerate(non_loss_columns):
        output_filename = str(xaxis).strip().replace(" ", "") + "_vs_" + str(yaxis[0]).strip().replace(" ", "") + \
                          "_vs_" + str(yaxis[1]).strip().replace(" ", "")
        output_file(get_output_filename(foldername, output_filename, 'html'))

        total_loss = np.array(df["Loss"])
        values = np.array([df[xaxis], df[yaxis[0]], df[yaxis[1]]])
        values = np.transpose(values)
        conf = 1
        for _ in range(1):
            mask, confidence = create_outlier_mask(total_loss)
            values = values[mask]
            conf *= confidence

        remove_distance = 20
        order = np.argsort(values[:, 0])
        values = values[order[remove_distance: -remove_distance]]
        values = np.transpose(values)

        p = figure(tools="pan,wheel_zoom,box_zoom,reset,save", plot_width=plot_height, plot_height=plot_height,
                   y_range=(min(values[1]), max(values[1])))
        p.extra_y_ranges = {yaxis[1]: Range1d(start=0, end=1)}
        p.add_layout(LinearAxis(y_range_name=yaxis[1]), 'right')

        p.dot(x=values[0], y=values[1], size=15, alpha=0.5, color='firebrick', legend_label=yaxis[1])
        p.dot(x=values[0], y=values[2], size=15, alpha=0.5, color='darkblue', legend_label=yaxis[1],
              y_range_name=yaxis[1])

        mean = np.mean(values, axis=1)
        std = np.std(values, axis=1)
        mins = [min(values[0]), min(values[1]), min(values[2])]
        maxs = [max(values[0]), max(values[1]), max(values[2])]

        result_data_rows.append({
            'X': xaxis,
            'Y': yaxis,
            'Size Total': len(df),
            'Inlier Fraction': conf,
            **create_result(None, len(values), len(values), mean, std, mins, maxs)
        })
        add_spans(p, mean, std, mins, maxs)

        set_plot_settings(p)
        p.legend.items = []
        p.xaxis.formatter.use_scientific = False
        p.xaxis.axis_label = xaxis
        p.yaxis[0].axis_label = yaxis[0]
        p.yaxis[1].axis_label = yaxis[1]
        p.xaxis.major_label_orientation = math.pi / 8
        plots.append(p)
        # show(p)
        export_png(p, filename=get_output_filename(foldername, output_filename, 'png'))
    finalize_pairplot(plots, foldername, result_data_rows)


def tsne_plot(df: pd.DataFrame):
    X = df.to_numpy()
    X_embedded = TSNE(n_components=2).fit_transform(X)

    p = figure(tools="hover,pan,wheel_zoom,box_zoom,reset", plot_width=1000, plot_height=1000)

    clustering = OPTICS(min_samples=20).fit_predict(X_embedded)
    draw_clusters(p, X_embedded, clustering)

    show(p)


def find_best_lambda_rotation_combination(df: pd.DataFrame):
    loss_column = df["Loss"].to_numpy()[:, np.newaxis]

    clustering = BayesianGaussianMixture(n_components=100, random_state=42).fit_predict(loss_column)
    unique, counts = np.unique(clustering, return_counts=True)
    relevant_indices = np.where(clustering == unique[np.argmax(counts)])
    relevant_values = loss_column[relevant_indices]
    #
    # clustering = BayesianGaussianMixture(n_components=100, random_state=42).fit_predict(relevant_values)
    # unique, counts = np.unique(clustering, return_counts=True)
    # relevant_indices = np.where(clustering == unique[np.argmax(counts)])
    # relevant_values = relevant_values[relevant_indices]

    mean = np.mean(relevant_values)
    std = np.std(relevant_values)

    n_std = 3
    filtered = df[df["Loss"] >= mean - n_std * std]
    filtered = filtered[filtered["Loss"] <= mean + n_std * std]

    group = filtered.groupby(["Penalize Scale [Lambdas]", "Penalize Scale [Rotation]"]).agg(['count', 'mean', 'std'])
    losses = group[["Loss", "Loss [Correspondences]", "Loss [Lambdas]", "Loss [Rotations]"]]
    losses_sorted = losses.sort_values(
        by=[("Loss", 'count'), ("Loss [Correspondences]", "mean"), ("Loss [Lambdas]", "mean")], ascending=False)
    print(group["Loss"])


if __name__ == "__main__":
    create_loss_compare_plot(df)
    # create_scatter_plot(df)
    # find_best_lambda_rotation_combination(df)
    # tsne_plot(df)
