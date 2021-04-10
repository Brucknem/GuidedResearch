import math
import os
from bokeh.models import ColumnDataSource, LabelSet, Whisker, FactorRange
from bokeh.transform import dodge

from commons import *

display = True


def get_output_filename(foldername, filename):
    directory = Path(join('evaluate_object_tracking', foldername))
    directory.mkdir(exist_ok=True, parents=True)
    return str(join(directory, filename + ".html"))


def setup(foldername, filename, title):
    output_file(get_output_filename(foldername, filename + "_" + title))

    p = figure(plot_width=plot_width, plot_height=plot_height, tools=tools)
    p.title.text = title + ' of ' + filename.replace('.csv', '') + ' [' + get_title_suffix() + ']'

    return p


def get_values(df, name):
    xes = []
    yes = []

    for column in df.columns:
        if str(column).startswith(name + " [mx]"):
            xes.append(df[name + " [mx]"])
        if str(column).startswith(name + " [my]"):
            yes.append(df[name + " [my]"])

    for i in range(len(xes)):
        indices = xes[i] >= 0
        xes[i] = xes[i][indices]
        yes[i] = yes[i][indices]

        xes[i] = xes[i].rolling(window).mean()
        yes[i] = yes[i].rolling(window).mean()

    return np.array(xes), np.array(yes)


def generate_xy_plots(df, foldername, filename, column_names):
    xy_plot = setup(foldername, filename, "Tracking")
    colors = get_colors(len(column_names))
    for i, name in enumerate(column_names):
        xes, yes = get_values(df, name)
        x = np.mean(xes, axis=0)
        y = np.mean(yes, axis=0)
        xy_plot.line(x, y, line_width=3, color=colors[i], alpha=0.65, legend_label=name)
        xy_plot.varea(x=x, y1=y, y2=y, color=colors[i], alpha=0.2, legend_label=name)

    xy_plot.y_range.flipped = True
    xy_plot.xaxis.axis_label = "X [px]"
    xy_plot.yaxis.axis_label = "Y [px]"

    set_plot_settings(xy_plot)
    show_or_save(xy_plot, display)


def distances(values):
    return np.array([np.linalg.norm(np.array(values[i]) - np.array(values[i - 1])) for i in range(1, len(values))])


def generate_curvature_plots(df, foldername, filename, column_names):
    title = "Normalized arc length"
    output_file(get_output_filename(foldername, 'normalized_arc_lengths'))
    data = {
        'Vehicles': vehicles
    }

    for vehicle in vehicles:
        df_vehicle = df[df["Vehicle"] == vehicle]
        for stabilizer in column_names:
            xes, yes = get_values(df_vehicle, stabilizer)
            total = []
            for run in range(len(xes)):
                dist = distances(list(zip(xes[run], yes[run])))
                total.append(np.nansum(dist))

            if stabilizer not in data:
                data[stabilizer] = []
            data[stabilizer].append(np.mean(total))

    data_original = np.array(data["Original"])
    for stabilizer in column_names:
        data[stabilizer] = np.array(data[stabilizer]) / data_original
        data[stabilizer + formatted_suffix] = [str(round(val, 2)) for val in data[stabilizer]]

    source = ColumnDataSource(data=data)
    p = figure(
        x_range=FactorRange(*vehicles),
        y_range=(-0.03, 1.20),
        plot_width=plot_width,
        plot_height=plot_height,
        tools=tools,
    )
    p.title.text = title + ' of ' + filename.replace('.csv', '') + ' [' + get_title_suffix() + ']'

    colors = get_colors(len(column_names))
    # colors = colors[]
    colors = colors * 4

    distance = 0.9 / len(column_names)
    for i, stabilizer in enumerate(column_names):
        xdodge = dodge('Vehicles', -0.34 + i * distance, range=p.x_range)
        p.vbar(x=xdodge, top=stabilizer, width=distance * 0.9, source=source,
               color=colors[i], legend_label=stabilizer, alpha=0.7)
        labels = LabelSet(x=xdodge, y=stabilizer, text=stabilizer + formatted_suffix,
                          y_offset=5, source=source, render_mode='canvas', text_align='center', text_font=font,
                          text_font_size=tick_font_size)
        p.add_layout(labels)
    set_plot_settings(p)
    p.xgrid.grid_line_alpha = 0

    show_or_save(p, display)


window = 1

df.dropna(how='all', axis=1, inplace=True)
column_names = list(set([str(name).split(' [')[0] for name in df.columns]))
vehicles = list(set(df["Vehicle"]))
foldername = Path(filename).parent.name
filename = Path(filename).name

column_names.remove("Frame")
column_names.remove("Vehicle")
column_names.remove("Original")
column_names.sort()
column_names = ["Original", *column_names]
# column_names = ["Original", "FAST", "ORB", "SURF", "FAST (No skew)", "ORB (No skew)", "SURF (No skew)"]
for vehicle in vehicles:
    df_vehicle = df[df["Vehicle"] == vehicle]
    generate_xy_plots(df_vehicle, foldername, vehicle, column_names)

generate_curvature_plots(df, foldername, "tracking", column_names)
