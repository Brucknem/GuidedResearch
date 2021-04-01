import math
import os
from bokeh.models import ColumnDataSource, LabelSet, Whisker
from commons import *

display = True

folder = sys.argv[1]
if not folder or not Path(folder).is_dir():
    print("Please specify an input directory containing the pixel paths as .csv")
    exit(-1)


def get_output_filename(foldername, filename):
    directory = Path(join('evaluate_object_tracking', foldername))
    directory.mkdir(exist_ok=True, parents=True)
    return str(join(directory, filename + ".html"))


def setup(foldername, filename, title):
    output_file(get_output_filename(foldername, filename + "_" + title))

    p = figure(plot_width=plot_width, plot_height=plot_height, tools=tools)
    p.title.text = title + ' of ' + filename.replace('.csv', '') + ' [' + title_suffix + ']'

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
    output_file(get_output_filename(foldername, filename + "_" + title))

    curvature_plot = figure(
        x_range=column_names,
        plot_width=plot_width,
        plot_height=plot_height,
        tools=tools,
    )
    curvature_plot.title.text = title + ' of ' + filename.replace('.csv', '') + ' [' + title_suffix + ']'

    means = []
    for i, name in enumerate(column_names):
        xes, yes = get_values(df, name)
        total = []
        for run in range(len(xes)):
            dist = distances(list(zip(xes[run], yes[run])))
            total.append(np.nansum(dist))
        means.append(np.mean(total))

    means = np.array([mean / means[0] if means[0] > 0 else mean for mean in means])

    source = ColumnDataSource(data=dict(x=column_names, means=means, color=get_colors(len(column_names))))
    source.data['means_formatted'] = [round(mean, 3) for mean in means]

    curvature_plot.vbar(x='x', top='means', width=0.9, color='color', legend_field='x', fill_alpha=0.5, source=source)

    labels = LabelSet(x='x', y='means', text='means_formatted',
                      y_offset=5, source=source, render_mode='canvas', text_align='center', text_font=font,
                      text_font_size=font_size)
    curvature_plot.add_layout(labels)

    set_plot_settings(curvature_plot)
    curvature_plot.legend.items = []
    curvature_plot.xaxis.major_label_orientation = math.pi / 4

    show_or_save(curvature_plot, display)


def generate_plots(df, foldername, filename):
    df.dropna(how='all', axis=1, inplace=True)
    column_names = list(set([str(name).split(' [')[0] for name in df.columns]))
    column_names.remove("Frame")
    column_names.remove("Original")
    column_names.sort()
    column_names = ["Original", *column_names]
    # column_names = ["Original", "FAST", "ORB", "SURF", "FAST (No skew)", "ORB (No skew)", "SURF (No skew)"]
    generate_xy_plots(df, foldername, filename, column_names)
    generate_curvature_plots(df, foldername, filename, column_names)


window = 1

for csv in os.listdir(folder):
    if csv.endswith(".csv"):
        print(os.path.join(folder, csv))
        df = pd.read_csv(os.path.join(folder, csv))
        generate_plots(df, Path(folder).name, csv)
