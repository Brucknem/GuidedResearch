import os
from bokeh.models import ColumnDataSource
from commons import *

display = False

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
    # p.add_layout(Legend(), 'right')

    return p


def get_values(df, name):
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
    return x, y


def generate_xy_plots(df, foldername, filename, column_names):
    xy_plot = setup(foldername, filename, "Tracking")
    colors = get_colors(len(column_names))
    for i, name in enumerate(column_names):
        x, y = get_values(df, name)
        xy_plot.line(x, y, line_width=2, color=colors[i], alpha=0.5, legend_label=name)
        xy_plot.varea(x=x, y1=y, y2=y, color=colors[i], alpha=0.2, legend_label=name)

    xy_plot.y_range.flipped = True
    xy_plot.xaxis.axis_label = "X [px]"
    xy_plot.yaxis.axis_label = "Y [px]"

    set_plot_settings(xy_plot)
    show_or_save(xy_plot, display)


def distances(values):
    return [np.linalg.norm(np.array(values[i]) - np.array(values[i - 1])) for i in range(1, len(values))]


def generate_curvature_plots(df, foldername, filename, column_names):
    title = "Arc length"
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
        x, y = get_values(df, name)
        dist = distances(list(zip(x, y)))
        mean = np.sum(dist)
        means.append(mean)
    means = [mean / means[0] for mean in means]

    source = ColumnDataSource(data=dict(x=column_names, means=means, color=get_colors(len(column_names))))
    curvature_plot.vbar(x='x', top='means', width=0.9, color='color', legend_field='x', fill_alpha=0.5, source=source)

    set_plot_settings(curvature_plot)
    show_or_save(curvature_plot, display)


def generate_plots(df, foldername, filename):
    column_names = ["Original", "FAST", "ORB", "SURF"]
    generate_xy_plots(df, foldername, filename, column_names)
    generate_curvature_plots(df, foldername, filename, column_names)


window = 1

for csv in os.listdir(folder):
    if csv.endswith(".csv"):
        print(os.path.join(folder, csv))
        df = pd.read_csv(os.path.join(folder, csv))
        generate_plots(df, Path(folder).name, csv)
