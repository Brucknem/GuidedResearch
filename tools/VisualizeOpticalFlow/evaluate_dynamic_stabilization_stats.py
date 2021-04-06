from bokeh.core.property.dataspec import value
from bokeh.models import Legend, ColumnDataSource, FactorRange, LabelSet, NumeralTickFormatter
from bokeh.transform import factor_cmap, dodge

from commons import *

display = True

filename = sys.argv[1]
if not filename or not filename.endswith('.csv'):
    print("Please specify .csv input file")
    exit(-1)

df = pd.read_csv(filename)
# df = df[df["Stabilizer"] != "Original"]

formatted_suffix = "_formatted"

cameras = list(set(df["Camera"]))
cameras.sort()
stabilizers = list(set(df["Stabilizer"]))
stabilizers.sort()
stabilizers.remove("Original")
stabilizers = ["Original", *stabilizers]
data = {
    'Cameras': cameras,
}
for name in stabilizers:
    data[name] = df[df["Stabilizer"] == name]["Better [frac]"]
    data[name + formatted_suffix] = [str(round(val * 100, 1)) + "%" for val in data[name]]

source = ColumnDataSource(data=data)


def get_output_filename():
    folder = Path(join('evaluate_dynamic_stabilization'))
    folder.mkdir(exist_ok=True, parents=True)
    return str(join(folder, "stats.html"))


def setup():
    output_file(get_output_filename())

    p = figure(x_range=FactorRange(*source.data["Cameras"]),
               y_range=(-0.03, 1.07),
               plot_width=plot_width,
               plot_height=plot_height,
               tools=tools)
    p.title.text = 'Percentage of more stable frames after dynamic stabilization'
    # p.add_layout(Legend(), 'right')

    return p


def plot():
    p = setup()

    colors = get_colors(len(stabilizers))
    # colors = colors[]
    colors = colors * 4

    distance = 0.9 / len(stabilizers)

    for i, stabilizer in enumerate(stabilizers):
        xdodge = dodge('Cameras', -0.34 + i * distance, range=p.x_range)
        p.vbar(x=xdodge, top=stabilizer, width=0.2, source=source,
               color=colors[i], legend_label=stabilizer, alpha=0.7)
        labels = LabelSet(x=xdodge, y=stabilizer, text=stabilizer + formatted_suffix,
                          y_offset=5, source=source, render_mode='canvas', text_align='center', text_font=font,
                          text_font_size=tick_font_size)
        p.add_layout(labels)

    p.yaxis.formatter = NumeralTickFormatter(format='0%')

    set_plot_settings(p)
    p.yaxis.axis_label = "More stable frames"
    p.xgrid.grid_line_alpha = 0
    # p.legend.visible = False
    # p.xaxis.major_label_orientation = np.deg2rad(45)
    # p.xaxis[0].major_label_orientation = np.deg2rad(45)

    # p.xaxis.axis_label_rotation = "vertical"
    show_or_save(p, display)


if __name__ == "__main__":
    plot()
