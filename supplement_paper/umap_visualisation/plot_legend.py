import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
from umap_vis import get_mcd_colors, get_color_kwargs
import argparse
import sys
import os
from pandas import read_csv


def plot_legend(class_names_file, out_file, style='seaborn', custom_colors=None, zero_color=None, no_columns=1):
    plt.style.use(style)
    if custom_colors is not None:
        if zero_color is None:
            ccol = get_mcd_colors(custom_colors)
        else:
            ccol = [zero_color] + get_mcd_colors(custom_colors)
    else:
        ccol = None

    labels = read_csv(class_names_file).iloc[:, 0].tolist()
    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [f("s", ccol[i]) for i in range(len(labels))]
    legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=False, ncol=no_columns)

    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(out_file, dpi="figure", bbox_inches=bbox)


def run_plot_legend(args):
    # parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('class_names_file')
    parser.add_argument('out_file')
    parser.add_argument('--style', dest='style', type=str, default="seaborn")
    parser.add_argument('--custom-color-palette', dest='custom_colors', type=str, default=None)
    parser.add_argument('--zeroth-class-color', dest='zero_color', type=str, default=None)
    parser.add_argument('--no-columns', dest='no_columns', type=int, default=1)
    params = parser.parse_args(args)
    plot_legend(params.class_names_file, params.out_file, params.style, params.custom_colors, params.zero_color,
                params.no_columns)


if __name__ == "__main__":
    run_plot_legend(sys.argv[1:])
