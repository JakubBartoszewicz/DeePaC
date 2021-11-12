import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab
from umap_vis import get_mcd_colors, get_color_kwargs
import argparse
import sys
from pandas import read_csv


def plot_legend(class_names_file, out_file, style='seaborn', custom_colors=None, zero_color=None, no_columns=1,
                width=None, height=None):
    plt.style.use(style)
    if custom_colors is not None:
        if zero_color is None:
            ccol = get_mcd_colors(custom_colors)
        else:
            ccol = [zero_color] + get_mcd_colors(custom_colors)
    else:
        ccol = None

    labels = read_csv(class_names_file).iloc[:, 0].tolist()

    fig = pylab.figure()
    figsize = None if (width is None or height is None) else (width, height)
    figlegend = pylab.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    f = lambda m, c: ax.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [f("o", ccol[i]) for i in range(len(labels))]

    figlegend.legend(handles, labels, ncol=no_columns)
    bbox = figlegend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    figlegend.savefig(out_file, bbox_inches=bbox, dpi=300, transparent=True)


def run_plot_legend(args):
    # parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('class_names_file')
    parser.add_argument('out_file')
    parser.add_argument('--style', dest='style', type=str, default="seaborn")
    parser.add_argument('--custom-color-palette', dest='custom_colors', type=str, default=None)
    parser.add_argument('--zeroth-class-color', dest='zero_color', type=str, default=None)
    parser.add_argument('--no-columns', dest='no_columns', type=int, default=1)
    parser.add_argument('-x', '--width', dest='width', type=float, default=None)
    parser.add_argument('-y', '--height', dest='height', type=float, default=None)
    params = parser.parse_args(args)
    plot_legend(params.class_names_file, params.out_file, params.style, params.custom_colors, params.zero_color,
                params.no_columns, params.width, params.height)


if __name__ == "__main__":
    run_plot_legend(sys.argv[1:])
