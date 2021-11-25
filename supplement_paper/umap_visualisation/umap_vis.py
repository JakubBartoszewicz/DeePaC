import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib._color_data as mcd
import tensorflow as tf
import random as rn
import umap
import utils
import pickle
import argparse
import sys
import os
from pandas import read_csv


# reshape the data - flatten reads from 3 to 2 dimensions
def reshape_data(data):

    print("Reshaping data...")
    data_reshaped = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))

    return data_reshaped


# runs umap with default parameters
def run_umap(path, file_name, data, n_components, min_dist, n_neighbors, metric, seed):

    print("Running umap...")
    reducer = umap.UMAP(random_state=seed, n_components=n_components, min_dist=min_dist, n_neighbors=n_neighbors,
                        metric=metric)
    reducer.fit(data)

    print("Transforming data...")
    embedding = reducer.transform(data)

    print("Saving the embedding...")
    parameters = "_".join([str(min_dist).replace(".", ""), str(n_neighbors), str(metric), str(n_components) + "d"])
    pickle.dump(reducer, open(os.path.join(path, file_name + "_" + parameters + "_embedding.pickle"), 'wb'),
                protocol=max(4, pickle.DEFAULT_PROTOCOL))

    assert (np.all(embedding == reducer.embedding_))

    print("Created embedding.")
    print("Embedding shape: ", embedding.shape)

    return embedding


def get_mcd_colors(palette):
    if palette == "xkcd":
        return list(mcd.XKCD_COLORS.keys())[::-1]
    elif palette == "base":
        return list(mcd.BASE_COLORS.keys())
    elif palette == "css4":
        return list(mcd.CSS4_COLORS.keys())
    elif palette == "tableau":
        return list(mcd.TABLEAU_COLORS.keys())
    else:
        return palette.split(";")


def get_color_kwargs(ccol, i):
    if ccol is not None:
        return {'color': ccol[i]}
    else:
        return {}


def create_scatter_plot(file_name, path, parameters, labels, classes, n_components, embedding,
                        style="seaborn", no_legend=False, class_names_file=None, alpha=0.8, size=3,
                        custom_colors=None, zero_color=None, fileformat="png", legend_columns=1, legend_loc=None):

    print("Creating plots...")
    plt.style.use(style)
    if custom_colors is not None:
        if zero_color is None:
            ccol = get_mcd_colors(custom_colors)
        else:
            ccol = [zero_color] + get_mcd_colors(custom_colors)
    else:
        ccol = None
    fig = plt.figure()

    print("Creating multiclass plots...")
    if n_components == 1:
        ax = fig.add_subplot(111)
        for cs in classes:
            ax.scatter(embedding[np.isclose(labels, cs), 0], range(len(embedding[np.isclose(labels, cs), 0])), s=size,
                       alpha=alpha, label=cs, **(get_color_kwargs(ccol, cs)))
    if n_components == 2:
        ax = fig.add_subplot(111)
        for cs in classes:
            ax.scatter(embedding[np.isclose(labels, cs), 0], embedding[np.isclose(labels, cs), 1],
                       s=size, alpha=alpha, label=cs, **(get_color_kwargs(ccol, cs)))
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        for cs in classes:
            ax.scatter(embedding[np.isclose(labels, cs), 0], embedding[np.isclose(labels, cs), 1],
                       embedding[np.isclose(labels, cs), 2], s=size, alpha=alpha, label=cs,
                       **(get_color_kwargs(ccol, cs)))

    plot_name = "_".join(["scatter_plot", file_name, parameters]) + "." + fileformat
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.colorbar(boundaries=np.arange(3)-0.5).set_ticks(np.arange(2))
    if not no_legend:
        if class_names_file is not None:
            cust_labels = read_csv(class_names_file).iloc[:, 0].tolist()
            plt.legend(labels=cust_labels, markerscale=2.0, ncol=legend_columns, loc=legend_loc)
        else:
            plt.legend(markerscale=2.0, ncol=legend_columns, loc=legend_loc)
    # plt.title(plot_name.replace(".png", ""), fontsize=12)
    plt.savefig(os.path.join(path, plot_name), dpi=300)
    plt.close()

    print("Creating single-class plots...")
    for cs in classes:
        fig = plt.figure()
        if n_components == 1:
            ax = fig.add_subplot(111)
            ax.scatter(embedding[np.isclose(labels, cs), 0], range(len(embedding[np.isclose(labels, cs), 0])), s=size,
                       alpha=alpha, **(get_color_kwargs(ccol, cs)))
        if n_components == 2:
            ax = fig.add_subplot(111)
            ax.scatter(embedding[np.isclose(labels, cs), 0], embedding[np.isclose(labels, cs), 1],
                       s=size, alpha=alpha, **(get_color_kwargs(ccol, cs)))
        if n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(embedding[np.isclose(labels, cs), 0], embedding[np.isclose(labels, cs), 1],
                       embedding[np.isclose(labels, cs), 2], s=size, alpha=alpha,
                       **(get_color_kwargs(ccol, cs)))

        plot_name_temp = plot_name.replace("scatter_plot", "_".join(["scatter_plot", str(cs)]))
        # plt.title(plot_name_temp.replace(".png", ""), fontsize=12)
        plt.savefig(os.path.join(path, plot_name_temp), dpi=300)
        plt.close()


# generates plots that highlight given classes
def highlight_classes(file_name, path, labels, hl_classes, n_components, embedding,
                      style="seaborn", no_legend=False, class_names_file=None, alpha=0.8, size=3,
                      custom_colors=None, zero_color=None, fileformat="png", legend_columns=1, legend_loc=None):

    plt.style.use(style)
    if custom_colors is not None:
        if zero_color is None:
            ccol = get_mcd_colors(custom_colors)
        else:
            ccol = [zero_color] + get_mcd_colors(custom_colors)
    else:
        ccol = None

    print("Highlighting classes...")
    for ind in hl_classes:

        print("Highlight class: ", str(ind))

        temp_labels = np.array(['other' if str(i) != str(ind) else str(i) for i in labels])

        fig = plt.figure()

        if n_components == 1:
            ax = fig.add_subplot(111)
            for cs in set(temp_labels):
                ax.scatter(embedding[temp_labels == cs, 0], range(len(embedding[temp_labels == cs, 0])), s=size,
                           alpha=alpha, label=cs, **(get_color_kwargs(ccol, cs)))
        if n_components == 2:
            ax = fig.add_subplot(111)
            for cs in set(temp_labels):
                ax.scatter(embedding[temp_labels == cs, 0], embedding[temp_labels == cs, 1],
                           s=size, alpha=alpha, label=cs, **(get_color_kwargs(ccol, cs)))
        if n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            for cs in set(temp_labels):
                ax.scatter(embedding[temp_labels == cs, 0], embedding[temp_labels == cs, 1],
                           embedding[temp_labels == cs, 2], s=size, alpha=alpha, label=cs,
                           **(get_color_kwargs(ccol, cs)))

        plot_name = "hl_class" + str(ind) + "_" + file_name + "." + fileformat
        plot_name = "_".join(["scatter_plot", plot_name])
        # plt.gca().set_aspect('equal', 'datalim')
        # plt.colorbar(boundaries=np.arange(3)-0.5).set_ticks(np.arange(2))
        if not no_legend:
            if class_names_file is not None:
                cust_labels = read_csv(class_names_file).iloc[:, 0].tolist()
                plt.legend(labels=cust_labels, markerscale=2.0, ncol=legend_columns, loc=legend_loc)
            else:
                plt.legend(markerscale=2.0, ncol=legend_columns, loc=legend_loc)
        # plt.title(plot_name.replace(".png", ""), fontsize=12)
        plt.savefig(os.path.join(path, plot_name), dpi=300)
        plt.close()


def run_workflow(dataset_filename, embedding_filename, label_filename, n_components, min_dist, n_neighbors, metric,
                 seed, do_highlight_classes, style, no_legend, class_names_file, alpha, size, custom_colors, zero_color,
                 fileformat, legend_columns, legend_loc=None):

    file_name, path = utils.get_data_name(dataset_filename)

    # set seed
    if seed is not None:
        seed = seed
        np.random.seed(seed)
        tf.random.set_seed(seed)
        rn.seed(seed)

    f_name, ext = os.path.splitext(file_name)

    # if embedding is not available - run umap
    if embedding_filename is None:
        data, labels = utils.read_data_and_labels(dataset_filename, label_filename)
        if len(data.shape) > 2:
            data_reshaped = reshape_data(data)
            embedding = run_umap(path, f_name, data_reshaped, n_components, min_dist, n_neighbors, metric, seed)
        else:
            embedding = run_umap(path, f_name, data, n_components, min_dist, n_neighbors, metric, seed)
    else:  # otherwise read and use ready embedding
        data, labels = utils.read_data_and_labels(dataset_filename, label_filename)
        umap_object = utils.read_embedding(embedding_filename)
        if len(data.shape) > 2:
            data_reshaped = reshape_data(data)
            embedding = umap_object.transform(data_reshaped)
        else:
            embedding = umap_object.transform(data)

    # check the number of classes in labels
    classes = utils.check_number_of_classes(labels)

    np.save(file=os.path.join(path, file_name + "_coords.npy"), arr=embedding)

    # plot embeddings
    if do_highlight_classes is not None:  # plot highlight classes
        hl_classes = do_highlight_classes.split(";")
        highlight_classes(f_name, path, labels, hl_classes, n_components, embedding,
                          style, no_legend, class_names_file, alpha, size, custom_colors, zero_color, fileformat,
                          legend_columns, legend_loc)
    else:  # plot all and separate class plots
        parameters = "_".join([str(min_dist).replace(".", ""), str(n_neighbors), str(metric), str(n_components) + "d"])
        create_scatter_plot(f_name, path, parameters, labels, classes, n_components, embedding,
                            style, no_legend, class_names_file, alpha, size, custom_colors, zero_color, fileformat,
                            legend_columns, legend_loc)


def parse_and_run(args):

    # parsing arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset-filename', dest="dataset_filename", default=None)
    parser.add_argument('-e', '--embedding-filename', dest='embedding_filename', default=None)
    parser.add_argument('-l', '--label-filename', dest="label_filename")
    parser.add_argument('-n', '--n-components', dest="n_components", type=int, default=2)
    parser.add_argument('-m', '--min-dist', dest="min_dist", type=float, default=0.1)
    parser.add_argument('-b', '--n-neighbors', dest="n_neighbors", type=int, default=15)
    parser.add_argument('-t', '--metric', dest="metric", type=str, default='euclidean')
    parser.add_argument('-s', '--seed', dest="seed", type=int, default=None)
    parser.add_argument('-c', '--highlight-classes', dest='highlight_classes', type=str, default=None)
    parser.add_argument('--class-names-file', dest='class_names_file', type=str, default=None)
    parser.add_argument('--style', dest='style', type=str, default="seaborn")
    parser.add_argument('--no-legend', dest='no_legend', action="store_true")
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.8)
    parser.add_argument('--size', dest='size', type=float, default=3)
    parser.add_argument('--custom-color-palette', dest='custom_colors', type=str, default=None)
    parser.add_argument('--zeroth-class-color', dest='zero_color', type=str, default=None)
    parser.add_argument('--format', dest='format', type=str, default="png")
    parser.add_argument('--legend-columns', dest='legend_columns', type=int, default=1)
    parser.add_argument('--legend-loc', dest='legend_loc', type=str, default="best")

    params = parser.parse_args(args)

    run_workflow(params.dataset_filename, params.embedding_filename, params.label_filename, params.n_components,
                 params.min_dist, params.n_neighbors, params.metric, params.seed, params.highlight_classes,
                 params.style, params.no_legend, params.class_names_file, params.alpha, params.size,
                 params.custom_colors, params.zero_color, params.format, params.legend_columns)


if __name__ == "__main__":
    parse_and_run(sys.argv[1:])


