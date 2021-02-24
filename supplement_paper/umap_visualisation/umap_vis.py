import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import random as rn
import umap
import utils
import pickle
import argparse
import sys
import os


# reshape the data - flatten reads from 3 to 2 dimensions
def reshape_data(data):

    print("Reshaping data...")
    data_reshaped = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))

    return data_reshaped


# runs umap with default parameters
def run_umap(path, data, n_components, seed):

    print("Running umap...")
    reducer = umap.UMAP(random_state=seed, n_components=n_components)
    reducer.fit(data)

    print("Transforming data...")
    embedding = reducer.transform(data)

    pickle.dump(reducer, open(os.path.join(path, str(n_components)+"d_embedding.pickle"), 'wb'))

    assert (np.all(embedding == reducer.embedding_))

    print("Created embedding.")
    print("Embedding shape: ", embedding.shape)

    return embedding


def create_scatter_plot(file_name, path, labels, classes, n_components, embedding):

    print("Creating plots...")
    mpl.style.use("seaborn")

    fig = plt.figure()

    if n_components == 1:
        ax = fig.add_subplot(111)
        for cs in classes:
            ax.scatter(embedding[np.isclose(labels, cs), 0], range(len(embedding)), s=1, alpha=0.3)
    if n_components == 2:
        ax = fig.add_subplot(111)
        for cs in classes:
            ax.scatter(embedding[np.isclose(labels, cs), 0], embedding[np.isclose(labels, cs), 1],
                       s=1, alpha=0.3, label=cs)
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        for cs in classes:
            ax.scatter(embedding[np.isclose(labels, cs), 0], embedding[np.isclose(labels, cs), 1],
                       embedding[np.isclose(labels, cs), 2], s=1, alpha=0.3, label=cs)

    pre, ext = os.path.splitext(file_name)
    plot_name = pre + ".png"
    plot_name = "_".join(["scatter_plot", plot_name])
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.colorbar(boundaries=np.arange(3)-0.5).set_ticks(np.arange(2))
    plt.legend(markerscale=5.0)
    plt.title(plot_name.replace(".png", ""), fontsize=18)
    plt.savefig(os.path.join(path, plot_name))
    plt.close()

    for cs in classes:
        fig = plt.figure()
        if n_components == 1:
            ax = fig.add_subplot(111)
            ax.scatter(embedding[np.isclose(labels, cs), 0], range(len(embedding)), s=1, alpha=0.3)
        if n_components == 2:
            ax = fig.add_subplot(111)
            ax.scatter(embedding[np.isclose(labels, cs), 0], embedding[np.isclose(labels, cs), 1],
                       s=1, alpha=0.3)
        if n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(embedding[np.isclose(labels, cs), 0], embedding[np.isclose(labels, cs), 1],
                       embedding[np.isclose(labels, cs), 2], s=1, alpha=0.3)

        plot_name_temp = plot_name.replace("scatter_plot", "_".join(["scatter_plot", str(cs), str(n_components)+"d"]))
        plt.title(plot_name_temp.replace(".png", ""), fontsize=18)
        plt.savefig(os.path.join(path, plot_name_temp))
        plt.close()


# generates plots that highlight given classes
def highlight_classes(file_name, path, labels, hl_classes, n_components, embedding):

    mpl.style.use("seaborn")

    for ind in hl_classes:

        print("Highlight class: ", str(ind))

        temp_labels = np.array(['other' if str(i) != str(ind) else str(i) for i in labels])

        fig = plt.figure()

        if n_components == 1:
            ax = fig.add_subplot(111)
            for cs in set(temp_labels):
                ax.scatter(embedding[temp_labels == cs, 0], range(len(embedding)), s=1, alpha=0.3, label=cs)
        if n_components == 2:
            ax = fig.add_subplot(111)
            for cs in set(temp_labels):
                ax.scatter(embedding[temp_labels == cs, 0], embedding[temp_labels == cs, 1],
                           s=1, alpha=0.3, label=cs)
        if n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            for cs in set(temp_labels):
                ax.scatter(embedding[temp_labels == cs, 0], embedding[temp_labels == cs, 1],
                           embedding[temp_labels == cs, 2], s=1, alpha=0.3, label=cs)

        pre, ext = os.path.splitext(file_name)
        plot_name = "hl_class" + str(ind) + "_" + pre + ".png"
        plot_name = "_".join(["scatter_plot", plot_name])
        # plt.gca().set_aspect('equal', 'datalim')
        # plt.colorbar(boundaries=np.arange(3)-0.5).set_ticks(np.arange(2))
        plt.legend(markerscale=5.0)
        plt.title(plot_name.replace(".png", ""), fontsize=18)
        plt.savefig(os.path.join(path, plot_name))
        plt.close()


def run(args):

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset-filename', dest="dataset_filename", default=None)
    parser.add_argument('-e', '--embedding-filename', dest='embedding_filename', default=None)
    parser.add_argument('-l', '--label-filename', dest="label_filename")
    parser.add_argument('-n', '--n-components', dest="n_components", type=int, default=2)
    parser.add_argument('-s', '--seed', dest="seed", type=int, default=None)
    parser.add_argument('-c', '--highlight_classes', dest='highlight_classes', type=str, default=None)

    params = parser.parse_args(args)
    file_name, path = utils.get_data_name(params.dataset_filename)

    # set seed
    if params.seed is not None:
        seed = params.seed
        np.random.seed(seed)
        tf.random.set_seed(seed)
        rn.seed(seed)

    # if embedding is not available - run umap
    if params.embedding_filename is None:
        data, labels = utils.read_data_and_labels(params.dataset_filename, params.label_filename)
        data_reshaped = reshape_data(data)
        embedding = run_umap(path, data_reshaped, params.n_components, params.seed)
    else:  # otherwise read and use ready embedding
        data, labels = utils.read_data_and_labels(params.dataset_filename, params.label_filename)
        umap_object = utils.read_embedding(params.embedding_filename)
        data_reshaped = reshape_data(data)
        embedding = umap_object.transform(data_reshaped)

    # check the number of classes in labels
    classes = utils.check_number_of_classes(labels)
    # classes, converted_labels = utils.convert_labels(labels)
    # plot embeddings
    if params.highlight_classes is not None:
        hl_classes = params.highlight_classes.split(";")
        highlight_classes(file_name, path, labels, hl_classes, params.n_components, embedding)
    else:
        create_scatter_plot(file_name, path, labels, classes, params.n_components, embedding)


if __name__ == "__main__":
    run(sys.argv[1:])



