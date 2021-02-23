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


# check number of classes in the data set
def check_number_of_classes(labels):

    classes = set(labels)
    print("Classes: ", classes)

    return classes


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

    pickle.dump(reducer, open(os.path.join(path, "embedding.pickle"), 'wb'))

    assert (np.all(embedding == reducer.embedding_))

    print("Created embedding.")
    print("Embedding shape: ", embedding.shape)

    return embedding


def create_scatter_plot(file_name, path, labels, classes, n_components, embedding):

    print("Creating plots...")

    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(embedding[:, 0], range(len(embedding)), c=labels, s=1, alpha=0.3)
        ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    if n_components == 2:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=1, alpha=0.3)
        ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=labels, s=1, alpha=0.3)
        ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")

    pre, ext = os.path.splitext(file_name)
    plot_name = pre + ".png"
    plot_name = "_".join(["scatter_plot", plot_name])
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.colorbar(boundaries=np.arange(3)-0.5).set_ticks(np.arange(2))
    mpl.style.use("seaborn")
    plt.title(plot_name.replace(".png", ""), fontsize=24)
    plt.savefig(os.path.join(path, plot_name))
    plt.close()

    for cs in classes:
        fig = plt.figure()
        if n_components == 1:
            ax = fig.add_subplot(111)
            ax.scatter(embedding[np.isclose(labels, cs), 0], range(len(embedding)), s=1, alpha=0.3)
        if n_components == 2:
            ax = fig.add_subplot(111)
            ax.scatter(embedding[np.isclose(labels, cs), 0], embedding[np.isclose(labels, cs), 1], s=1, alpha=0.3)
        if n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(embedding[np.isclose(labels, cs), 0], embedding[np.isclose(labels, cs), 1],
                       embedding[np.isclose(labels, cs), 2], s=1, alpha=0.3)

        plot_name_temp = plot_name.replace("scatter_plot", "_".join(["scatter_plot", str(cs), str(n_components)+"d"]))
        plt.title(plot_name_temp.replace(".png", ""), fontsize=24)
        plt.savefig(os.path.join(path, plot_name_temp))
        plt.close()


def run(args):

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset-filename', dest="dataset_filename", default=None)
    parser.add_argument('-e', '--embedding-filename', dest='embedding_filename', default=None)
    parser.add_argument('-l', '--label-filename', dest="label_filename")
    parser.add_argument('-n', '--n-components', dest="n_components", type=int, default=2)
    parser.add_argument('-s', '--seed', dest="seed", type=int, default=None)

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
    classes = check_number_of_classes(labels)
    # plot embeddings
    create_scatter_plot(file_name, path, labels, classes, params.n_components, embedding)


if __name__ == "__main__":
    run(sys.argv[1:])



