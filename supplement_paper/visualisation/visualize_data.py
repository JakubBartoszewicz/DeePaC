import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
import random as rn
import umap
import os
import argparse
import sys
from datetime import datetime

seed = 1

np.random.seed(seed)
tf.random.set_seed(seed)
rn.seed(seed)


def get_data_name(data_path):
    path_splitted = os.path.split(data_path)
    path = path_splitted[0]
    file_name = path_splitted[1]

    return file_name, path


def read_data_and_labels(data_path, label_path):
    print("Reading data...")
    try:
        data = np.load(data_path)  # read the data
        labels = np.load(label_path)
        print("Data succesfully read.")

    except IOError:
        print("Error: ", data_path, " or ", label_path, " not found.")
        sys.exit(0)

    return data, labels


def check_number_of_classes(labels):
    classes = set(labels)
    print("Classes: ", classes)

    return classes


def reshape_data(data):
    print("Reshaping data...")
    data_reshaped = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))

    return data_reshaped


def run_umap(data, n_components):
    print("Running umap...")
    reducer = umap.UMAP(random_state=42, n_components=n_components)
    reducer.fit(data)

    embedding = reducer.transform(data)

    assert (np.all(embedding == reducer.embedding_))

    print("Created embedding.")
    print("Embedding shape: ", embedding.shape)

    return embedding


def create_scatter_plot(file_name, path, labels, classes, n_components, embedding):
    print("Creating plots...")

    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(embedding[:, 0], range(len(embedding)), c=labels, s=1, alpha=0.3)
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=1, alpha=0.3)
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=labels, s=1, alpha=0.3)

    plot_name = "_".join(["scatter_plot", file_name.replace(".npy", ".png")])
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.colorbar(boundaries=np.arange(3)-0.5).set_ticks(np.arange(2))
    plt.title(plot_name.replace(".png", ""), fontsize=24);
    plt.savefig("/".join([path, plot_name]))
    plt.close()

    for c in classes:
        plot_name_temp = plot_name.replace("scatter_plot", ("_").join(["scatter_plot", str(c)]))
        plt.scatter(embedding[np.isclose(labels, c), 0], embedding[np.isclose(labels, c), 1],
                    embedding[np.isclose(labels, c), 2], s=1)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title(plot_name_temp.replace(".png", ""), fontsize=24);
        plt.savefig("/".join([path, plot_name_temp]))
        plt.close()


def run(args):
    date = datetime.now()
    date_now = date.strftime("%Y-%m-%d_%H-%M-%S")

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset-filename', dest="dataset_filename")
    parser.add_argument('-l', '--label-filename', dest="label_filename")
    parser.add_argument('-n', '--n-components', dest="n_components", type=int, default=2)

    params = parser.parse_args(args)

    file_name, path = get_data_name(params.dataset_filename)
    data, labels = read_data_and_labels(params.dataset_filename, params.label_filename)
    classes = check_number_of_classes(labels)
    data_reshaped = reshape_data(data)
    embedding = run_umap(data_reshaped, params.n_components)
    file_name = "_".join([date_now, file_name])
    create_scatter_plot(file_name, path, labels, classes, params.n_components, embedding)


if __name__ == "__main__":
    run(sys.argv[1:])



