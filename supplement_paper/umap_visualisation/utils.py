import numpy as np
import pickle
import sys
import os


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


def read_embedding(embedding_path):

    print("Reading embedding...")
    try:
        openFile = open(embedding_path, 'rb')
        umap_object = pickle.load(openFile)  # read the data
        openFile.close()
        print("Embedding succesfully read.")

    except IOError:
        print("Error: ", embedding_path, " not found.")
        sys.exit(0)

    return umap_object
