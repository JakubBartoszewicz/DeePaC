from sklearn.utils import shuffle
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

    if len(data) != len(labels):
        print("Number of reads and labels not equal.")
        sys.exit(0)

    return data, labels


# check number of classes in the data set
def check_number_of_classes(labels):

    classes = set(labels)
    print("Classes: ", classes)

    return classes


def convert_labels(labels):

    classes = check_number_of_classes(labels)

    counts = [np.count_nonzero(labels == i) for i in classes]

    new_labels = np.array(range(0, len(classes)))

    converted_labels = np.repeat(new_labels, counts)

    return classes, converted_labels


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


def shuffle_arrays(array_a, array_b):

    if len(array_a) != len(array_a):
        print("Arrays cannot be shuffled together - different sizes!")
        sys.exit(0)

    array_a_sh, array_b_sh = shuffle(array_a, array_b, random_state=1)

    return array_a_sh, array_b_sh
