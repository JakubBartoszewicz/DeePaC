from sklearn.utils import shuffle
import numpy as np
import pickle
import pandas as pd
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


def read_data_desc(data_desc_csv):

    file_name, path = get_data_name(data_desc_csv)
    f_name, ext = os.path.splitext(file_name)
    data_desc = pd.read_csv(data_desc_csv, sep=";", header=None)

    species_labels = data_desc.iloc[:, 1].values.tolist()
    number_of_reads = data_desc.iloc[:, 2].values.tolist()

    read_sum = np.sum(number_of_reads)
    print("Number of reads: ", read_sum)
    species_labels_long = np.repeat(species_labels, number_of_reads)

    for i in range(0, len(species_labels)):
        assert np.count_nonzero(species_labels_long == species_labels[i]) == number_of_reads[i]

    np.save(os.path.join(path, "_".join([f_name, "species_labels.npy"])), species_labels_long)
    print("Species label file created!")


def shuffle_arrays(array_a, array_b, array_c):

    a_a = np.load(array_a)
    a_b = np.load(array_b)
    a_c = np.load(array_c)

    if len(a_a) != len(a_b) or len(a_a) != len(a_c):
        print("Arrays cannot be shuffled together - different sizes!")
        sys.exit(0)

    array_a_sh, array_b_sh, array_c_sh = shuffle(a_a, a_b, a_c, random_state=1)

    file_name, path = get_data_name(array_a)
    f_name, ext = os.path.splitext(file_name)

    np.save(os.path.join(path, "_".join([f_name, "shuffled.npy"])), array_a_sh)

    file_name, path = get_data_name(array_b)
    f_name, ext = os.path.splitext(file_name)

    np.save(os.path.join(path, "_".join([f_name, "shuffled.npy"])), array_b_sh)

    file_name, path = get_data_name(array_c)
    f_name, ext = os.path.splitext(file_name)

    np.save(os.path.join(path, "_".join([f_name, "shuffled.npy"])), array_c_sh)
