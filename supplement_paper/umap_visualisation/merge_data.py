import numpy as np
import pandas as pd
import os
import argparse
import sys
from datetime import datetime
import utils


def read_data_info(data_info):

    df = pd.read_csv(data_info, sep=";")

    names = df["name"].tolist()
    data_paths = df["d_path"].tolist()
    label_paths = df["l_path"].tolist()

    return names, data_paths, label_paths


def read_data(data_paths, label_paths):

    data_list = []
    label_list = []

    for d, l in zip(data_paths, label_paths):
        data, labels = utils.read_data_and_labels(d, l)
        data_list.append(data)
        label_list.append(labels)

    return data_list, label_list


def merge_data(names, data_list, label_list):

    merged_data = np.concatenate(data_list)
    i = 0

    multiclass_label_list = []

    for lb in label_list:
        lb = lb + 2 * i
        i += 1
        multiclass_label_list.append(lb)

    merged_labels = np.concatenate(multiclass_label_list)

    data_name = "_".join(names) + "_merged_data.npy"
    label_name = "_".join(names) + "_merged_labels.npy"

    np.save(data_name, merged_data)
    np.save(label_name, merged_labels)
    print("Files successfully merged!")


def run(args):

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_info', dest="data_info", required=True)

    params = parser.parse_args(args)

    names, data_paths, label_paths = read_data_info(params.data_info)
    data_list, label_list = read_data(data_paths, label_paths)
    merge_data(names, data_list, label_list)


if __name__ == "__main__":
    run(sys.argv[1:])

