import argparse
import umap_vis
import utils
import sys


def run(args):

    # parsing arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset-filename', dest="dataset_filename", default=None)
    parser.add_argument('-l', '--label-filename', dest="label_filename")
    parser.add_argument('-s', '--seed', dest="seed", type=int, default=None)

    params = parser.parse_args(args)

    data, labels = utils.read_data_and_labels(params.dataset_filename, params.label_filename)
    element_no = len(data)

    n_neighbors_params = [int(0.0005*element_no),
                          int(0.001*element_no),
                          int(0.002*element_no),
                          int(0.005*element_no),
                          int(0.01*element_no)]

    min_dist_params = [0.0, 0.1, 0.25, 0.5, 0.75, 0.99]
    n_components_params = [1, 2, 3]

    params = parser.parse_args(args)

    for c in n_components_params:
        for n in n_neighbors_params:
            umap_vis.run_workflow(params.dataset_filename, None, params.label_filename, c, 0.1, n, 1, None)

        for d in min_dist_params:
            umap_vis.run_workflow(params.dataset_filename, None, params.label_filename, c, 15, d, 1, None)


if __name__ == "__main__":

    run(sys.argv[1:])