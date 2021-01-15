import numpy as np


def get_rf_size(mdl, idx, verbose=False):
    """Calculate receptive field size (motif length)"""
    kernel_sizes = []
    strides = []
    dilations = []
    for i, layer in enumerate(mdl.layers):
        config = layer.get_config()
        try:
            if config["kernel_size"][0] > 1:
                if verbose:
                    print("kernel size: {}".format(config["kernel_size"][0]))
                kernel_sizes.append(config["kernel_size"][0])
            else:
                continue
        except KeyError:
            try:
                if verbose:
                    print("pooling size: {}".format(config["pool_size"][0]))
                kernel_sizes.append(config["pool_size"][0])
            except KeyError:
                continue
        try:
            strides.append(config["strides"][0])
        except KeyError:
            continue
        try:
            dilations.append(config["dilation_rate"][0])
        except KeyError:
            dilations.append(1)
        if i == idx:
            break
    rf = get_rf_size_from_ksd(kernel_sizes, strides, dilations)
    return rf


def get_rf_size_from_ksd(kernel_sizes, strides, dilations):
    """Calculate receptive field size (motif length) from lists of kernel sizes, strides and dilations"""
    kernel_span = np.multiply(np.asarray(dilations), np.asarray(kernel_sizes) - 1) + 1
    rf = np.sum([(kernel_span[k] - 1) * np.prod(strides[:k - 1]) for k in range(len(kernel_span))]) + 1
    return int(rf)
