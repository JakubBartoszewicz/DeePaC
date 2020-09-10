import numpy as np


def get_rf_size(mdl, idx, conv_ids, insert_pooling=True):
    """Calculate receptive field size (motif length)"""
    kernel_sizes = []
    strides = []
    dilations = []
    for i in range(len(conv_ids)):
        if i <= idx:
            config = mdl.get_layer(index=conv_ids[i]).get_config()
            kernel_sizes.append(config["kernel_size"][0])
            strides.append(config["strides"][0])
            dilations.append(config["dilation_rate"][0])
            if insert_pooling:
                kernel_sizes.append(2)
        else:
            break
    return get_rf_size_from_ksd(kernel_sizes, strides, dilations)


def get_rf_size_from_ksd(kernel_sizes, strides, dilations):
    """Calculate receptive field size (motif length) from lists of kernel sizes, strides and dilations"""
    s = np.asarray(strides[:len(kernel_sizes)-2])
    k = np.multiply(np.asarray(dilations), np.asarray(kernel_sizes)-1) + 1
    rf = int(np.sum((k-1)*np.prod(s)) + 1)
    return rf
