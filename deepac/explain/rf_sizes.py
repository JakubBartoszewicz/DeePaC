import numpy as np


# def get_rf_size(mdl, idx, conv_ids, insert_pooling=True):
#     """Calculate receptive field size (motif length)"""
#     kernel_sizes = []
#     strides = []
#     dilations = []
#     for i in range(len(conv_ids)):
#         config = mdl.get_layer(index=conv_ids[i]).get_config()
#         kernel_sizes.append(config["kernel_size"][0])
#         strides.append(config["strides"][0])
#         dilations.append(config["dilation_rate"][0])
#         if i == idx:
#             break
#         elif insert_pooling:
#             kernel_sizes.append(2)
#             strides.append(2)
#             dilations.append(1)
#     return get_rf_size_from_ksd(kernel_sizes, strides, dilations)

def get_rf_size(mdl, idx):
    """Calculate receptive field size (motif length)"""
    kernel_sizes = []
    strides = []
    dilations = []
    for i, layer in enumerate(mdl.layers):
        config = layer.get_config()
        try:
            if config["kernel_size"][0] > 1:
                kernel_sizes.append(config["kernel_size"][0])
            else:
                continue
        except KeyError:
            try:
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
