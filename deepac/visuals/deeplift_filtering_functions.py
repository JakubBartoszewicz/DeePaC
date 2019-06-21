from deeplift.models import *
from deeplift.layers.convolutional import *
from deeplift.layers import helper_functions as hf
from deeplift.layers.helper_functions import conv1d_transpose_via_conv2d
import tensorflow as tf


def get_pos_and_neg_mxts(self):
    """
    Returns the positive and negative multipliers of a layer.
    """
    return tf.concat([self._pos_mxts, self._neg_mxts], axis=2)


# add method to convolutional layer
Conv1D.get_mxts = get_pos_and_neg_mxts


def get_mxts_after_filtering(conv_layer, diff_from_ref, pos_mxts, neg_mxts, filter_mask, kernel): 
    """
    Based on implementation from https://github.com/kundajelab/deeplift/
    Performs transposed convolution to compute the multipliers of the inputs of the convolutional layer.
    Deeplift multipliers are multiplied with a filter mask containing ones or zeros to backpropagate
    only relevance scores of filter neurons where the corresponding entry of the filter mask is one.
    """
    pos_mxts *= filter_mask
    neg_mxts *= filter_mask
    pos_inp_mask = hf.gt_mask(diff_from_ref, 0.0)
    neg_inp_mask = hf.lt_mask(diff_from_ref, 0.0)
    zero_inp_mask = hf.eq_mask(diff_from_ref, 0.0)
    inp_mxts_increments = pos_inp_mask*(
        conv1d_transpose_via_conv2d(
            value=pos_mxts,
            kernel=kernel*(hf.gt_mask(kernel, 0.0)),
            tensor_with_output_shape=tf.zeros_like(diff_from_ref),
            padding=conv_layer.padding,
            stride=conv_layer.stride)
        + conv1d_transpose_via_conv2d(
            value=neg_mxts,
            kernel=kernel*(hf.lt_mask(kernel, 0.0)),
            tensor_with_output_shape=tf.zeros_like(diff_from_ref),
            padding=conv_layer.padding,
            stride=conv_layer.stride))
    inp_mxts_increments += neg_inp_mask*(
        conv1d_transpose_via_conv2d(
            value=pos_mxts,
            kernel=kernel*(hf.lt_mask(kernel, 0.0)),
            tensor_with_output_shape=tf.zeros_like(diff_from_ref),
            padding=conv_layer.padding,
            stride=conv_layer.stride)
        + conv1d_transpose_via_conv2d(
            value=neg_mxts,
            kernel=kernel*(hf.gt_mask(kernel, 0.0)),
            tensor_with_output_shape=tf.zeros_like(diff_from_ref),
            padding=conv_layer.padding,
            stride=conv_layer.stride))
    inp_mxts_increments += zero_inp_mask*(
        conv1d_transpose_via_conv2d(
            value=0.5*(neg_mxts+pos_mxts),
            kernel=kernel,
            tensor_with_output_shape=tf.zeros_like(diff_from_ref),
            padding=conv_layer.padding,
            stride=conv_layer.stride))
    pos_mxts_increments = inp_mxts_increments
    neg_mxts_increments = inp_mxts_increments
    return pos_mxts_increments, neg_mxts_increments


def get_pos_and_neg_contribs_input_layer(diff_from_ref):
    """
    Based on implementation from https://github.com/kundajelab/deeplift/
    Computes positive and negative part of the difference from reference
    of the input neurons.
    """
    pos_contribs = (diff_from_ref *
                    hf.gt_mask(diff_from_ref, 0.0))
    neg_contribs = (diff_from_ref *
                    hf.lt_mask(diff_from_ref, 0.0))
    return pos_contribs, neg_contribs


def get_contribs_of_inputs_after_filtering_conv_layer(conv_layer, diff_from_ref_input, pos_mxts_conv,
                                                      neg_mxts_conv, filter_mask, kernel):
    """
    Computes DeepLIFT contribution scores of the input layer after performing filtering in the convolutional layer.
    """
    pos_mxts_input, neg_mxts_input = get_mxts_after_filtering(conv_layer, diff_from_ref_input, pos_mxts_conv,
                                                              neg_mxts_conv, filter_mask, kernel)
    pos_delta_input, neg_delta_input = get_pos_and_neg_contribs_input_layer(diff_from_ref_input)
    return pos_delta_input * pos_mxts_input + neg_delta_input*neg_mxts_input
