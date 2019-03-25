from deeplift.conversion.kerasapi_conversion import *
from deeplift.util import *
from deeplift_additional_layers import GlobalMaxPool1D, GlobalAvgPool1D, RevCompConv1D, DenseAfterRevcompWeightedSum

'''
Functions to convert Keras 2 models to DeepLIFT models (Shrikumar et al. "Learning important features through propagating activation differences." arXiv preprint, 2017)
for some additional layers.
'''

def layer_name_to_conversion_function(layer_name):
    '''
    Based on implementation from https://github.com/kundajelab/deeplift/tree/keras2compat
    Inclusion of new conversion functions namely globalavgpooling1d_conversion, revcompconv1d_conversion,
    revcompconv1dbatchnorm_conversion and denseafterrevcompweightedsum_conversion
    '''
    name_dict = {
        'inputlayer': input_layer_conversion,

        'conv1d': conv1d_conversion,
        'revcompconv1d': revcompconv1dfilter_conversion, 
        'maxpooling1d': maxpool1d_conversion,
        'globalmaxpooling1d': globalmaxpooling1d_conversion,
        'averagepooling1d': avgpool1d_conversion,
        'globalaveragepooling1d': globalavgpooling1d_conversion,

        'conv2d': conv2d_conversion,
        'maxpooling2d': maxpool2d_conversion,
        'averagepooling2d': avgpool2d_conversion,

        'batchnormalization': batchnorm_conversion,
        'revcompconv1dbatchnorm': revcompconv1dbatchnorm_conversion,
        'dropout': noop_conversion, 
        'flatten': flatten_conversion,
        'dense': dense_conversion,
        'denseafterrevcompweightedsum': denseafterrevcompweightedsum_conversion,

        'activation': activation_conversion,
        'prelu': prelu_conversion,

        'sequential': sequential_container_conversion,
        'model': functional_container_conversion,
        'concatenate': concat_conversion_function 
    }
    # lowercase to create resistance to capitalization changes
    # was a problem with previous Keras versions
    return name_dict[layer_name.lower()]

#override original deeplift conversion function
deeplift.conversion.kerasapi_conversion.layer_name_to_conversion_function = layer_name_to_conversion_function

def globalavgpooling1d_conversion(config, name, verbose, **kwargs):
    return [GlobalAvgPool1D(
             name=name,
             verbose=verbose)]

def globalmaxpooling1d_conversion(config, name, verbose,
                                  maxpool_deeplift_mode, **kwargs):
    return [GlobalMaxPool1D(
             name=name,
             verbose=verbose,
             maxpool_deeplift_mode=maxpool_deeplift_mode)]



def revcompconv1dfilter_conversion(config,
                      name,
                      verbose,
                      nonlinear_mxts_mode,
                      conv_mxts_mode, **kwargs):
    '''
    Based on implementation from https://github.com/kundajelab/deeplift/
    '''
    validate_keys(config, [KerasKeys.weights,
                           KerasKeys.activation,
                           KerasKeys.filters,
                           KerasKeys.kernel_size,
                           KerasKeys.padding,
                           KerasKeys.strides])
    #nonlinear_mxts_mode only used for activation
    converted_activation = activation_conversion(
                            config=config,
                            name=name,
                            verbose=verbose,
                            nonlinear_mxts_mode=nonlinear_mxts_mode)
    to_return = [RevCompConv1D(
            name=("preact_" if len(converted_activation) > 0
                        else "")+name,
            kernel=config[KerasKeys.weights][0],
            bias=config[KerasKeys.weights][1],
            stride=config[KerasKeys.strides],
            padding=config[KerasKeys.padding].upper(),
            conv_mxts_mode=conv_mxts_mode)]
    to_return.extend(converted_activation)
    return deeplift.util.connect_list_of_layers(to_return)


def revcompconv1dbatchnorm_conversion(config,
                      name,
                      verbose,
                      nonlinear_mxts_mode,
                      conv_mxts_mode, **kwargs):
    raise NotImplementedError("Not yet implemented!")


def denseafterrevcompweightedsum_conversion(config,
                     name,
                     verbose,
                     dense_mxts_mode,
                     nonlinear_mxts_mode,
                     **kwargs):

    validate_keys(config, [KerasKeys.weights,
                           KerasKeys.activation])

    converted_activation = activation_conversion(
                            config=config,
                            name=name,
                            verbose=verbose,
                            nonlinear_mxts_mode=nonlinear_mxts_mode) 
    to_return = [DenseAfterRevcompWeightedSum(
                  name=("preact_" if len(converted_activation) > 0
                        else "")+name, 
                  kernel=config[KerasKeys.weights][0],
                  bias=config[KerasKeys.weights][1],
                  verbose=verbose,
                  dense_mxts_mode=dense_mxts_mode)]
    to_return.extend(converted_activation)
    return deeplift.util.connect_list_of_layers(to_return)

