from deeplift.conversion.kerasapi_conversion import *
from deeplift.util import *
from deeplift_with_filtering import SequentialModelFilter, Conv1DFilter, GlobalAvgPool1D, RevCompConv1DFilter, DenseAfterRevcompWeightedSum


def convert_model_from_saved_files(
    h5_file, json_file=None, yaml_file=None, **kwargs):
    '''
    Based on implementation from https://github.com/kundajelab/deeplift/tree/keras2compat
    Creates SequentialModelFilter object with the functionality to backprogate only relevance
    scores which pass through certain filter neurons.  
    '''
    assert json_file is None or yaml_file is None,\
        "At most one of json_file and yaml_file can be specified"
    if (json_file is not None):
        model_class_and_config=json.loads(open(json_file))
    elif (yaml_file is not None):
        model_class_and_config=yaml.load(open(yaml_file))
    else:
        str_data = h5py.File(h5_file).attrs["model_config"]
        if (hasattr(str_data,'decode')):
            str_data = str_data.decode("utf-8")
        model_class_and_config = json.loads(str_data)
    model_class_name = model_class_and_config["class_name"] 
    model_config = model_class_and_config["config"]

    model_weights = h5py.File(h5_file)
    if ('model_weights' in model_weights.keys()):
        model_weights=model_weights['model_weights']

    if (model_class_name=="Sequential"):
        layer_configs = model_config
        model_conversion_function = convert_sequential_model_filter #changed
    elif (model_class_name=="Model"):
        layer_configs = model_config["layers"]
        model_conversion_function = convert_functional_model
    else:
        raise NotImplementedError("Don't know how to convert "
                                  +model_class_name)

    #add in the weights of the layer to the layer config
    for layer_config in layer_configs:
         
        layer_name = layer_config["config"]["name"]
        assert layer_name in model_weights,\
            ("Layer "+layer_name+" is in the layer names but not in the "
             +" weights file which has layer names "+model_weights.keys())

        if (layer_config["class_name"] in ["Model", "Sequential"]):
            nested_model_weights =\
                OrderedDict(zip(
                 model_weights[layer_name].attrs["weight_names"],
                 [model_weights[layer_name][x] for x in
                  model_weights[layer_name].attrs["weight_names"]]))

        if (layer_config["class_name"]=="Model"):
            insert_weights_into_nested_model_config(
                nested_model_weights=nested_model_weights,
                nested_model_layer_config=layer_config["config"]["layers"])
        elif (layer_config["class_name"]=="Sequential"):
            insert_weights_into_nested_model_config(
                nested_model_weights=nested_model_weights,
                nested_model_layer_config=layer_config["config"])
        else:  
            layer_weights = [np.array(model_weights[layer_name][x]) for x in
                             model_weights[layer_name].attrs["weight_names"]]
            layer_config["config"]["weights"] = layer_weights
        
    return model_conversion_function(model_config=model_config, **kwargs)



def convert_sequential_model_filter(
    model_config,
    nonlinear_mxts_mode=\
     NonlinearMxtsMode.DeepLIFT_GenomicsDefault,
    verbose=True,
    dense_mxts_mode=DenseMxtsMode.Linear,
    conv_mxts_mode=ConvMxtsMode.Linear,
    maxpool_deeplift_mode=default_maxpool_deeplift_mode,
    layer_overrides={}):

    '''
    Based on implementation from https://github.com/kundajelab/deeplift/tree/keras2compat
    Convert standard keras model to extended deeplift model (SequentialModelFilter)
    '''
    if (verbose):
        print("nonlinear_mxts_mode is set to: "
              +str(nonlinear_mxts_mode))
        sys.stdout.flush()

    converted_layers = []
    batch_input_shape = model_config[0]['config'][KerasKeys.batch_input_shape]
    converted_layers.append(
        layers.core.Input(batch_shape=batch_input_shape, name="input"))
    #converted_layers is actually mutated to be extended with the
    #additional layers so the assignment is not strictly necessary,
    #but whatever
    converted_layers = sequential_container_conversion( #changed
                config=model_config, name="", verbose=verbose,
                nonlinear_mxts_mode=nonlinear_mxts_mode,
                dense_mxts_mode=dense_mxts_mode,
                conv_mxts_mode=conv_mxts_mode,
                maxpool_deeplift_mode=maxpool_deeplift_mode,
                converted_layers=converted_layers,
                layer_overrides=layer_overrides)
    converted_layers[-1].build_fwd_pass_vars()
    return SequentialModelFilter(converted_layers) #changed



def sequential_container_conversion(config,
                                    name, verbose,
                                    nonlinear_mxts_mode,
                                    dense_mxts_mode,
                                    conv_mxts_mode,
                                    maxpool_deeplift_mode,
                                    converted_layers=None,
                                    layer_overrides={}):
    '''
    Implementation from https://github.com/kundajelab/deeplift/tree/keras2compat
    '''
    if (converted_layers is None):
        converted_layers = []
    name_prefix=name
    for layer_idx, layer_config in enumerate(config):
        modes_to_pass = {'dense_mxts_mode': dense_mxts_mode,
                         'conv_mxts_mode': conv_mxts_mode,
                         'nonlinear_mxts_mode': nonlinear_mxts_mode,
                         'maxpool_deeplift_mode': maxpool_deeplift_mode}
        if layer_idx in layer_overrides:
            for mode in ['dense_mxts_mode', 'conv_mxts_mode',
                         'nonlinear_mxts_mode']:
                if mode in layer_overrides[layer_idx]:
                    modes_to_pass[mode] = layer_overrides[layer_idx][mode] 
        if (layer_config["class_name"] != "InputLayer"):
            conversion_function = layer_name_to_conversion_function(
                                   layer_config["class_name"])
            converted_layers.extend(conversion_function(
                                 config=layer_config["config"],
                                 name=(name_prefix+"-" if name_prefix != ""
                                       else "")+str(layer_idx),
                                 verbose=verbose,
                                 **modes_to_pass)) 
        else:
            print("Encountered an Input layer in sequential container; "
                  "skipping due to redundancy")
    deeplift.util.connect_list_of_layers(converted_layers)
    return converted_layers


def layer_name_to_conversion_function(layer_name):
    '''
    Based on implementation from https://github.com/kundajelab/deeplift/tree/keras2compat
    Inclusion of new conversion functions namely conv1dfilter_conversion, revcompconv1d_conversion,
    revcompconv1dbatchnorm_conversion, denseafterrevcompweightedsum_conversion
    '''
    name_dict = {
        'inputlayer': input_layer_conversion,

        'conv1d': conv1dfilter_conversion,
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


def globalavgpooling1d_conversion(config, name, verbose, **kwargs):
    return [GlobalAvgPool1D(
             name=name,
             verbose=verbose)]


def conv1dfilter_conversion(config,
                      name,
                      verbose,
                      nonlinear_mxts_mode,
                      conv_mxts_mode, **kwargs):
    '''
    Based on implementation from https://github.com/kundajelab/deeplift/tree/keras2compat
    Builds Conv1DFilter layer (which inherits from Conv1D) with filtering functionalities.
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
    to_return = [Conv1DFilter( #changed
            name=("preact_" if len(converted_activation) > 0
                        else "")+name,
            kernel=config[KerasKeys.weights][0],
            bias=config[KerasKeys.weights][1],
            stride=config[KerasKeys.strides],
            padding=config[KerasKeys.padding].upper(),
            conv_mxts_mode=conv_mxts_mode)] 
    to_return.extend(converted_activation)
    return deeplift.util.connect_list_of_layers(to_return)


def revcompconv1dfilter_conversion(config,
                      name,
                      verbose,
                      nonlinear_mxts_mode,
                      conv_mxts_mode, **kwargs):
    '''
    Based on implementation from https://github.com/kundajelab/deeplift/tree/keras2compat
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
    to_return = [RevCompConv1DFilter( #changed
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
