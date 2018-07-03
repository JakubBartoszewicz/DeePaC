from keras.layers.convolutional import *
from keras.layers.normalization import *
from keras.layers.core import *

class RevCompConv1D(Conv1D):
    '''
    Based on implementation from https://github.com/kundajelab/keras/tree/keras_1
    Now compatible with Keras 2 models.
    Like Convolution1D, except the reverse-complement filters with tied
    weights are added in the channel dimension. The reverse complement
    of the channel at index i is at index -i.

    # Example

    ```python
        # apply a reverse-complemented convolution 1d of length 20
        # to a sequence with 100bp input, with 2*64 output filters
        model = Sequential()
        model.add(RevCompConv1D(filters=64, kernel_size=20,
                                padding='same', input_shape=(100, 4)))
        # now model.output_shape == (None, 100, 128)

        # add a new reverse-complemented conv1d on top
        model.add(RevCompConv1D(filters=32, kernel_size=10,
                                padding='same'))
        # now model.output_shape == (None, 10, 64)
    ```

    # Arguments
        filters: Number of non-reverse complemented convolution kernels
            to use (half the dimensionality of the output).
        kernel_size: The extension (spatial or temporal) of each filter.
        kernel_initializer: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
#        weights: list of numpy arrays to set as initial weights
#            (reverse-complemented portion should not be included as
            it's applied during compilation)
        padding: 'valid', 'same' or 'full'. ('full' requires the Theano backend.)
        strides: factor by which to subsample output.
        kernel_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        bias_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        kernel_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        bias_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        use_bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
#        input_dim: Number of channels/dimensions in the input.
#            Either this argument or the keyword argument `input_shape`must be
#            provided when using this layer as the first layer in a model.
#        input_length: Length of input sequences, when it is constant.
#            This argument is required if you are going to connect
#            `Flatten` then `Dense` layers upstream
#            (without it, the shape of the dense outputs cannot be computed).
        #new for Keras 2
        dilation_rate: An integer or tuple/list of n integers, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).

    # Input shape
        3D tensor with shape: `(samples, steps, input_dim)`.

    # Output shape
        3D tensor with shape: `(samples, new_steps, filters)`.
        `steps` value might have changed due to padding.
    '''

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (2*self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], 2*self.filters) + tuple(new_space)

    def call(self, inputs):
        #create a rev-comped W. The last axis is the output channel axis.
        #Rev comp is along both the length (dim 0) and input channel (dim 1)
        #axes; that is the reason for ::-1, ::-1 in the first and third dims.
        #The rev-comp of channel at index i should be at index -i
        #This is the reason for the ::-1 in the last dim.
        rev_comp_kernel = K.concatenate([self.kernel, self.kernel[::-1,::-1,::-1]], axis=-1)
        output = K.conv1d(inputs, rev_comp_kernel, strides=self.strides[0], padding=self.padding, data_format=self.data_format, dilation_rate=self.dilation_rate[0])
        if self.use_bias:
            rev_comp_bias = K.concatenate([self.bias, self.bias[::-1]], axis=-1)
            output = K.bias_add(output, rev_comp_bias, data_format=self.data_format)

        if self.activation is not None:
            return self.activation(output)
        return output


class RevCompConv1DBatchNorm(Layer):
    '''
    Based on implementation from https://github.com/kundajelab/keras/tree/keras_1
    Now compatible with Keras 2 models.
    Batch norm that shares weights over reverse complement channels
    '''
    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(RevCompConv1DBatchNorm, self).__init__(**kwargs)
        self.supports_masking = True
        assert axis==-1 or axis==2, "Intended for Conv1D"
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(moving_variance_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        self.num_input_chan = input_shape[self.axis]
        if self.num_input_chan is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_len = input_shape[1]
        assert len(input_shape)==3,\
         "Implementation done with RevCompConv1D input in mind"
        assert self.input_len is not None,\
         "not implemented for undefined input len"
        assert self.num_input_chan%2 == 0, "should be even for revcomp input"
        self.input_spec = InputSpec(ndim=3,
                                    axes={self.axis: self.num_input_chan})
        shape = (int(self.num_input_chan/2),)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)
        self.moving_variance = self.add_weight(
            shape=shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        inputs_orig = inputs
        inputs = K.concatenate(
            tensors=[inputs[:,:,:int(self.num_input_chan/2)],
                     inputs[:,:,int(self.num_input_chan/2):][:,:,::-1]], axis=1)
        input_shape = K.int_shape(inputs)
        # Prepare broadcasting shape.
        reduction_axes = list(range(3))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * 3
        broadcast_shape[self.axis] = int(self.num_input_chan/2)

        # Determines whether broadcasting is needed.
        #needs_broadcasting = (sorted(reduction_axes) != list(range(3))[:-1])
        needs_broadcasting = True

        def normalize_inference():
            if needs_broadcasting:
                # In this case we must explicitly broadcast all parameters.
                broadcast_moving_mean = K.reshape(self.moving_mean,
                                                  broadcast_shape)
                broadcast_moving_variance = K.reshape(self.moving_variance,
                                                      broadcast_shape)
                if self.center:
                    broadcast_beta = K.reshape(self.beta, broadcast_shape)
                else:
                    broadcast_beta = None
                if self.scale:
                    broadcast_gamma = K.reshape(self.gamma,
                                                broadcast_shape)
                else:
                    broadcast_gamma = None
                return K.batch_normalization(
                    inputs,
                    broadcast_moving_mean,
                    broadcast_moving_variance,
                    broadcast_beta,
                    broadcast_gamma,
                    axis=self.axis,
                    epsilon=self.epsilon)
            else:
                return K.batch_normalization(
                    inputs,
                    self.moving_mean,
                    self.moving_variance,
                    self.beta,
                    self.gamma,
                    axis=self.axis,
                    epsilon=self.epsilon)

        # If the learning phase is *static* and set to inference:
        if training in {0, False}:
            inputs_normed = normalize_inference()
            true_inputs_normed = K.concatenate(tensors=[inputs_normed[:,:self.input_len,:], inputs_normed[:,self.input_len:,:][:,:,::-1]], axis=2)
            return true_inputs_normed

        # If the learning is either dynamic, or set to training:
        normed_training, mean, variance = K.normalize_batch_in_training(
            inputs, self.gamma, self.beta, reduction_axes,
            epsilon=self.epsilon)

        if K.backend() != 'cntk':
            sample_size = K.prod([K.shape(inputs)[axis]
                                  for axis in reduction_axes])
            sample_size = K.cast(sample_size, dtype=K.dtype(inputs))

            # sample variance - unbiased estimator of population variance
            variance *= sample_size / (sample_size - (1.0 + self.epsilon))

        self.add_update([K.moving_average_update(self.moving_mean,
                                                 mean,
                                                 self.momentum),
                         K.moving_average_update(self.moving_variance,
                                                 variance,
                                                 self.momentum)],
                        inputs)

        # Pick the normalized form corresponding to the training phase.
        inputs_normed = K.in_train_phase(normed_training, normalize_inference, training=training)
        #recover the reverse-complemented channels
        true_inputs_normed = K.concatenate(
            tensors=[inputs_normed[:,:self.input_len,:],
                     inputs_normed[:,self.input_len:,:][:,:,::-1]],
            axis=2)
        return true_inputs_normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer': initializers.serialize(self.moving_variance_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(RevCompConv1DBatchNorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
		
		

class DenseAfterRevcompWeightedSum(Dense):
    '''  
    Based on implementation from https://github.com/kundajelab/keras/tree/keras_1
    Now compatible with Keras 2 models.
    '''
    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[-1]
        assert input_dim%2 == 0,\
         ("input_dim should be even if input is WeightedSum layer with"+
          " input_is_revcomp_conv=True")
        self.input_dim = input_dim

        self.kernel = self.add_weight((int(input_dim/2), self.units),
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight((self.units,),
                                     initializer=self.bias_initializer,
                                     name='bias',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)
        else:
            self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        output = K.dot(inputs, K.concatenate(
                             tensors=[self.kernel, self.kernel[::-1,:]], axis=0))
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output


class DenseAfterRevcompConv1D(Dense):
    '''For dense layers that follow 1D Convolutional or Pooling layers that
    have reverse-complement weight sharing
    '''

    def build(self, input_shape):
        assert len(input_shape) == 3, "layer designed to follow 1D conv/pool"
        num_chan = input_shape[-1]
        input_length = input_shape[-2]
        assert num_chan%2 == 0,\
         ("num_chan should be even if input is WeightedSum layer with"+
          " input_is_revcomp_conv=True")
        self.num_chan = num_chan
        self.input_length = input_length

        self.kernel = self.add_weight(shape=(int(input_length*num_chan/2), self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.num_chan})
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 3
        output_shape = [input_shape[0], self.units]
        return tuple(output_shape)

    def call(self, inputs):
        kernel = K.reshape(self.kernel, (self.input_length, int(self.num_chan/2),
                               self.units))
        concatenated_reshaped_kernel = K.reshape(K.concatenate(
            tensors=[kernel, kernel[::-1,::-1,:]], axis=1),
            (self.input_length*self.num_chan, self.units))
        reshaped_inputs = K.reshape(inputs, (-1, self.input_length*self.num_chan))
        output = K.dot(reshaped_inputs, concatenated_reshaped_kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output
