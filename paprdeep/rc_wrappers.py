import copy

from keras.utils.generic_utils import has_arg
from keras.layers.wrappers import Wrapper, Bidirectional
from keras import backend as K

from keras.layers import recurrent

class RevCompBidirectional(Bidirectional):
    """Reverse complement Bidirectional wrapper for RNNs.
    # Arguments
        layer: `Recurrent` instance.
        merge_mode: Mode by which outputs of the
            forward and backward RNNs will be combined.
            One of {'sum', 'mul', 'concat', 'ave', None}.
            If None, the outputs will not be combined,
            they will be returned as a list.
    # Raises
        ValueError: In case of invalid `merge_mode` argument.
    # Examples
    ```python
        model = Sequential()
        model.add(RevCompBidirectional(LSTM(10, return_sequences=True),
                                input_shape=(5, 10)))
        model.add(RevCompBidirectional(LSTM(10)))
        model.add(Dense(5))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    ```
    """

    def __init__(self, layer, merge_mode='concat', weights=None, **kwargs):
        super(RevCompBidirectional, self).__init__(layer, merge_mode, weights, **kwargs)

    def compute_output_shape(self, input_shape):
        output_shape = self.forward_layer.compute_output_shape(input_shape)
        if self.return_state:
            state_shape = output_shape[1:]
            output_shape = output_shape[0]

        if self.merge_mode == 'concat':
            output_shape = list(output_shape)
            output_shape[-1] *= 4
            output_shape = tuple(output_shape)
        elif self.merge_mode is None:
            output_shape = [output_shape, copy.copy(output_shape), copy.copy(output_shape), copy.copy(output_shape)]

        if self.return_state:
            if self.merge_mode is None:
                return output_shape + state_shape + copy.copy(state_shape) + copy.copy(state_shape) + copy.copy(state_shape)
            return [output_shape] + state_shape + copy.copy(state_shape) + copy.copy(state_shape) + copy.copy(state_shape)
        return output_shape

    def call(self,
             inputs,
             mask=None,
             training=None,
             initial_state=None,
             constants=None):
        kwargs = {}
        if has_arg(self.layer.call, 'training'):
            kwargs['training'] = training
        if has_arg(self.layer.call, 'mask'):
            kwargs['mask'] = mask
        if has_arg(self.layer.call, 'constants'):
            kwargs['constants'] = constants

        if initial_state is not None and has_arg(self.layer.call, 'initial_state'):
            forward_state = initial_state[:len(initial_state) // 2]
            backward_state = initial_state[len(initial_state) // 2:]
            y = self.forward_layer.call(inputs, initial_state=forward_state, **kwargs)
            y_rc = self.forward_layer.call(inputs[::-1,::-1], initial_state=forward_state, **kwargs)
            y_rev = self.backward_layer.call(inputs, initial_state=backward_state, **kwargs)
            y_rev_rc = self.forward_layer.call(inputs[::-1,::-1], initial_state=backward_state, **kwargs)
        else:
            y = self.forward_layer.call(inputs, **kwargs)
            y_rc = self.forward_layer.call(inputs[::-1,::-1], **kwargs)
            y_rev = self.backward_layer.call(inputs, **kwargs)
            y_rev_rc = self.backward_layer.call(inputs[::-1,::-1], **kwargs)

        if self.return_state:
            states = y[1:] + y_rc[1:] + y_rev[1:] + y_rev_rc[1:]
            y = y[0]
            y_rc = y_rc[0]
            y_rev = y_rev[0]
            y_rev_rc = y_rev_rc[0]

        if self.return_sequences:
            y_rc = K.reverse(y_rc, 1)
            y_rev = K.reverse(y_rev, 1)
        if self.merge_mode == 'concat':
            output = K.concatenate([y, y_rev, y_rev_rc, y_rc])
        elif self.merge_mode == 'sum':
            output = y, y_rev, y_rev_rc, y_rc
        elif self.merge_mode == 'ave':
            output = (y + y_rev + y_rev_rc + y_rc) / 4
        elif self.merge_mode == 'mul':
            output = y * y_rev * y_rev_rc * y_rc
        elif self.merge_mode is None:
            output = [y, y_rev, y_rev_rc, y_rc]

        # Properly set learning phase
        if (getattr(y, '_uses_learning_phase', False) or
           getattr(y_rev, '_uses_learning_phase', False)):
            if self.merge_mode is None:
                for out in output:
                    out._uses_learning_phase = True
            else:
                output._uses_learning_phase = True

        if self.return_state:
            if self.merge_mode is None:
                return output + states
            return [output] + states
        return output



