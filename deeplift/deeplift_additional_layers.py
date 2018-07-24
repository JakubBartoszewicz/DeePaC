#from deeplift.models import *
from deeplift.layers.convolutional import *
from deeplift.layers.pooling import *
#from tensorflow.python.ops.gen_nn_ops import avg_pool_grad

class RevCompConv1D(Conv1D):
    '''
    Based on implementation from https://github.com/kundajelab/deeplift
    Converts Keras 2 compatibile RevCompConv1D layer (Keras 1 implementation see: https://github.com/kundajelab/keras/tree/keras_1)
    to deeplift layer with filtering capacity.
    '''
    def _compute_shape(self, input_shape):
        '''
        Based on implementation from https://github.com/kundajelab/deeplift
        Double number of output channels (filter) for fwd and rev-complement.
        '''
        #assuming a theano dimension ordering here...
        shape_to_return = [None]
        if (input_shape is None or input_shape[1] is None):
            shape_to_return += [None]
        else:
            if (self.padding == PaddingMode.valid):
                #overhands are excluded
                shape_to_return.append(
                    1+int((input_shape[1]-self.kernel.shape[0])/self.stride))
            elif (self.padding == PaddingMode.same):
                shape_to_return.append(
                    int((input_shape[1]+self.stride-1)/self.stride)) 
            else:
                raise RuntimeError("Please implement shape inference for"
                                   " padding mode: "+str(self.padding))
        shape_to_return.append(2*self.kernel.shape[-1]) 
        return shape_to_return

    def _build_activation_vars(self, input_act_vars):
        '''
        Based on implementation from https://github.com/kundajelab/deeplift
        Computes convolution with concatenated fwd and reverse complemented kernels and bias.
        '''
        rev_comp_kernel = tf.concat([self.kernel, self.kernel[::-1,::-1,::-1]], axis=-1)
        rev_comp_conv_without_bias = self._compute_conv_without_bias(
                                input_act_vars,
                                kernel=rev_comp_kernel)
        rev_comp_bias = tf.concat([self.bias, self.bias[::-1]], axis=-1)
        return rev_comp_conv_without_bias + rev_comp_bias[None,None,:]

    def _build_pos_and_neg_contribs(self):
        '''
        Based on implementation from https://github.com/kundajelab/deeplift
        Computes contribution scores with concatenated fwd and reverse-complemented kernels.
        '''
        if (self.conv_mxts_mode == ConvMxtsMode.Linear):
            concatenated_kernel = tf.concat([self.kernel, self.kernel[::-1,::-1,::-1]], axis=-1)
            inp_diff_ref = self._get_input_diff_from_reference_vars() 
            pos_contribs = (self._compute_conv_without_bias(
                             x=inp_diff_ref*hf.gt_mask(inp_diff_ref,0.0),
                             kernel=concatenated_kernel*hf.gt_mask(concatenated_kernel,0.0))
                           +self._compute_conv_without_bias(
                             x=inp_diff_ref*hf.lt_mask(inp_diff_ref,0.0),
                             kernel=concatenated_kernel*hf.lt_mask(concatenated_kernel,0.0)))
            neg_contribs = (self._compute_conv_without_bias(
                             x=inp_diff_ref*hf.lt_mask(inp_diff_ref,0.0),
                             kernel=concatenated_kernel*hf.gt_mask(concatenated_kernel,0.0))
                           +self._compute_conv_without_bias(
                             x=inp_diff_ref*hf.gt_mask(inp_diff_ref,0.0),
                             kernel=concatenated_kernel*hf.lt_mask(concatenated_kernel,0.0)))
        else:
            raise RuntimeError("Unsupported conv_mxts_mode: "+
                               self.conv_mxts_mode)
        return pos_contribs, neg_contribs

    def _get_mxts_increments_for_inputs(self):
        '''
        Based on implementation from https://github.com/kundajelab/deeplift
        Computes multipliers with concatenated fwd and reverse complemented kernels.
        ''' 
        pos_mxts = self.get_pos_mxts()
        neg_mxts = self.get_neg_mxts()
        inp_diff_ref = self._get_input_diff_from_reference_vars() 
        output_shape = self._get_input_shape()
        if (self.conv_mxts_mode == ConvMxtsMode.Linear): 
            pos_inp_mask = hf.gt_mask(inp_diff_ref,0.0)
            neg_inp_mask = hf.lt_mask(inp_diff_ref,0.0)
            zero_inp_mask = hf.eq_mask(inp_diff_ref,0.0)
            concatenated_kernel = tf.concat([self.kernel, self.kernel[::-1,::-1,::-1]], axis=-1)
            inp_mxts_increments = pos_inp_mask*(
                conv1d_transpose_via_conv2d(
                    value=pos_mxts,
                    kernel=concatenated_kernel*(hf.gt_mask(concatenated_kernel,0.0)),
                    tensor_with_output_shape=self.inputs.get_activation_vars(),
                    padding=self.padding,
                    stride=self.stride)
                +conv1d_transpose_via_conv2d(
                    value=neg_mxts,
                    kernel=concatenated_kernel*(hf.lt_mask(concatenated_kernel,0.0)),
                    tensor_with_output_shape=self.inputs.get_activation_vars(),
                    padding=self.padding,
                    stride=self.stride))
            inp_mxts_increments += neg_inp_mask*(
                conv1d_transpose_via_conv2d(
                    value=pos_mxts,
                    kernel=concatenated_kernel*(hf.lt_mask(concatenated_kernel,0.0)),
                    tensor_with_output_shape=self.inputs.get_activation_vars(),
                    padding=self.padding,
                    stride=self.stride)
                +conv1d_transpose_via_conv2d(
                    value=neg_mxts,
                    kernel=concatenated_kernel*(hf.gt_mask(concatenated_kernel,0.0)),
                    tensor_with_output_shape=self.inputs.get_activation_vars(),
                    padding=self.padding,
                    stride=self.stride))
            inp_mxts_increments += zero_inp_mask*(
                conv1d_transpose_via_conv2d(
                    value=0.5*(neg_mxts+pos_mxts),
                    kernel=concatenated_kernel,
                    tensor_with_output_shape=self.inputs.get_activation_vars(),
                    padding=self.padding,
                    stride=self.stride))
            pos_mxts_increments = inp_mxts_increments
            neg_mxts_increments = inp_mxts_increments
        else:
            raise RuntimeError("Unsupported conv mxts mode: "
                               +str(self.conv_mxts_mode))
        return pos_mxts_increments, neg_mxts_increments



class DenseAfterRevcompWeightedSum(Dense):
    '''
    Based on implementation from https://github.com/kundajelab/deeplift
    Converts Keras 2 compatibile DenseAfterRevcompWeightedSum layer (Keras 1 implementation see: https://github.com/kundajelab/keras/tree/keras_1)
    to deeplift layer.
    '''
    def _build_activation_vars(self, input_act_vars):
        '''
        Based on implementation from https://github.com/kundajelab/deeplift
        Computes activation vars with concatenated fwd and reverse-complemented weights.
        '''
        return tf.matmul(input_act_vars, tf.concat([self.kernel, self.kernel[::-1,:]], axis=0)) + self.bias

    def _build_pos_and_neg_contribs(self):
        '''
        Based on implementation from https://github.com/kundajelab/deeplift
        Computes contribution scores with fwd and reverse-complemented weights.
        '''
        if (self.dense_mxts_mode == DenseMxtsMode.Linear): 
            inp_diff_ref = self._get_input_diff_from_reference_vars()
            concatenated_kernel = tf.concat([self.kernel, self.kernel[::-1,:]], axis=0) 
            pos_contribs = (tf.matmul(
                             inp_diff_ref*hf.gt_mask(inp_diff_ref, 0.0),
                             concatenated_kernel*hf.gt_mask(concatenated_kernel,0.0))
                            +tf.matmul(
                              inp_diff_ref*hf.lt_mask(inp_diff_ref, 0.0),
                              concatenated_kernel*hf.lt_mask(concatenated_kernel,0.0)))
            neg_contribs = (tf.matmul(
                             inp_diff_ref*hf.gt_mask(inp_diff_ref, 0.0),
                             concatenated_kernel*hf.lt_mask(concatenated_kernel,0.0))
                            +tf.matmul(
                              inp_diff_ref*hf.lt_mask(inp_diff_ref, 0.0),
                              concatenated_kernel*hf.gt_mask(concatenated_kernel,0.0)))
        else:
            raise RuntimeError("Unsupported dense_mxts_mode: "+
                               self.dense_mxts_mode)
        return pos_contribs, neg_contribs

    def _get_mxts_increments_for_inputs(self):
        '''
        Based on implementation from https://github.com/kundajelab/deeplift
        Computes multipliers with concatenated fwd and reverse complemented weights.
        '''
        if (self.dense_mxts_mode == DenseMxtsMode.Linear): 
            #different inputs will inherit multipliers differently according
            #to the sign of inp_diff_ref (as this sign was used to determine
            #the pos_contribs and neg_contribs; there was no breakdown
            #by the pos/neg contribs of the input)
            inp_diff_ref = self._get_input_diff_from_reference_vars() 
            pos_inp_mask = hf.gt_mask(inp_diff_ref,0.0)
            neg_inp_mask = hf.lt_mask(inp_diff_ref,0.0)
            zero_inp_mask = hf.eq_mask(inp_diff_ref,0.0)
            concatenated_kernel = tf.concat([self.kernel, self.kernel[::-1,:]], axis=0)
            inp_mxts_increments = pos_inp_mask*(
                tf.matmul(self.get_pos_mxts(),
                          tf.transpose(concatenated_kernel)*(hf.gt_mask(tf.transpose(concatenated_kernel), 0.0)))
                + tf.matmul(self.get_neg_mxts(),
                            tf.transpose(concatenated_kernel)*(hf.lt_mask(tf.transpose(concatenated_kernel), 0.0)))) 
            inp_mxts_increments += neg_inp_mask*(
                tf.matmul(self.get_pos_mxts(),
                          tf.transpose(concatenated_kernel)*(hf.lt_mask(tf.transpose(concatenated_kernel), 0.0)))
                + tf.matmul(self.get_neg_mxts(),
                            tf.transpose(concatenated_kernel)*(hf.gt_mask(tf.transpose(concatenated_kernel), 0.0)))) 
            inp_mxts_increments += zero_inp_mask*(
                tf.matmul(0.5*(self.get_pos_mxts()
                               +self.get_neg_mxts()), tf.transpose(concatenated_kernel)))
            #pos_mxts and neg_mxts in the input get the same multiplier
            #because the breakdown between pos and neg wasn't used to
            #compute pos_contribs and neg_contribs in the forward pass
            #(it was based entirely on inp_diff_ref)
            return inp_mxts_increments, inp_mxts_increments
        else:
            raise RuntimeError("Unsupported mxts mode: "+str(self.dense_mxts_mode))

			
class GlobalAvgPool1D(SingleInputMixin, Node):

    def __init__(self, **kwargs):
        super(GlobalAvgPool1D, self).__init__(**kwargs) 

    def _compute_shape(self, input_shape):
        assert len(input_shape)==3
        shape_to_return = [None, input_shape[-1]] 
        return shape_to_return

    def _build_activation_vars(self, input_act_vars):
        return tf.reduce_mean(input_act_vars, axis=1)

    def _build_pos_and_neg_contribs(self):
        inp_pos_contribs, inp_neg_contribs =\
            self._get_input_pos_and_neg_contribs()
        pos_contribs = self._build_activation_vars(inp_pos_contribs)
        neg_contribs = self._build_activation_vars(inp_neg_contribs) 
        return pos_contribs, neg_contribs

    def _grad_op(self, out_grad):
        #input_activation_vars has shape [batch, input_length (width), num_filter (channel)]
        #out_grad has shape [batch, channel]
        width = self._get_input_activation_vars().get_shape().as_list()[1]
        mask = tf.ones_like(self._get_input_activation_vars()) / float(width)
        return tf.multiply(tf.expand_dims(out_grad, axis=1), mask)        

        #same as:
        #return tf.concat([tf.expand_dims(out_grad/float(width),1)] * width,1)

        #same as:
        #avg_pool_grad: grad: 4-D with shape `[batch, height, width, channels]`, output: 4-D. Gradients w.r.t. the input of avg_pool
        #return tf.squeeze(avg_pool_grad(
        #    orig_input_shape=tf.shape(tf.expand_dims(self._get_input_activation_vars(),1)), #add height dim
        #    grad=tf.expand_dims(tf.expand_dims(out_grad,1),1), #add width + height dim
        #    ksize=(1,1,width,1),
        #    strides=(1,1,1,1),
        #    padding=PaddingMode.valid),1)


    def _get_mxts_increments_for_inputs(self):
        pos_mxts_increments = self._grad_op(self.get_pos_mxts())
        neg_mxts_increments = self._grad_op(self.get_neg_mxts())
        return pos_mxts_increments, neg_mxts_increments

		
class GlobalMaxPool1D(SingleInputMixin, Node):

    '''
    Implementation from https://github.com/kundajelab/deeplift with
    modified backprop.
    '''
    def __init__(self, maxpool_deeplift_mode, **kwargs):
        super(GlobalMaxPool1D, self).__init__(**kwargs) 
        self.maxpool_deeplift_mode = maxpool_deeplift_mode

    def _compute_shape(self, input_shape):
        assert len(input_shape)==3
        shape_to_return = [None, input_shape[-1]] 
        return shape_to_return

    def _build_activation_vars(self, input_act_vars):
        return tf.reduce_max(input_act_vars, axis=1) 

    def _build_pos_and_neg_contribs(self):
        if (self.verbose):
            print("Heads-up: current implementation assumes maxpool layer "
                  "is followed by a linear transformation (conv/dense layer)")
        #placeholder; not used for linear layer, hence assumption above
        return tf.zeros_like(tensor=self.get_activation_vars(),
                      name="dummy_pos_cont_"+str(self.get_name())),\
               tf.zeros_like(tensor=self.get_activation_vars(),
                      name="dummy_neg_cont_"+str(self.get_name()))

    def _grad_op(self, out_grad):
        input_act_vars = self._get_input_activation_vars()
        mask = 1.0*tf.cast(
                tf.equal(tf.reduce_max(input_act_vars, axis=1, keepdims=True),
                        input_act_vars), dtype=tf.float32)
        #mask should sum to 1 across axis=1
        mask = mask/tf.reduce_sum(mask, axis=1, keepdims=True)
        return tf.multiply(tf.expand_dims(out_grad, axis=1), mask)

    def _get_mxts_increments_for_inputs(self):
        if (self.maxpool_deeplift_mode==MaxPoolDeepLiftMode.gradient):
            pos_mxts_increments = self._grad_op(self.get_pos_mxts())
            neg_mxts_increments = self._grad_op(self.get_neg_mxts())
        else:
            raise RuntimeError("Unsupported maxpool_deeplift_mode: "+
                               str(self.maxpool_deeplift_mode))
        return pos_mxts_increments, neg_mxts_increments





