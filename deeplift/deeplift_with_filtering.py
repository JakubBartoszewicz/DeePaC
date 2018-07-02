from deeplift.models import *
from deeplift.layers.convolutional import *
from deeplift.layers.pooling import *

from tensorflow.python.ops.gen_nn_ops import avg_pool_grad


class SequentialModelFilter(SequentialModel):
	'''
	Based on implementation from https://github.com/kundajelab/deeplift/tree/keras2compat
	Extended by the option to specify the convolutional layer and filter neurons though which 
        the relevance scores should be passed during the backward pass.
	'''
	def _get_func2(self, find_scores_layers, 
							target_layer, conv_layer,
							input_layers, func_type,
							slice_objects=None):
			if isinstance(find_scores_layers,list)==False:
				remove_list_wrapper_on_return = True
				find_scores_layers = [find_scores_layers] 
			else:
				remove_list_wrapper_on_return = False
			for find_scores_layer in find_scores_layers:
				find_scores_layer.reset_mxts_updated()
			self._set_scoring_mode_for_target_layer(target_layer)
			self._set_filter_index_conv_layer(conv_layer)
			for find_scores_layer in find_scores_layers:
				find_scores_layer.update_mxts()
			if (func_type == FuncType.contribs):
				output_symbolic_vars = [
				 find_scores_layer.get_target_contrib_vars() for find_scores_layer
				 in find_scores_layers]
			elif (func_type == FuncType.multipliers):
				output_symbolic_vars = [
				 find_scores_layer.get_mxts() for find_scores_layer in
				 find_scores_layers]
#			elif (func_type == FuncType.multipliers_filter):
#				output_symbolic_vars = [
#				 find_scores_layer._get_mxts_increments_for_inputs() for find_scores_layer in
#				 find_scores_layers]
			elif (func_type == FuncType.contribs_of_input_with_filter_refs):
				output_symbolic_vars =\
				 [find_scores_layer.get_contribs_of_inputs_with_filter_refs()
				  for find_scores_layer in find_scores_layers]
			else:
				raise RuntimeError("Unsupported func_type: "+func_type)
			if (slice_objects is not None):
				output_symbolic_vars = output_symbolic_vars[slice_objects]
			core_function = compile_func([input_layer.get_activation_vars()
										for input_layer in input_layers]+
									   [input_layer.get_reference_vars()
										for input_layer in input_layers],
									   output_symbolic_vars)
			def func(task_idx, input_data_list,
					 progress_update,
					 input_references_list=None, filter_index = None, batch_size = 1):
				if (isinstance(input_data_list, dict)):
					assert hasattr(self, '_input_layer_names'),\
					 ("Dictionary supplied for input_data_list but model does "
					  "not have an attribute '_input_layer_names")
					input_data_list = [input_data_list[x] for x in
									   self._input_layer_names]
				if (input_references_list is None):
					print("No reference provided - using zeros")
					input_references_list = [0.0 for x in input_data_list]
				if (isinstance(input_references_list, dict)):
					assert hasattr(self, '_input_layer_names'),\
					 ("Dictionary supplied for input_references_list but model "
					  "does not have an attribute '_input_layer_names")
					input_references_list = [input_references_list[x] for x in
											 self._input_layer_names]
				input_references_list = [
					np.ones_like(input_data)*reference
					for (input_data, reference) in
					zip(input_data_list, input_references_list)]
				#WARNING: this is not thread-safe. Do not try to
				#parallelize or you can end up with multiple target_layers
				#active at once
				target_layer.set_active()
				target_layer.update_task_index(task_idx)
				if (filter_index is None):
					self._set_filter_index_conv_layer(conv_layer)
				to_return = run_function_in_batches( #changed
						func = core_function,
						input_data_list = input_data_list+input_references_list,
						batch_size = batch_size,
						progress_update = progress_update,
						multimodal_output=True,
						filter_index = filter_index,
						conv_layer = conv_layer)
				target_layer.set_inactive()
				if (remove_list_wrapper_on_return):
					#remove the enclosing []; should be only one element
					assert len(to_return)==1
					to_return = to_return[0]
				return to_return
			return func

	def _set_filter_index_conv_layer(self, conv_layer):
		conv_layer._init_filter_index()

        
	def _get_func(self, find_scores_layer_idx, target_layer_idx=-2, conv_layer_idx = -1, **kwargs):
		'''
		Based on implementation from https://github.com/kundajelab/deeplift/tree/keras2compat
		Extended by the option to specify the convolutional layer for which filtering during the
		backward pass should be perfomed.
		'''
		if (isinstance(find_scores_layer_idx, list)):
			find_scores_layers = [self.get_layers()[x] for x in find_scores_layer_idx]
		else:
			find_scores_layers = self.get_layers()[find_scores_layer_idx]
		if (conv_layer_idx == -1 and "Conv1DFilter" in [type(layer).__name__ for layer in self.get_layers()]):
			conv_layer_idx = [type(layer).__name__ for layer in self.get_layers()].index("Conv1DFilter")
		return self._get_func2(find_scores_layers=find_scores_layers, target_layer=self.get_layers()[target_layer_idx], conv_layer=self.get_layers()[conv_layer_idx], input_layers=[self.get_layers()[0]], **kwargs) #changed

class Conv1DFilter(Conv1D):
    '''
    Based on implementation from https://github.com/kundajelab/deeplift/tree/keras2compat
    Extended by the option to backpropagate only relevance scores through certain filter neurons.
    '''

    def _init_filter_index(self):
        '''
	Initial configuration: backprogate relevance scores through all filter neurons (no filtering)
        '''
        self.filter = None
        self.mask = tf.Variable(np.ones_like(self._pos_mxts), dtype = tf.float32)
        deeplift.util.get_session().run(tf.variables_initializer([self.mask]))

    def set_filter_index(self, mask, mask_update, y):
        '''
        Fill filter mask per input batch.
        '''
        deeplift.util.get_session().run(mask_update, feed_dict={y:mask})

    def _get_mxts_increments_for_inputs(self):
        '''
        Based on implementation from https://github.com/kundajelab/deeplift/tree/keras2compat
        Multiplication of deeplift multipliers with a filter mask containing ones or zeros to
        backpropagate only relevance scores of filter neurons where the corresponding entry of
        the filter mak is one.
        '''
        pos_mxts = self.get_pos_mxts()
        neg_mxts = self.get_neg_mxts()
        pos_mxts = self.mask * pos_mxts
        neg_mxts = self.mask * neg_mxts
        inp_diff_ref = self._get_input_diff_from_reference_vars() 
        output_shape = self._get_input_shape()
        if (self.conv_mxts_mode == ConvMxtsMode.Linear): 
            pos_inp_mask = hf.gt_mask(inp_diff_ref,0.0)
            neg_inp_mask = hf.lt_mask(inp_diff_ref,0.0)
            zero_inp_mask = hf.eq_mask(inp_diff_ref,0.0)
            inp_mxts_increments = pos_inp_mask*(
                conv1d_transpose_via_conv2d(
                    value=pos_mxts,
                    kernel=self.kernel*(hf.gt_mask(self.kernel,0.0)),
                    tensor_with_output_shape=self.inputs.get_activation_vars(),
                    padding=self.padding,
                    stride=self.stride)
                +conv1d_transpose_via_conv2d(
                    value=neg_mxts,
                    kernel=self.kernel*(hf.lt_mask(self.kernel,0.0)),
                    tensor_with_output_shape=self.inputs.get_activation_vars(),
                    padding=self.padding,
                    stride=self.stride))
            inp_mxts_increments += neg_inp_mask*(
                conv1d_transpose_via_conv2d(
                    value=pos_mxts,
                    kernel=self.kernel*(hf.lt_mask(self.kernel,0.0)),
                    tensor_with_output_shape=self.inputs.get_activation_vars(),
                    padding=self.padding,
                    stride=self.stride)
                +conv1d_transpose_via_conv2d(
                    value=neg_mxts,
                    kernel=self.kernel*(hf.gt_mask(self.kernel,0.0)),
                    tensor_with_output_shape=self.inputs.get_activation_vars(),
                    padding=self.padding,
                    stride=self.stride))
            inp_mxts_increments += zero_inp_mask*(
                conv1d_transpose_via_conv2d(
                    value=0.5*(neg_mxts+pos_mxts),
                    kernel=self.kernel,
                    tensor_with_output_shape=self.inputs.get_activation_vars(),
                    padding=self.padding,
                    stride=self.stride))
            pos_mxts_increments = inp_mxts_increments
            neg_mxts_increments = inp_mxts_increments
        else:
            raise RuntimeError("Unsupported conv mxts mode: "
                               +str(self.conv_mxts_mode))
        return pos_mxts_increments, neg_mxts_increments


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



def run_function_in_batches(func,
                            input_data_list,
                            learning_phase=None,
                            batch_size=10,
                            progress_update=1000,
                            multimodal_output=False,
                            filter_index = None,
                            conv_layer = None):
    '''
    Based on implementation from https://github.com/kundajelab/deeplift/tree/keras2compat
    Passes a filtermask which allows backpropagating of relevance scores through selected
    filter neurons in convolutional layer.
    '''
    #func has a return value such that the first index is the
    #batch. This function will run func in batches on the inputData
    #and will extend the result into one big list.
    #if multimodal_output=True, func has a return value such that first
    #index is the mode and second index is the batch
    assert isinstance(input_data_list, list), "input_data_list must be a list"
    #input_datas is an array of the different input_data modes.
    to_return = [];
    i = 0;
    #build assign operation to later fill filter mask
    if (conv_layer != None):
        y = tf.placeholder(dtype = tf.float32)
        mask_update = tf.assign(conv_layer.mask, y, validate_shape = False)
    while i < len(input_data_list[0]):
        if (progress_update is not None):
            if (i%progress_update == 0):
                print("Done",i)
        #fill filter mask with ones for filter neurons through which which relevance scores should be based,
        #zero otherwise
        if (filter_index is not None):
            this_filter = filter_index[i:i+batch_size]
            mask_shape = conv_layer._shape
            mask_shape[0] = len(this_filter)
            mask = np.zeros(mask_shape)
            for j in range(len(this_filter)):
                mask[j, this_filter[j][0], this_filter[j][1]] = 1
            conv_layer.set_filter_index(mask, mask_update, y)

        func_output = func(([x[i:i+batch_size] for x in input_data_list]
                                +([] if learning_phase is
                                   None else [learning_phase])
                        ))
        if (multimodal_output):
            assert isinstance(func_output, list),\
             "multimodal_output=True yet function return value is not a list"
            if (len(to_return)==0):
                to_return = [[] for x in func_output]
            for to_extend, batch_results in zip(to_return, func_output):
                to_extend.extend(batch_results)
        else:
            to_return.extend(func_output)
        i += batch_size;
    return to_return
