

#Install deeplift from https://github.com/kundajelab/deeplift/blob/keras2compat
import deeplift
from deeplift.util import compile_func

import numpy as np
from keras.models import load_model
import deeplift_with_filtering_conversion as conversion
from rc_layers import RevCompConv1D, RevCompConv1DBatchNorm, DenseAfterRevcompWeightedSum, DenseAfterRevcompConv1D

###Creates deeplift model from keras model and checks whether the conversion was successful by comparing the predictions of the original model and the deeplift model

#model_file = "nn-f128_l15_d64avg_pool-e001.h5"
model_file = "nn-RC_f128_l15_d64avg_pool-e001.h5"
#model_file = "cnn-10e7-new-fold1-e09.h5"
test_data_file = "/scratch/seidela/data/test_data.npy"

#creates keras model and loads weights
print("Loading keras model ...")
model = load_model(model_file, custom_objects={'RevCompConv1D': RevCompConv1D, 'RevCompConv1DBatchNorm': RevCompConv1DBatchNorm, 'DenseAfterRevcompWeightedSum': DenseAfterRevcompWeightedSum, 'DenseAfterRevcompConv1D': DenseAfterRevcompConv1D})
#convert keras model to deeplift model
print("Loading deeplift model ...")
deeplift_model = conversion.convert_model_from_saved_files(model_file, nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault) 
print(deeplift_model.get_layers())
#print(model.get_layer(index=0).get_weights()[0].shape)
#print(model.get_layer(index=0).get_weights()[1].shape)

print("Loading test data ...")
x_test = np.load(test_data_file)
x_test = x_test[0:1000,:,:]
#keras model predictions
print("Making predictions with the keras model ...")
y_pred =  np.ndarray.flatten(model.predict_proba(x_test))

#deeplift model predictions
#compile forward propagation function: input: activations of the input layer; output = activations of the output layer
print("Making predictions with the deeplift model ...")
deeplift_prediction_func = compile_func([deeplift_model.get_layers()[0].get_activation_vars()],
                                         deeplift_model.get_layers()[-1].get_activation_vars())
y_pred_deeplift = np.ndarray.flatten(np.array(deeplift.util.run_function_in_batches(input_data_list=[x_test], func=deeplift_prediction_func, batch_size=1, progress_update=10000)))

#compare model predictions
print("maximum difference in predictions: ", np.max(np.abs(y_pred - y_pred_deeplift)))





