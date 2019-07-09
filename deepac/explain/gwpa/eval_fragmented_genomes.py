import os
import argparse
import numpy as np
from keras.models import load_model
from rc_layers import RevCompConv1D, RevCompConv1DBatchNorm, DenseAfterRevcompWeightedSum, DenseAfterRevcompConv1D


#parse command line options
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True, help="Model (.h5)")
parser.add_argument("-t", "--test_data", required=True, help="Test data set (.npy)")
parser.add_argument("-o", "--out_dir", required=True, help="Output directory")
args = parser.parse_args()

#create output directory
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

model = load_model(args.model, custom_objects={'RevCompConv1D': RevCompConv1D, 'RevCompConv1DBatchNorm': RevCompConv1DBatchNorm, 'DenseAfterRevcompWeightedSum': DenseAfterRevcompWeightedSum, 'DenseAfterRevcompConv1D': DenseAfterRevcompConv1D})

# Read data to memory
x_test = np.load(args.test_data)

# Predict class probabilities
y_pred =  np.ndarray.flatten(model.predict_proba(x_test))

np.save(file = args.out_dir + "/" + os.path.splitext(os.path.basename(args.test_data))[0] + "_predictions.npy", arr = y_pred)
