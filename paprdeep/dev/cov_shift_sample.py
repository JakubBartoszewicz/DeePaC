# resamples and reshuffles the datasets for covariate shift test. negative = training, positive = test
import math
import numpy as np
np.random.seed(0)

print("Loading test data...")
# load the test set and permute the reads to get random order
x_test = np.random.permutation(np.load("SCRATCH_NOBAK/test_data_1_bal.npy"))
size = x_test.shape[0]

print("Loading training data...")
# load the training set
x_train = np.load("SCRATCH_NOBAK/train_data_1_10e7.npy")

print("Sampling data...")
# sample the reads in the training set to get equal numbers of reads from test and training sets (and equal numbers of reads in each class assuming balanced input datasets)
# then permute the reads to get random order
x_train_sample = np.random.permutation(x_train[np.random.choice(x_train.shape[0], size, replace=False),:,:])

print("Preparing new training data...")
# assign 80% of old sets to the new training dataset
train_size = math.ceil(size*0.8)
x_train_new = np.concatenate((x_train_sample[:train_size,:,:], x_test[:train_size,:,:]))
y_train_new = np.concatenate((np.repeat(0,train_size),np.repeat(1,train_size)))

print("Preparing new test data...")
# assign 20% of old sets to the new test dataset
test_size = size - train_size
x_test_new = np.concatenate((x_train_sample[train_size:,:,:], x_test[train_size:,:,:]))
y_test_new = np.concatenate((np.repeat(0,test_size),np.repeat(1,test_size)))

# save
print("Saving data...")
np.save(file = "SCRATCH_NOBAK/cov_test_data_1.npy", arr = x_test_new)
np.save(file = "SCRATCH_NOBAK/cov_test_labels_1.npy", arr = y_test_new)
np.save(file = "SCRATCH_NOBAK/cov_train_data_1.npy", arr = x_train_new)
np.save(file = "SCRATCH_NOBAK/cov_train_labels_1.npy", arr = y_train_new)