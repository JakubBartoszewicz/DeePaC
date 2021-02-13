import numpy as np
for left in range(25, 250, 25):
    print(left)
    dat = np.loadtxt("query_1_{}.4knn.preds".format(left), skiprows=1, usecols=2)
    dat = dat.reshape((dat.shape[0], 1))
    np.save("knn-e001-predictions-test_1_{}.npy".format(left), dat)
print("right")
for right in range(25, 250, 25):
    print(right)
    dat_1 = np.loadtxt("predict_result_500", skiprows=1, usecols=2)
    dat_1 = dat_1.reshape((dat.shape[0], 1))
    dat_2 = np.loadtxt("query_2_{}.4knn.preds".format(right), skiprows=1, usecols=2)
    dat_2 = dat_2.reshape((dat.shape[0], 1))
    dat = np.zeros(dat_1.shape)
    for i in range(dat_1.shape[0]):
        if np.isnan(dat_1[i]):
            dat[i,0] = dat_2[i]
        elif np.isnan(dat_2[i]):
            dat[i,0] = dat_1[i]
        else:
            if dat_1[i] == dat_2[i]:
                dat[i,0] = dat_1[i]
            else:
                dat[i,0] = np.nan
    np.save("knn-e001-predictions-test_ext_{}.npy".format(right), dat)

