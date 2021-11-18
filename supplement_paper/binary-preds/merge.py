import numpy as np

infile_1 = "bowtie-e001-predictions-all_test_1.npy"
infile_2 = "bowtie-e001-predictions-all_test_2.npy"

dat_1 = np.load(infile_1)
dat_2 = np.load(infile_2)

dat_1 = dat_1.reshape((dat_1.shape[0], 1))
dat_2 = dat_2.reshape((dat_2.shape[0], 1))

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

dat = dat.reshape((dat.shape[0], 1))

np.save("bowtie-e001-predictions-all_test_merged.npy", dat)

