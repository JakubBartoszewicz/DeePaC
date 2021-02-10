import numpy as np

def conv(fld):
    return np.nan if fld == b'na' else float(fld)

infile_0 = "bowtie_neg_1_labs.txt"
infile_1 = "bowtie_pos_1_labs.txt"

dat_0 = np.loadtxt(infile_0, converters={0: conv})
dat_1 = np.loadtxt(infile_1, converters={0: conv})

dat = np.concatenate((dat_0, dat_1), axis=0)
dat = dat.reshape((dat.shape[0], 1))

np.save("bowtie-e001-predictions-all_test_1.npy", dat)

