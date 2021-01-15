from tensorflow.keras.preprocessing.text import Tokenizer
from Bio import SeqIO
import numpy as np
import time
import tensorflow as tf
import tensorflow.keras.backend as K

config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
                        allow_soft_placement=True, device_count = {'CPU': 1})
session = tf.Session(config=config)
K.set_session(session)

alphabet = "ACTG"
seq_dim = len(alphabet) + 1
tokenizer = Tokenizer(char_level = True)
tokenizer.fit_on_texts(alphabet)


start_time = time.time()

neg_train = list(SeqIO.parse("SCRATCH_NOBAK/nonpathogenic_train_1_trimmed.fasta", "fasta"))
pos_train = list(SeqIO.parse("SCRATCH_NOBAK/pathogenic_train_1_trimmed.fasta", "fasta"))

x_train = np.array([tokenizer.texts_to_matrix(x) for x in neg_train] + [tokenizer.texts_to_matrix(x) for x in pos_train])

end_time = time.time()
time = end_time - start_time

print(time)
print(time/len(y_test))