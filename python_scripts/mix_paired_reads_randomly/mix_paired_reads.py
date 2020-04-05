import csv
import numpy as np
import os

# script to rearange the numpy predictions 
# was used to investigate the paired read effect
# the npy files contain the data from all bacteria strains used                                                    
# the order of the strains needs to be maintained 
# shuffeling only needs to occure within one spec.
  
fragment_lengths = [250,300,400,500,600,800,1000,1300,1600,2000,2500,3200,4000,5000,6300,8000,10000,12700,16000,20000]

for fl in fragment_lengths:
    read_numbers = []
    read_number_files = [os.path.join("read_numbers_new_test_data","p_fl_250img_test_NP_reads.csv"),
                         os.path.join("read_numbers_new_test_data","p_fl_250img_test_HP_reads.csv")]

    for file in read_number_files:
        with open(file) as csvfile:
             reader = csv.reader(csvfile, delimiter=',')
             for row in reader:
                 read_numbers.append(row[1])

    start_index = 0
    order = []
    for number in read_numbers:
        number = int(number)
        order_species = np.array(range(start_index,start_index+number))
        np.random.shuffle(order_species)
        order += list(order_species)
        start_index += number

    npy_file = "/home/uli/Documents/RKI/deepac_analysis/pairedRead_effect/paired_data/normal_order/lstm/read_length_250bp/test_1_data_fl"+str(fl)+"_predictions.npy"
    data = np.load(npy_file)
    np.save("/home/uli/Documents/RKI/deepac_analysis/pairedRead_effect/paired_data/random_order/lstm/read_length_250bp/test_1_data_fl"+str(fl)+"_predictions.npy",data[order])
