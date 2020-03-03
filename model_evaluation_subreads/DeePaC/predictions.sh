#!/bin/bash

for sub_length in {50,100,150,200}
do
	for fl_length in {50,100,150,200} 
	do
 
		dataPath=~/SCRATCH_NOBAK/paired_data/test_data_npy_read_length_"$sub_length"bp/p_fl_"$fl_length"/test_1_data.npy
 		outputPath=~/SCRATCH_NOBAK/predictions_lstm/lstm_trained_on_250bp_reads/paired_data/test_1_data_fl"$fl_length"_subread_"$sub_length"_predictions.npy
       
        	deepac predict -s -a $dataPath -g 1 -o $outputPath 

		dataPath=~/SCRATCH_NOBAK/paired_data/test_data_npy_read_length_"$sub_length"bp/p_fl_"$fl_length"/test_2_data.npy
		outputPath=~/SCRATCH_NOBAK/predictions_lstm/lstm_trained_on_250bp_reads/paired_data/test_2_data_fl"$fl_length"_subread_"$sub_length"_predictions.npy
       
     	  	deepac predict -s -a $dataPath -g 1 -o $outputPath 

	done
done
