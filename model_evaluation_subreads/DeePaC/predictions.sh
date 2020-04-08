#!/bin/bash

for length in {025..500..25} 
do
	dataPath=/home/genskeu/SCRATCH_NOBAK/train_val_test_data/test_data/test_data_nano_500_subread_"$length".npy
	modelPath=/home/genskeu/SCRATCH_NOBAK/models_lstm/builtinConfig/img-sensitive-lstm-nano-500-logs/nn-img-sensitive-lstm-nano-500-e014.h5
	outputPath=/home/genskeu/SCRATCH_NOBAK/predictions_lstm/builtinConfig/lstm_trained_on_500bp_reads_nanopore/e014/test_1_data_subread_"$length"_predictions.npy
	
	deepac predict -c $modelPath -a $dataPath -g 1 -o $outputPath
done
