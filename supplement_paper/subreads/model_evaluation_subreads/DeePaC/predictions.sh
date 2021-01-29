#!/bin/bash

for length in {025..250..25} 
do
	dataPath=/home/genskeu/SCRATCH_NOBAK/train_val_test_data/test_data/test_1_data_subread_"$length".npy
	modelPath=/home/genskeu/SCRATCH_NOBAK/models_lstm/builtinConfig/img-sensitive-lstm-nano-500-logs/nn-img-sensitive-lstm-nano-500-e014.h5
	mkdir $(basename $modelPath)
	outputPath=$(basename $modelPath)/test_1_data_subread_"$length"_predictions.npy
	
	deepac predict -c $modelPath -a $dataPath -g 1 -o $outputPath
done
