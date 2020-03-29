#!/bin/bash

prediction_folder=$1
test_set_1=$2
test_set_2=$3
model_name=$(basename $1)

sed -i 's/RunName.*/RunName = '$model_name'/g' eval_config.ini
sed -i 's/DataSet.*/DataSet = '$test_set_1'/g' eval_config.ini
sed -i 's/PairedSet.*/PairedSet = '$test_set_2'/g' eval_config.ini

# get files
cp $prediction_folder/$test_set_1"_data"* ./
cp $prediction_folder/$test_set_2"_data"* ./


# rename
var=1
for length_1 in {025..250..25}
do
	for length_2 in {025..250..25}
	do
	printf -v epoch "%03d" $var
	cp $test_set_1"_data_subread_"$length_1"_predictions.npy" $model_name"-e"$epoch"-predictions-"$test_set_1".npy"
	cp $test_set_2"_data_subread_"$length_2"_predictions.npy" $model_name"-e"$epoch"-predictions-"$test_set_2".npy"
	var=$((var+1))
	done
done

# run eval
deepac eval -r eval_config_paired.ini

rm *.npy

