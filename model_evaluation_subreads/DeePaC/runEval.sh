#!/bin/bash

prediction_folder=$1
test_set=$2
model_name=$3

sed -i 's/RunName.*/RunName = '$model_name'/g' eval_config.ini
sed -i 's/DataSet.*/DataSet = '$test_set'/g' eval_config.ini

# get files
cp $prediction_folder/$test_set"_data_"* ./

# rename
var=1
for length in {025..250..25}
do
	printf -v epoch "%03d" $var
	mv $test_set"_data_subread_"$length"_predictions.npy" $model_name"-e"$epoch"-predictions-"$test_set".npy"
	var=$((var+1))
done

# run eval
deepac eval -r eval_config.ini

# rm temp files 
rm $model_name"-e"*
