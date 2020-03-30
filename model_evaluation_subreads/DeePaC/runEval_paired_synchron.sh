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
	if [ $length_1 == "025" ] || [ $length_1 == "075" ] || [ $length_1 == "125" ] || [ $length_1 == "175" ] || [ $length_1 == "225" ]
	then
		var=$((var+1))
	else
		printf -v epoch "%03d" $var
		mv $test_set_1"_data_subread_"$length_1"_predictions.npy" $model_name"-e"$epoch"-predictions-"$test_set_1".npy"
		mv $test_set_2"_data_subread_"$length_1"_predictions.npy" $model_name"-e"$epoch"-predictions-"$test_set_2".npy"
		# run eval
		sed -i 's/EpochStart =.*/EpochStart = '$var'/g' eval_config.ini
		var=$((var+1))
		sed -i 's/EpochEnd =.*/EpochEnd = '$var'/g' eval_config.ini

		deepac eval -r eval_config.ini
	fi
done


rm *.npy
