#!/bin/bash

pred_model_1=$1
pred_model_2=$2
run_name_1=$3
run_name_2=$4
test_set=$5

sed -i 's/EnsembleName.*/EnsembleName = '$run_name_1'_'$run_name_2'/g' eval_ens_config.ini
sed -i 's/RunNames.*/RunNames = '$run_name_1','$run_name_2'/g' eval_ens_config.ini
sed -i 's/DataSet.*/DataSet = '$test_set'/g' eval_ens_config.ini

# get files
cp $1/"$test_set"_data_* ./

# rename
var=2
for length in {025..250..25}
do
	printf -v epoch "%03d" $var
	mv "$test_set"_data_subread_"$length"_predictions.npy "$run_name_1"-e"$epoch"-predictions-"$test_set".npy
	var=$((var+1))
done

# get files
cp $2/"$test_set"_data_* ./

# rename
var=2
for length in {025..250..25}
do
	printf -v epoch "%03d" $var
	mv "$test_set"_data_subread_"$length"_predictions.npy "$run_name_2"-e"$epoch"-predictions-"$test_set".npy
	var=$((var+1))
done

for epoch in {2..11}
do
	# run eval
	sed -i '/Epoch =/c\Epoch = '$epoch,$epoch eval_ens_config.ini
	deepac eval -e eval_ens_config.ini
done


# rm temp files 
rm "$run_name_1"-e*
rm "$run_name_2"-e*

