#!/bin/bash

pred_model_1=$1
pred_model_2=$2
run_name_1=$(basename $1)
run_name_2=$(basename $2)
test_set=$3
paired_set=$4

sed -i 's/EnsembleName.*/EnsembleName = '$run_name_1'_'$run_name_2'/g' eval_ens_config.ini
sed -i 's/RunNames.*/RunNames = '$run_name_1','$run_name_2'/g' eval_ens_config.ini
sed -i 's/DataSet.*/DataSet = '$test_set'/g' eval_ens_config.ini
sed -i 's/PairedSet.*/PairedSet = '$paired_set'/g' eval_ens_config.ini




# get files
cp $1/$test_set"_data_"* ./
cp $1/$paired_set"_data_"* ./

# rename
var=1
for length in {025..250..25}
do
	for length_2 in {025..250..25}
	do
	printf -v epoch "%03d" $var
	cp $test_set"_data_subread_"$length"_predictions.npy" $run_name_1"-e"$epoch"-predictions-"$test_set".npy"
	cp $paired_set"_data_subread_"$length_2"_predictions.npy" $run_name_1"-e"$epoch"-predictions-"$paired_set".npy"
	var=$((var+1))
	done
done


# get files
cp $2/$test_set"_data_"* ./
cp $2/$paired_set"_data_"* ./

# rename
var=1
for length in {025..250..25}
do
	for length_2 in {025..250..25}
	do
	printf -v epoch "%03d" $var
	cp $test_set"_data_subread_"$length"_predictions.npy" $run_name_2"-e"$epoch"-predictions-"$test_set".npy"
	cp $paired_set"_data_subread_"$length_2"_predictions.npy" $run_name_2"-e"$epoch"-predictions-"$paired_set".npy"
	var=$((var+1))
	done
done

for epoch in {1..100}
do
	# run eval
	sed -i 's/Epoch =.*/Epoch = '$epoch','$epoch'/g' eval_ens_config.ini
	deepac eval -e eval_ens_config.ini
done


# rm temp files 
rm *.npy


