#!/bin/bash

pred_model_1=$1
pred_model_2=$2
run_name_1=$(basename $1)
run_name_2=$(basename $2)
test_set=$3
paired_set= $4

sed -i 's/EnsembleName.*/EnsembleName = '$run_name_1'_'$run_name_2'/g' eval_ens_config.ini
sed -i 's/RunNames.*/RunNames = '$run_name_1','$run_name_2'/g' eval_ens_config.ini
sed -i 's/DataSet.*/DataSet = '$test_set'/g' eval_ens_config.ini
if [ -z ${paired_set+x} ]
	then
		sed -i 's/PairedSet.*/PairedSet = '$test_set_2'/g' eval_ens_config.ini
	else
		sed -i 's/PairedSet.*/PairedSet = none/g' eval_ens_config.ini
fi



# get files
cp $1/$test_set"_data_"* ./

# rename
var=1
for length in {025..250..25}
do
	printf -v epoch "%03d" $var
	mv $test_set"_data_subread_"$length"_predictions.npy" $run_name_1"-e"$epoch"-predictions-"$test_set".npy"
	var=$((var+1))
done

# get files
cp $2/$test_set"_data_"* ./

# rename
var=1
for length in {025..250..25}
do
	printf -v epoch "%03d" $var
	mv $test_set"_data_subread_"$length"_predictions.npy" $run_name_2"-e"$epoch"-predictions-"$test_set".npy"
	var=$((var+1))
done

for epoch in {1..10}
do
	# run eval
	sed -i 's/Epoch =.*/Epoch = '$epoch','$epoch'/g' eval_ens_config.ini
	deepac eval -e eval_ens_config.ini
done


# rm temp files 
rm *.npy


