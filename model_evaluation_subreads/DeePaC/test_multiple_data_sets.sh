#!/bin/bash

for length in {50,100,150,200,250,300,400,500,600,800,1000,1300,1600,2000,2500,3200,4000,5000,6300,8000,10000,12700,16000,20000}
do
	./runEval_paired_synchron.sh ./predictions/lstm_250bp_d02_img/ test_1_fl"$length" test_2_fl"$length"
done
