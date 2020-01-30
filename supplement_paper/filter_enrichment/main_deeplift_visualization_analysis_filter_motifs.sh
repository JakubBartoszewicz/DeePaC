#!/bin/bash

#Full pipeline to perform visualization and analysis of convolutional filter motifs:
#computes DeepLIFT filter scores on test data set, ranks filters according to their pathogenic/nonpathogenic potential, builds standard and weighted weblogos, compares them (correlation + IC), computes DeepLIFT nucleotide relevance scores and creates extended weblogos.

#main directory to save all results
deeplift_dir=$1
#model (.h5)
model_file=$2
#test data (.npy)
test_data_file=$3
#test labels (.npy)
test_labels_file=$4
#nonpatho reads test data (.fasta)
nonpatho_fasta_test_data=$5
#patho reads test data (.fasta)
patho_fasta_test_data=$6
#model predictions test data (.npy)
test_preds_file=$7
#train data (.npy)
train_data_file=$8


echo "Compute DeepLIFT filter contribution scores"
python3 deeplift_filter_contribution_scores.py  -m $model_file -b -t $test_data_file -n $nonpatho_fasta_test_data -p $patho_fasta_test_data -o $deeplift_dir -r N

echo "Plot filter contribution scores and rank filters according to theitr pathogenic/nonpathogenic potential"
python3 deeplift_filter_ranking_4_histograms.py -m original -f $deeplift_dir/filter_scores/ -y $test_labels_file -p $test_preds_file -o $deeplift_dir/filter_ranking/

# convert fasta file per filter to transfac file (count matrix)
echo "Build transfac files ..."
python3 fasta2transfac.py -i $deeplift_dir/fasta/ -o $deeplift_dir/transfac
cat $deeplift_dir"/"transfac/*.transfac > $deeplift_dir"/"transfac/all.transfac
echo "Build weighted transfac files ..."
python3 fasta2transfac.py -i $deeplift_dir/fasta/ -w -d $deeplift_dir/filter_scores/ -o $deeplift_dir/transfac_weighted 
cat $deeplift_dir"/"transfac_weighted/*.transfac > $deeplift_dir"/"transfac_weighted/all.transfac


echo "Build standard weblogos ..."
python3 weblogos.py -i  $deeplift_dir/transfac -o $deeplift_dir/weblogos_standard -f .transfac -t "$train_data_file"
echo "Build weighted weblogos ..."
python3 weblogos.py -i  $deeplift_dir/transfac_weighted -o $deeplift_dir/weblogos_weighted -f .transfac -t "$train_data_file"


echo "Compute IC of standard and weighted weblogos"
python3 IC_from_transfac.py -i  $deeplift_dir/transfac/all.transfac -t "$train_data_file" -o $deeplift_dir/IC_standard_vs_weighted_weblogo/IC_transfac.txt
python3 IC_from_transfac.py -i  $deeplift_dir/transfac_weighted/all.transfac -t "$train_data_file" -o $deeplift_dir/IC_standard_vs_weighted_weblogo/IC_transfac_weighted.txt


echo "Compute correlation between standard and weighted filter motifs"
python3 motif_comparison_biopython.py -q $deeplift_dir/transfac/all.transfac -t $deeplift_dir/transfac_weighted/all.transfac -o $deeplift_dir/motif_comparison_standard_vs_weighted


echo "Compute DeepLIFT nucleotide contribution scores"
python3 deeplift_nucleotide_contribution_scores.py -m $model_file -b -t $test_data_file -i $deeplift_dir/filter_scores -o $deeplift_dir -r N


echo "Build weighted extended weblogos ..."
python3 weblogos_extended.py -f $deeplift_dir/fasta/ -s $deeplift_dir/filter_scores/ -l  $deeplift_dir/nucleotide_scores/ -t "$train_data_file" -o $deeplift_dir/weblogos_weighted_extended
