#!/bin/bash

#Fragments genomes, converts fragments to .npy and makes model predictions per fragment, build bedgraph showing genome wide pathogenicity, computing genome wide filter activation values

#directory with species in .fasta
genomes_dir="/mnt/wissdaten/MF1_User/SeidelA/data/IMGdata/case_studies/fasta/"
#directory with species annotation in .gb (genbank-format)
gbk_dir="/mnt/wissdaten/MF1_User/SeidelA/data/IMGdata/case_studies/gbk/"
model="/mnt/wissdaten/MF1_User/SeidelA/nn-RC_f256_l15_d128max_bf_conv_dropout_pool_train_aug-e003.h5"
num_filters=512
#length of genome fragments
read_len=250
#shift per fragment
shift=50
complete_GO="goa_uniprot_all.gaf.gz"
go_basic="go-basic.obo"

#output directories:
#main out dir
out_dir="/mnt/wissdaten/MF1_User/SeidelA/data/IMGdata/case_studies/"
#directory with species annotation in .gff
gff_dir="$out_dir""/gff/"
#directory with species in .genome
genome_sizes_dir="$out_dir""/genomes/"
#directory with GO-annotations (.gaf) per species
GO_dir="$out_dir""/GO/"
#genomes fragmented .fasta
genomes_fragmented_dir="$out_dir""/genomes_fragmented_shift_"$shift"/"
#predictions per fragment .npy
genomes_fragmented_preds_dir="$out_dir""/genomes_fragmented_shift_"$shift"_predictions/"
#genome wide filter activations
filter_activation_dir="$out_dir""/genomes_fragmented_shift_"$shift"_filter_activations/"
#genome wide pathogenicity scores (.bedgraph)
genomes_pathogenicity_dir="$out_dir""/genomes_fragmented_shift_"$shift"_pathogenicity/"
#GO-enrichment analysis of highly pathogenic genes
patho_genes_GO_enrichment_dir="$out_dir""/genomes_fragmented_shift_50_GO_pathogenic_genes_enrichment_analysis"
#filter genomic enrichment analysis
filter_genomic_enrichment_dir="$out_dir""/genomes_fragmented_shift_"$shift"_genome_filter_enrichment_analysis/"
#GO-enrichment analysis of genomic enriched filters
filter_GO_enrichment_dir="$out_dir""/genomes_fragmented_shift_"$shift"_GO_filter_enrichment_analysis/"


##Preprocessing###
echo "Perform some preprocessing steps"
echo "Converting genbank files to gff"
if [ ! -d "$gff_dir" ]; then
	mkdir -p "$gff_dir"
fi
./gbk2gff.sh $gbk_dir $gff_dir

echo "Building genome info files from gff files"
./gff2genome.sh $gff_dir $genome_sizes_dir

echo "Extracting all GO-terms which are associated with the selected species (.gaf)"
if [ ! -d "$GO_dir" ]; then
      mkdir -p "$GO_dir"
fi
for genome_gff in $gff_dir/*gff; do
	taxon=$( grep -Po "taxon:[0-9]+" "$genome_gff" | uniq)
	zcat $complete_GO | grep -v "^!" | awk -F'\t' -v taxon=$taxon '{if ($13 == taxon) print $0}' > "$GO_dir"/$(basename "$genome_gff" .gff).gaf
done

echo "Fragmenting genomes ..."
python3 fragments_genomes_new.py -g "$genomes_dir" -r "$read_len" -s "$shift" -o "$genomes_fragmented_dir"



###Genome-wide pathogenicity analysis (GWPA)###
echo "Perform GWPA"
echo "Making model predictions per fragment ..."
for genome_reads in "$genomes_fragmented_dir"/*npy; do
	echo Processing "$genome_reads"...
	python3 eval_fragmented_genomes.py -m "$model" -t "$genome_reads" -o "$genomes_fragmented_preds_dir"
done

echo "Building bedgraph file per genome representing its predicted pathogenicity ..."
python3 genomes_pathogenicity_new.py -f "$genomes_fragmented_dir" -p "$genomes_fragmented_preds_dir" -g "$genome_sizes_dir" -o "$genomes_pathogenicity_dir"

echo "Computing mean pathogenicity per genome ..."
> "$genomes_pathogenicity_dir"/temp.txt
for genome_bedgraph in $genomes_pathogenicity_dir/*bedgraph; do
	mean=$(awk -F'\t' -v total_num_bases=0 -v total_score=0 -v num_rows=$(wc -l "$genome_bedgraph" | cut -d' ' -f1) '{len=$3-$2; score = len*$4; total_num_bases = total_num_bases + len; total_score=total_score + score; if (NR == num_rows) {print total_score/total_num_bases}}' "$genome_bedgraph")
	echo -e $(basename "$genome_bedgraph" _fragmented_genomes_pathogenicity.bedgraph)'\t'$mean >> "$genomes_pathogenicity_dir"/temp.txt
done
#sort species by pathogenicity score
sort -t$'\t' -k2,2rn temp.txt > "$genomes_pathogenicity_dir"/summary.txt
rm "$genomes_pathogenicity_dir"/temp.txt

echo "Ranking genes according to their mean pathogenicity score ..."
python3 pathogenicity_gene_ranking.py -p "$genomes_pathogenicity_dir" -g "$gff_dir" -o "$genomes_pathogenicity_dir""/gene_ranking/"

echo "Performing GO-enrichment analysis of above mean pathogenic genes"
./GO_pathogenicity_new.sh $GO_dir $go_basic $genomes_pathogenicity_dir $patho_genes_GO_enrichment_dir upper mean




###Convolutional filter enrichment analysis###
echo "Perform convolutional filter enrichment analysis"
echo "Computing filter activation values"
for genome_reads_npy in $genomes_fragmented_dir/*npy; do
	
	base_name=$(basename "$genome_reads_npy" .npy)
	echo "Processing ""$base_name"
	genome_reads_fasta="$genomes_fragmented_dir"/"$base_name".fasta

	python3 genomes_filter_activations.py -t $genome_reads_npy -f $genome_reads_fasta -m $model -o "$filter_activation_dir"/"$base_name"/
	for filter_activation_file in "$filter_activation_dir"/"$base_name"/"$base_name"_filter_*bed; do

		#sort bed file by sequence, filter start position and convolution score and remove duplicates (due to overlapping reads) 
		#or take max of two scores at the same genomic position (can occurs if filter motif is recognized at the border of one read)
		sort -t$'\t' -k1,1 -k2,2n -k5,5rn "$filter_activation_file" | sort -t$'\t' -u -k1,1 -k2,2n > "$filter_activation_dir"/"$base_name"/$(basename "$filter_activation_file" .bed)_sorted_filtered.bed
		rm "$filter_activation_file"
	done
	gzip "$filter_activation_dir"/"$base_name"/*

done

echo "Performing genomic enrichment analysis for filter motifs ..."
for genome in "$filter_activation_dir"/*/; do

	bioproject_accession_id=$(basename $genome | sed 's/_fragmented_genomes$//')
	gff_file=$gff_dir/$bioproject_accession_id.gff

	#annotation data for genome available
	if [ -f $gff_file ];then
		echo "Processing ""$bioproject_accession_id""..."
		python3 enrichment_analysis_new2.py -i $genome -g "$gff_file" -o "$filter_genomic_enrichment_dir"/"$bioproject_accession_id"/
	fi
done

echo "Performing GO-enrichment analysis of genomic enriched filter motifs ..."
./GO_genomic_enriched_filters.sh $GO_dir $go_basic $filter_genomic_enrichment_dir $(($num_filters-1)) $filter_GO_enrichment_dir






