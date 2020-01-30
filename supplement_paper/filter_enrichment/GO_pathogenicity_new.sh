#!/bin/bash

#Performing GO-enrichment analysis for highly pathogenic or nonpathogenic genes

#directory with gaf file per species
gaf_dir=$1
#go-basic.obo
go_basic=$2
#directory with genome-wide pathogenicity scores (.bedgraph) and gene ranking by pathogenicity scores
pathogenicity_dir=$3
#output directory
out_dir=$4
mode=$5 #upper, lower
cutoff=$6 #"mean" or "fixed"
pathogenicity_cutoff=$7 #only used if cutoff is "fixed", S. pneumoniae: upper quartile: 0.65, lower quartile: 0.54


#create output directory
if [ ! -d "$out_dir" ]; then
	mkdir "$out_dir"
fi


for genome_gaf in $gaf_dir/*gaf; do

	bioproject_accession_id=$(basename "$genome_gaf" .gaf)
	if [ ! -d "$out_dir"/"$bioproject_accession_id" ]; then
		mkdir  "$out_dir"/"$bioproject_accession_id"
	fi
	echo $bioproject_accession_id

	#build population set by intersecting annotated genes with all genes from gff file per genome
	join -1 1 -2 1 <(tail -n +2 $pathogenicity_dir/gene_ranking/"$bioproject_accession_id"_feature_pathogenicity.csv | cut -d$'\t' -f1  | sort) <(cut -d$'\t' -f3 "$genome_gaf" | uniq | sort | uniq ) > "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_population.txt
	 
	#population set is empty -> no enrichment analysis possible
	if [ ! -s "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_population.txt ];then
		echo "No enrichment analysis of ""$bioproject_accession_id"" (no population set)..."
		continue
	fi

	#extract gene name and associated GO-term per species
	join -t$'\t' -1 1 -2 1 <(cut -d$'\t' -f3,5 "$genome_gaf" | sort -t$'\t' -k1,1) "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_population.txt > "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_association.txt

    #compute mean pathogenicity score of that species from .bedgraph
	if [ "$cutoff" == "mean" ]; then
        	pathogenicity_cutoff=$(awk -F'\t' -v total_num_bases=0 -v total_score=0 -v num_rows=$(wc -l "$pathogenicity_dir"/"$bioproject_accession_id"_fragmented_genomes_pathogenicity.bedgraph | cut -d' ' -f1) '{len=$3-$2; score = len*$4; total_num_bases = total_num_bases + len; total_score=total_score + score; if (NR == num_rows) {print total_score/total_num_bases}}' "$pathogenicity_dir"/"$bioproject_accession_id"_fragmented_genomes_pathogenicity.bedgraph)
		echo "Using ""$pathogenicity_cutoff"" as mean pathogenicity cutoff ..."
		study_set_file="$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_study_mean_"$mode".txt
		enrichment_file="$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_enrichment_analysis_mean_"$mode".tsv
	else
		study_set_file="$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_study_"$pathogenicity_cutoff"_"$mode".txt
		enrichment_file="$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_enrichment_analysis_"$pathogenicity_cutoff"_"$mode".tsv
	fi

	#build study set
	if [ $mode == "upper" ]; then
		join -t$'\t' -1 1 -2 1 <(tail -n+2 $pathogenicity_dir/gene_ranking/"$bioproject_accession_id"_feature_pathogenicity.csv | awk -F'\t' -v pathogenicity_cutoff="$pathogenicity_cutoff" '{if ($3 > pathogenicity_cutoff) print $1}' | sort) "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_population.txt > "$study_set_file"
	else
		join -t$'\t' -1 1 -2 1 <(tail -n+2 $pathogenicity_dir/gene_ranking/"$bioproject_accession_id"_feature_pathogenicity.csv | awk -F'\t' -v pathogenicity_cutoff="$pathogenicity_cutoff" '{if ($3 < pathogenicity_cutoff) print $1}' | sort) "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_population.txt > "$study_set_file"
	fi

    #study set is empty -> no enrichment analysis possible
    if [ ! -s "$study_set_file" ];then
            echo "No enrichment analysis of ""$bioproject_accession_id"" (no study set)..."
            continue
    fi

	#perform gene enrichment analysis with goatools package
	python3 ~/my_venv/bin/find_enrichment.py  "$study_set_file" "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_population.txt "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_association.txt --obo "$go_basic" --alpha 0.05 --method fdr_bh --pvalcalc=fisher_scipy_stats --outfile "$enrichment_file"

done

