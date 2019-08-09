#!/bin/bash

#directory with gaf file per species
gaf_dir=$1
#go-basic.obo
go_basic=$2
#directory with gennomic filter enrichment results
filter_genomic_enrichment_dir=$3
#number convolutional filters
num_filters=$4
#output directory
out_dir=$5


#create output directory
if [ ! -d "$out_dir" ]; then
	mkdir "$out_dir"
fi

#for each species do
for genome_gaf in $gaf_dir/*gaf; do

	bioproject_accession_id=$(basename "$genome_gaf" .gaf)
	if [ ! -d "$out_dir"/"$bioproject_accession_id" ]; then
		mkdir  "$out_dir"/"$bioproject_accession_id"
	fi
	echo $bioproject_accession_id

	#intersect annotated genes with all genes from gff file per genome
	echo "Building population set"
	#changed
	join -1 1 -2 1 <(cut -d$'\t' -f3 $filter_genomic_enrichment_dir/"$bioproject_accession_id"/"$bioproject_accession_id"_filter_201.csv | grep -v "feature" | sort) <(cut -d$'\t' -f3 "$genome_gaf" | uniq | sort | uniq ) > "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_population.txt
	#population set is empty -> no enrichment analysis possible
	if [ ! -s "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_population.txt ];then
		echo "No enrichment analysis of ""$bioproject_accession_id"" (no population set)..."
		continue
	fi
	
	#combined enrichment results over all filter
	> "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_all_filter_enrichment_analysis_wo_depletions.tsv

	#extract gene name and associated GO-term per species
	echo "Building GO-gene association file"
	join -t$'\t' -1 1 -2 1 <(cut -d$'\t' -f3,5 "$genome_gaf" | sort -t$'\t' -k1,1) "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_population.txt > "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_association.txt
	
	#for each filter do
	for ((filter=0; filter<=num_filters; filter++)); do
		join -1 1 -2 1 <(cut -d$'\t' -f3 $filter_genomic_enrichment_dir/"$bioproject_accession_id"/"$bioproject_accession_id"_filter_"$filter"_sorted_filtered.csv | grep -v "feature" | sort) "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_population.txt > "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_filter_"$filter"_study.txt

		#gene enrichment analysis from goatools package
		if [ ! -s "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_filter_"$filter"_study.txt ]; then
			echo "No enrichment analysis of ""$filter"" (no study set)..."
			continue
		fi
		python3 ~/my_venv/bin/find_enrichment.py  "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_filter_"$filter"_study.txt "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_population.txt "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_association.txt --obo "$go_basic" --alpha 0.05 --method fdr_bh --pvalcalc=fisher_scipy_stats --outfile "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_filter_"$filter"_enrichment_analysis.tsv
		
		#remove depletions, since we are just interested in enrichments
		if [ -f "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_filter_"$filter"_enrichment_analysis.tsv ]; then
			awk -F'\t' '{if ($3 == "e" || $3 == "enrichment") print $0}' "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_filter_"$filter"_enrichment_analysis.tsv > "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_filter_"$filter"_enrichment_analysis_wo_depletions.tsv
			num_hits=$(grep -v "^#" "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_filter_"$filter"_enrichment_analysis_wo_depletions.tsv | wc -l)
			if [ "$num_hits" == 0 ]; then
				rm "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_filter_"$filter"_enrichment_analysis_wo_depletions.tsv
			else
				#add filter name as column
				sed -i "s/^/filter\_$filter\t/" "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_filter_"$filter"_enrichment_analysis_wo_depletions.tsv
				#correct header line
				sed -i "s/^filter\_$filter\t#/#filter\t/" "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_filter_"$filter"_enrichment_analysis_wo_depletions.tsv
				grep -v "^#" "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_filter_"$filter"_enrichment_analysis_wo_depletions.tsv >> "$out_dir"/"$bioproject_accession_id"/"$bioproject_accession_id"_all_filter_enrichment_analysis_wo_depletions.tsv
			fi
		fi
		
		done
done

