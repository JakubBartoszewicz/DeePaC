import os
import multiprocessing
import pybedtools
import pandas as pd
from scipy.stats import fisher_exact
import re
import argparse
from statsmodels.sandbox.stats.multicomp import multipletests

'''
Perform genomic enrichment analysis per filter.
'''


#parse command line options
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--bed_dir", required=True, help="Input directory with filter activation values for a species (.bed)")
parser.add_argument("-g", "--gff", required=True, help="Gff file of species")
parser.add_argument("-o", "--out_dir", default=".", help="Output directory")
args = parser.parse_args()


#create output directory
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)


def featuretype_filter(feature, featuretype):
	
	#CDS
	if feature[2] == featuretype:
		return True
	
	elif feature[2] == 'gene':
		if feature.attrs.get('gene', None) == featuretype:
			return True
	
	elif feature[2] in ["rRNA", "tRNA", "tmRNA", "ncRNA"]:
		if feature.attrs.get('product', None) == featuretype:
			return True
	
	return False


def subset_featuretypes(featuretype):
    result = gff.filter(featuretype_filter, featuretype).saveas()
    return pybedtools.BedTool(result.fn)


def count_reads_in_features(features_fn):
    """
    Callback function to count reads in features
    """
    return pybedtools.BedTool(bed).intersect( b=features_fn, stream=True, f = 1/3).count() ##overlap of at least 5bp (motif_length/3)

def count_num_feature_occurences(features_fn):
	return features_fn.count()
	
def count_len_feature_region(features_fn):
	return features_fn.total_coverage()
	
	

print("Processing gff file ...")
gff = pybedtools.BedTool(args.gff)
bioproject_id = os.path.splitext(os.path.basename(args.gff))[0]	
#extract all feature types (genes, RNAs, CDS) from gff file
all_feature_types = []
for feature in gff:
	if feature[2] == 'CDS':
		all_feature_types.append('CDS')
	if feature[2] == 'gene':
		if 'gene' in feature.attrs:
			all_feature_types.append(feature.attrs['gene'])
	elif feature[2] in ["rRNA", "tRNA", "tmRNA", "ncRNA"]:
		if 'product' in feature.attrs:
			all_feature_types.append(feature.attrs['product'])

all_feature_types = sorted(list(set(all_feature_types)))

motif_length = 15
genome_size = gff.total_coverage()
num_possible_hits = genome_size - motif_length + 1
min_overlap = 5

#one bed file per filter motif
print("Processing bed files ...")
for bed_file in os.listdir(args.bed_dir):
	if bed_file.endswith(".bed.gz") and os.stat(args.bed_dir + "/" + bed_file).st_size > 0:

		filter = re.search("filter_[0-9]+", bed_file).group()		
		print("Processing " + bed_file + " ...")
		bed = pybedtools.BedTool(args.bed_dir + "/" + bed_file)
		num_entries = bed.count()

		#filter gff files for feature of interest
		pool = multiprocessing.Pool(processes=20)
		filtered_gffs = pool.map(subset_featuretypes, all_feature_types)

		num_entries = bed.count()
		num_hits_feature = pool.map(count_reads_in_features, filtered_gffs)
		num_feature_occurences = pool.map(count_num_feature_occurences, filtered_gffs)
		len_feature_region = pool.map(count_len_feature_region, filtered_gffs)
		num_possible_hits_feature = [len_feature_region[i] + num_feature_occurences[i] + motif_length*num_feature_occurences[i] - 2*min_overlap*num_feature_occurences[i] for i in range(len(all_feature_types))]
		num_possible_hits_outside_feature = [num_possible_hits - num_possible_hits_feature[i] for i in range(len(all_feature_types))]
		num_hits_outside_feature =  [num_entries - num_hits_feature[i] for i in range(len(all_feature_types))]
		
		motif_results = pd.DataFrame(columns=["motif_id", "bioproject_id", "feature","num_hits_feature", "num_hits_outside_feature", "num_possible_hits_feature", "num_possible_hits_outside_feature", "fisher_logoddsratio", "fisher_p_value_2sided", "fisher_p_value_feature", "fisher_p_value_outside_feature"])
	
		for idx in range(len(all_feature_types)):
		
			feature_type = all_feature_types[idx]
			#perform fisher exact test to assess whether motif occurs significantly more often in or outside of feature
			contingency_table = [[num_hits_feature[idx], num_possible_hits_feature[idx] - num_hits_feature[idx]], [num_hits_outside_feature[idx], num_possible_hits_outside_feature[idx] - num_hits_outside_feature[idx]]]
			oddsratio, p_value = fisher_exact(contingency_table, alternative = "two-sided") #H1: motif occurence is significantly biased towards coding or noncoding region
			oddsratio, p_value_c = fisher_exact(contingency_table, alternative = "greater") #H1: motif occurs significantly more often in coding regions
			oddsratio, p_value_nc = fisher_exact(contingency_table, alternative = "less") #H1: motif occurs significantly more often in noncoding regions

			motif_results.loc[idx] = [filter, bioproject_id, feature_type, num_hits_feature[idx], num_hits_outside_feature[idx], num_possible_hits_feature[idx], num_possible_hits_outside_feature[idx], oddsratio, p_value, p_value_c, p_value_nc]



		#save enrichment results per motif
		out_file = args.out_dir +  "/" + bioproject_id + "_" + filter + ".csv" 
		motif_results.to_csv(out_file, sep = "\t", index = False)
		pool.close()
		
		#multiple testing correction
		fisher_q_value_2sided = multipletests(motif_results.fisher_p_value_2sided, alpha = 0.05, method = "fdr_bh")[1]
		fisher_q_value_feature = multipletests(motif_results.fisher_p_value_feature, alpha = 0.05, method = "fdr_bh")[1]
		fisher_q_value_outside_feature = multipletests(motif_results.fisher_p_value_outside_feature, alpha = 0.05, method = "fdr_bh")[1]

		#add new columns to data frame
		motif_results['fisher_q_value_2sided'] = fisher_q_value_2sided
		motif_results['fisher_q_value_feature'] = fisher_q_value_feature
		motif_results['fisher_q_value_outside_feature'] = fisher_q_value_outside_feature

		out_file = args.out_dir +  "/" + bioproject_id + "_" + filter +  "_ext.csv"
		#save results
		motif_results.to_csv(out_file , sep = "\t", index = False)
		
		#filtering out entries with FDR >= 0.05
		out_file = args.out_dir +  "/" + bioproject_id + "_" + filter + "_sorted_filtered.csv"
		motif_results = motif_results[motif_results.fisher_q_value_feature < 0.05]
		motif_results = motif_results.sort_values(by=['fisher_p_value_2sided', 'fisher_p_value_feature'])
		if len(motif_results.index):
			motif_results.to_csv(out_file , sep = "\t", index = False)
