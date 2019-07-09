import os
import multiprocessing
import pybedtools
import pandas as pd
import argparse
from collections import OrderedDict

'''
Compute mean pathogenicity score per gene and species.
'''

#parse command line options
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--patho_dir", required=True, help="Directory containing the pathogenicity scores over all genomic regions per species (.bedgraph)")
parser.add_argument("-g", "--gff_dir", required=True, help="Directory containing the annotation data of the species (.gff)")
parser.add_argument("-o", "--out_dir", default=".", help="Output directory")
args = parser.parse_args()


#create output directory
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

def featuretype_filter(feature, featuretype):

    if feature[2] == 'gene':
        if feature.attrs.get('gene', None) == featuretype:
            return True

    elif feature[2] in ["rRNA", "tRNA", "tmRNA", "ncRNA"]:
        if feature.attrs.get('product', None) == featuretype:
            return True

    return False


def subset_featuretypes(featuretype):
    result = gff.filter(featuretype_filter, featuretype).saveas()
    return pybedtools.BedTool(result.fn)


def compute_gene_pathogenicity(filtered_gff):

    #intersection = pybedtools.BedTool(bedgraph).intersect( b=filtered_gff)
    intersection = bedgraph.intersect( b=filtered_gff)
    total_num_bases = 0.
    patho_score = 0.
    for entry in intersection:
        num_bases = entry.length
        patho_score += float(entry.fields[3]) * num_bases
        total_num_bases += num_bases
    patho_score /= float(total_num_bases)
    return patho_score


#for each species do
for gff_file in os.listdir(args.gff_dir):
    if gff_file.endswith(".gff"):
        print("Processing " + gff_file + " ...")
        gff = pybedtools.BedTool(args.gff_dir + "/" + gff_file)
        bioproject_id = os.path.splitext(os.path.basename(gff_file))[0]

        #extract all feature types (genes, RNAs) from gff file
        all_feature_types = []
        for feature in gff:
            if feature[2] == 'gene':
                if 'gene' in feature.attrs:
                    all_feature_types.append(feature.attrs['gene'])
            elif feature[2] in ["rRNA", "tRNA", "tmRNA", "ncRNA"]:
                if 'product' in feature.attrs:
                    all_feature_types.append(feature.attrs['product'])

        all_feature_types = sorted(list(set(all_feature_types)))

        patho_file = args.patho_dir + bioproject_id + "_fragmented_genomes_pathogenicity.bedgraph"
        print("Processing " + patho_file + " ...")
        bedgraph = pybedtools.BedTool(patho_file)

        pool = multiprocessing.Pool(processes=20)
        #filter gff files for feature of interest
        filtered_gffs = pool.map(subset_featuretypes, all_feature_types)
        #compute mean pathogencity score per feature
        feature_pathogenicities = [compute_gene_pathogenicity(filtered_gff) for filtered_gff in filtered_gffs]

        #save results
        patho_table = pd.DataFrame(OrderedDict( (('feature', all_feature_types), ('bioproject_id', bioproject_id), ('pathogenicity_score', feature_pathogenicities)) ))
        patho_table = patho_table.sort_values(by=['pathogenicity_score'], ascending = False)
        patho_table.to_csv(args.out_dir +  "/" + bioproject_id + "_feature_pathogenicity.csv" , sep = "\t", index = False)
        pool.close()
