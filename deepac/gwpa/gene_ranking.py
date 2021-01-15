import os
import multiprocessing
import pybedtools
import pandas as pd
from functools import partial
from collections import OrderedDict


def featuretype_filter(feature, featuretype):
    """Check if feature is a feature of interest (genes, CDSs and RNAs)."""
    # CDS
    if feature[2] == featuretype:
        return True

    if feature[2] == 'CDS':
        if feature.attrs.get('product', None) == featuretype:
            return True

    if feature[2] == 'gene':
        if feature.attrs.get('gene', None) == featuretype:
            return True
        elif feature.attrs.get('Name', None) == featuretype:
            return True
        elif feature.attrs.get('ID', None) == featuretype:
            return True

    elif feature[2] in ["rRNA", "tRNA", "tmRNA", "ncRNA"]:
        if feature.attrs.get('product', None) == featuretype:
            return True

    return False


def subset_featuretypes(featuretype, gff):
    """Select features of interest."""
    result = gff.filter(featuretype_filter, featuretype).saveas()
    return pybedtools.BedTool(result.fn)


def compute_gene_pathogenicity(filtered_gff, bedgraph):
    """Compute mean pathogenicity score of a gene."""
    # intersection = pybedtools.BedTool(bedgraph).intersect( b=filtered_gff)
    intersection = bedgraph.intersect(b=filtered_gff)
    total_num_bases = 0.
    patho_score = 0.
    for entry in intersection:
        num_bases = entry.length
        patho_score += float(entry.fields[3]) * num_bases
        total_num_bases += num_bases
    patho_score /= float(total_num_bases)
    return patho_score


def gene_rank(args):
    """Compute mean pathogenicity score per gene and species."""
    # create output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.n_cpus is None:
        cores = multiprocessing.cpu_count()
    else:
        cores = args.n_cpus

    # for each species do
    for gff_file in os.listdir(args.gff_dir):
        if gff_file.endswith(".gff") or gff_file.endswith(".gff3"):
            print("Processing " + gff_file + " ...")
            gff = pybedtools.BedTool(args.gff_dir + "/" + gff_file)
            bioproject_id = os.path.splitext(os.path.basename(gff_file))[0]

            # extract all feature types (genes, RNAs) from gff file
            all_feature_types = []
            for feature in gff:
                if feature[2] == 'gene':
                    if 'gene' in feature.attrs:
                        all_feature_types.append(feature.attrs['gene'])
                    elif args.extended:
                        if 'Name' in feature.attrs:
                            all_feature_types.append(feature.attrs['Name'])
                        elif 'ID' in feature.attrs:
                            all_feature_types.append(feature.attrs['ID'])
                if feature[2] == 'CDS':
                    all_feature_types.append('CDS')
                    if args.extended and 'product' in feature.attrs:
                        all_feature_types.append(feature.attrs['product'])
                elif feature[2] in ["rRNA", "tRNA", "tmRNA", "ncRNA"]:
                    if 'product' in feature.attrs:
                        all_feature_types.append(feature.attrs['product'])

            all_feature_types = sorted(list(set(all_feature_types)))

            patho_file = os.path.join(args.patho_dir, bioproject_id + "_fragmented_genomes_pathogenicity.bedgraph")
            print("Processing " + patho_file + " ...")
            bedgraph = pybedtools.BedTool(patho_file)

            with multiprocessing.Pool(processes=cores) as pool:
                # filter gff files for feature of interest
                filtered_gffs = pool.map(partial(subset_featuretypes, gff=gff), all_feature_types)
                # compute mean pathogencity score per feature
                feature_pathogenicities = [compute_gene_pathogenicity(filtered_gff, bedgraph)
                                           for filtered_gff in filtered_gffs]

                # save results
                patho_table = pd.DataFrame(OrderedDict((('feature', all_feature_types),
                                                        ('bioproject_id', bioproject_id),
                                                        ('pathogenicity_score', feature_pathogenicities))))
                patho_table = patho_table.sort_values(by=['pathogenicity_score'], ascending=False)

                if args.extended:
                    patho_table.to_csv(args.out_dir + "/" + bioproject_id + "_feature_pathogenicity_extended.csv",
                                       sep="\t",
                                       index=False)
                else:
                    patho_table.to_csv(args.out_dir + "/" + bioproject_id + "_feature_pathogenicity.csv",
                                       sep="\t",
                                       index=False)
