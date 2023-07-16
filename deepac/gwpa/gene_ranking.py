import os
import multiprocessing
import pybedtools
import pandas as pd
from functools import partial
from collections import OrderedDict
from scipy.stats import ttest_ind
from statsmodels.sandbox.stats.multicomp import multipletests
import numpy as np
from shutil import rmtree


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
    intersection_df = intersection.to_dataframe()
    in_scores = intersection_df.iloc[:, 3]
    in_lengths = intersection_df['end'] - intersection_df['start']
    patho_score = sum((in_lengths * in_scores))/sum(in_lengths)
    return patho_score


def compute_gene_ttest(filtered_gff, bedgraph, filter_annot=False, min_length=1, mean_score=None):
    """Test for elevated pathogenicity score within a gene."""
    merged_gff = filtered_gff.merge()
    intersection = bedgraph.intersect(b=merged_gff)
    subtraction = bedgraph.subtract(b=merged_gff)
    if min_length > 1:
        intersection = intersection.filter(lambda x: len(x) >= min_length).saveas()
        subtraction = subtraction.filter(lambda x: len(x) >= min_length).saveas()
    index = 3 if not filter_annot else 4
    if len(intersection) > 0:
        intersection_df = intersection.to_dataframe()
        in_scores = intersection_df.iloc[:, index]
        if not filter_annot:
            in_lengths = intersection_df['end'] - intersection_df['start']
            in_list = np.repeat(in_scores, in_lengths)
        else:
            in_list = in_scores
    else:
        in_list = []

    if len(subtraction) > 0:
        subtraction_df = subtraction.to_dataframe()
        out_scores = subtraction_df.iloc[:, index]
        if not filter_annot:
            out_lengths = subtraction_df['end'] - subtraction_df['start']
            out_list = np.repeat(out_scores, out_lengths)
        else:
            out_list = out_scores
    else:
        out_list = []

    if mean_score is None:
        all_df = bedgraph.to_dataframe()
        all_scores = all_df.iloc[:, index]
        if not filter_annot:
            all_lengths = all_df['end'] - all_df['start']
            total = np.mean(np.repeat(all_scores, all_lengths))
        else:
            total = np.mean(all_scores)
    else:
        total = mean_score

    if len(in_list) > 0 and len(out_list) > 0:
        mean_in = np.mean(in_list)
        mean_out = np.mean(out_list)
        difference = mean_in - mean_out
        # total = np.mean(np.concatenate((in_list, out_list)))
        contribution = total - mean_out
        return difference, ttest_ind(in_list, out_list)[1], mean_out, total, contribution
    else:
        return np.nan, np.nan, np.nan, np.nan, np.nan


def gene_rank(args):
    """Compute mean pathogenicity score per gene and species."""
    # create output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    temp_path = os.path.join(args.out_dir, "tmp")
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    pybedtools.helpers.set_tempdir(temp_path)

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
            all_df = bedgraph.to_dataframe()
            all_scores = all_df.iloc[:, 3]
            all_lengths = all_df['end'] - all_df['start']
            mean_score = np.mean(np.repeat(all_scores, all_lengths))

            with multiprocessing.Pool(processes=cores) as pool:
                # t-test inside vs outside feature
                ttest_results = pool.map(partial(compute_gene_ttest, bedgraph=bedgraph, mean_score=mean_score),
                                         filtered_gffs)
            ttest_diffs, ttest_pvals, ttest_mean_out, total, contribs = zip(*ttest_results)
            ttest_qvals = multipletests(ttest_pvals, alpha=0.05, method="fdr_bh")[1]

            # save results
            patho_table = pd.DataFrame(OrderedDict((('feature', all_feature_types),
                                                    ('bioproject_id', bioproject_id),
                                                    ('pathogenicity_score', feature_pathogenicities),
                                                    ('raw_p_value', ttest_pvals),
                                                    ('q_value', ttest_qvals),
                                                    ('difference', ttest_diffs),
                                                    ('out_score', ttest_mean_out),
                                                    ('genome_score', total),
                                                    ('feature_contrib', contribs)
                                                    )))
            patho_table = patho_table.sort_values(by=['feature_contrib'], ascending=False)

            if args.extended:
                patho_table.to_csv(args.out_dir + "/" + bioproject_id + "_feature_pathogenicity_extended.csv",
                                   sep="\t",
                                   index=False)
            else:
                patho_table.to_csv(args.out_dir + "/" + bioproject_id + "_feature_pathogenicity.csv",
                                   sep="\t",
                                   index=False)
            pybedtools.cleanup()
    rmtree(temp_path)
