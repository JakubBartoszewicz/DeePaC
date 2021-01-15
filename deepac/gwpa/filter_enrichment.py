import os
import multiprocessing
import pybedtools
import pandas as pd
from scipy.stats import fisher_exact
import re
from statsmodels.sandbox.stats.multicomp import multipletests
from functools import partial


def featuretype_filter(feature, featuretype):
    """Check if feature is a feature of interest (genes, CDSs and RNAs)."""
    # CDS
    if feature[2] == featuretype:
        return True

    if feature[2] == 'CDS':
        if feature.attrs.get('product', None) == featuretype:
            return True

    elif feature[2] == 'gene':
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


def count_reads_in_features(features_fn, bed):
    """Callback function to count reads in features"""
    # overlap of at least 5bp (motif_length/3)
    return pybedtools.BedTool(bed).intersect(b=features_fn, stream=True,
                                             f=1 / 3).count()


def count_num_feature_occurences(features_fn):
    return features_fn.count()


def count_len_feature_region(features_fn):
    return features_fn.total_coverage()


def filter_enrichment(args):
    """Perform genomic enrichment analysis per filter."""
    # create output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.n_cpus is None:
        cores = multiprocessing.cpu_count()
    else:
        cores = args.n_cpus

    print("Processing gff file ...")
    gff = pybedtools.BedTool(args.gff)
    bioproject_id = os.path.splitext(os.path.basename(args.gff))[0]
    # extract all feature types (genes, RNAs, CDS) from gff file
    all_feature_types = []
    for feature in gff:
        if feature[2] == 'CDS':
            all_feature_types.append('CDS')
            if args.extended and 'product' in feature.attrs:
                all_feature_types.append(feature.attrs['product'])
        if feature[2] == 'gene':
            if 'gene' in feature.attrs:
                all_feature_types.append(feature.attrs['gene'])
            elif args.extended:
                if 'Name' in feature.attrs:
                    all_feature_types.append(feature.attrs['Name'])
                elif 'ID' in feature.attrs:
                    all_feature_types.append(feature.attrs['ID'])
        elif feature[2] in ["rRNA", "tRNA", "tmRNA", "ncRNA"]:
            if 'product' in feature.attrs:
                all_feature_types.append(feature.attrs['product'])

    all_feature_types = sorted(list(set(all_feature_types)))
    motif_length = args.motif_length
    genome_size = gff.total_coverage()
    num_possible_hits = 2 * (genome_size - motif_length + 1)
    min_overlap = motif_length//3

    # one bed file per filter motif
    print("Processing bed files ...")
    for bed_file in os.listdir(args.bed_dir):
        if (bed_file.endswith(".bed.gz") or bed_file.endswith(".bed")) and os.stat(
                args.bed_dir + "/" + bed_file).st_size > 0:

            c_filter = re.search("filter_[0-9]+", bed_file).group()
            print("Processing " + bed_file + " ...")
            bed = pybedtools.BedTool(args.bed_dir + "/" + bed_file)

            # filter gff files for feature of interest
            with multiprocessing.Pool(processes=cores) as pool:
                filtered_gffs = pool.map(partial(subset_featuretypes, gff=gff), all_feature_types)

            filtered_gffs = [b.merge() for b in filtered_gffs]
            num_entries = bed.count()
            with multiprocessing.Pool(processes=cores) as pool:
                num_hits_feature = pool.map(partial(count_reads_in_features, bed=bed), filtered_gffs)
            with multiprocessing.Pool(processes=cores) as pool:
                num_feature_occurences = pool.map(count_num_feature_occurences, filtered_gffs)
            with multiprocessing.Pool(processes=cores) as pool:
                len_feature_region = pool.map(count_len_feature_region, filtered_gffs)
            num_possible_hits_feature = [
                2 * (len_feature_region[i] + num_feature_occurences[i] + motif_length * num_feature_occurences[
                    i] - 2 * min_overlap * num_feature_occurences[i]) for i in range(len(all_feature_types))]
            num_possible_hits_outside_feature = [num_possible_hits - num_possible_hits_feature[i] for i in
                                                 range(len(all_feature_types))]
            num_hits_outside_feature = [num_entries - num_hits_feature[i] for i in range(len(all_feature_types))]

            motif_results = pd.DataFrame(
                columns=["motif_id", "bioproject_id", "feature", "num_hits_feature", "num_hits_outside_feature",
                         "num_possible_hits_feature", "num_possible_hits_outside_feature", "fisher_logoddsratio",
                         "fisher_p_value_2sided", "fisher_p_value_feature", "fisher_p_value_outside_feature"])

            for idx in range(len(all_feature_types)):
                feature_type = all_feature_types[idx]
                # perform fisher exact test to assess whether
                # motif occurs significantly more often in or outside of feature
                contingency_table = [[num_hits_feature[idx], num_possible_hits_feature[idx] - num_hits_feature[idx]],
                                     [num_hits_outside_feature[idx],
                                      num_possible_hits_outside_feature[idx] - num_hits_outside_feature[idx]]]

                oddsratio, p_value = fisher_exact(contingency_table,
                                                  alternative="two-sided")
                # H1: motif occurence is significantly biased towards coding or noncoding region
                oddsratio, p_value_c = fisher_exact(contingency_table,
                                                    alternative="greater")
                # H1: motif occurs significantly more often in coding regions
                oddsratio, p_value_nc = fisher_exact(contingency_table,
                                                     alternative="less")
                # H1: motif occurs significantly more often in noncoding regions

                motif_results.loc[idx] = [c_filter, bioproject_id, feature_type, num_hits_feature[idx],
                                          num_hits_outside_feature[idx], num_possible_hits_feature[idx],
                                          num_possible_hits_outside_feature[idx],
                                          oddsratio, p_value, p_value_c, p_value_nc]

            # save enrichment results per motif
            out_file = args.out_dir + "/" + bioproject_id + "_" + c_filter + ".csv"
            motif_results.to_csv(out_file, sep="\t", index=False)

            # multiple testing correction
            fisher_q_value_2sided = multipletests(motif_results.fisher_p_value_2sided, alpha=0.05, method="fdr_bh")[1]
            fisher_q_value_feature = multipletests(motif_results.fisher_p_value_feature, alpha=0.05, method="fdr_bh")[1]
            fisher_q_value_outside_feature = \
                multipletests(motif_results.fisher_p_value_outside_feature, alpha=0.05, method="fdr_bh")[1]

            # add new columns to data frame
            motif_results['fisher_q_value_2sided'] = fisher_q_value_2sided
            motif_results['fisher_q_value_feature'] = fisher_q_value_feature
            motif_results['fisher_q_value_outside_feature'] = fisher_q_value_outside_feature

            out_file = args.out_dir + "/" + bioproject_id + "_" + c_filter + "_ext.csv"
            # save results
            motif_results.to_csv(out_file, sep="\t", index=False)

            if args.extended:
                out_file = args.out_dir + "/" + bioproject_id + "_" + c_filter + "_sorted_filtered_extended.csv"
            else:
                out_file = args.out_dir + "/" + bioproject_id + "_" + c_filter + "_sorted_filtered.csv"
            # filtering out entries with FDR >= 0.05
            motif_results = motif_results[motif_results.fisher_q_value_feature < 0.05]
            motif_results = motif_results.sort_values(by=['fisher_p_value_2sided', 'fisher_p_value_feature'])
            if len(motif_results.index):
                motif_results.to_csv(out_file, sep="\t", index=False)
