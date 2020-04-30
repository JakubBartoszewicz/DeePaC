import os
import csv
import numpy as np
from Bio.motifs import transfac
from scipy.stats import pearsonr, spearmanr


def motif_compare(args):
    """Compare PSSMs of filter motifs."""
    # create output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # load training data to determine background nucleotide content
    train_samples = np.load(args.train_data, mmap_mode='r')
    probs = np.mean(np.mean(train_samples, axis=1), axis=0)
    bg = {'A': probs[0], 'C': probs[1], 'G': probs[2], 'T': probs[3]}

    # load all filter motifs from first file
    with open(args.in_file1) as handle:
        records1 = transfac.read(handle)

    # load all filter motifs from second file
    with open(args.in_file2) as handle:
        records2 = transfac.read(handle)

    # convert motifs to pssm's
    pssms1 = {}
    pssms2 = {}
    rc_pssms2 = {}

    for idx, m1 in enumerate(records1):
        pwm1 = m1.counts.normalize(pseudocounts=bg)
        pssm1 = pwm1.log_odds(background=bg)
        pssms1[m1.get("ID")] = pssm1

    for idx, m2 in enumerate(records2):
        pwm2 = m2.counts.normalize(pseudocounts=bg)
        pssm2 = pwm2.log_odds(background=bg)
        pssms2[m2.get("ID")] = pssm2
        # build reverse complement
        if args.rc:
            rc_pssm2 = pssm2.reverse_complement()
            rc_pssms2[idx] = rc_pssm2

    result_table = []
    # compare motifs
    for idx1, pssm1 in pssms1.items():

        for idx2, pssm2 in pssms2.items():

            if args.extensively or idx1 == idx2:

                row = [idx1, idx2]

                for measure in [pearsonr, spearmanr]:

                    cor, p_value, offset = get_motif_similarity(measure, pssm1, pssm2,
                                                                args.min_overlap if args.shift else pssm1.length)
                    orientation = "+"
                    if args.rc:
                        rc_pssm2 = rc_pssms2[idx2]
                        cor_rc, p_value_rc, offset_rc = get_motif_similarity(measure, pssm1, rc_pssm2,
                                                                             args.min_overlap if args.shift else pssm1.length)
                        # if cor < cor_rc:
                        if p_value > p_value_rc:
                            cor, p_value, offset, orientation = cor_rc, p_value_rc, offset_rc, "-"
                    row.extend([cor, p_value, offset, orientation])

                result_table.append(row)

    # write results to output file
    out_file_name = args.out_dir + "/correlation_motifs" + ("_extensively" if args.extensively else "") + (
        "_rc" if args.rc else "") + ("_shift_min_overlap=" + str(args.min_overlap) if args.shift else "") + ".txt"

    with open(out_file_name, 'w') as csv_file:
        file_writer = csv.writer(csv_file, delimiter="\t")
        file_writer.writerow(["ID1", "ID2", "cor_pearson", "p_value_pearson", "offset_pearson", "orientation_pearson",
                              "cor_spearman", "p_value_spearman", "offset_spearman", "orientation_spearman"])
        for row in result_table:
            file_writer.writerow(row)


def get_motif_similarity(measure, pssm1, pssm2, min_overlap):
    """Compute similarity between two position specific scoring matrices (pssm's)."""

    assert pssm1.length == pssm2.length, "Motifs with different length are currently not supported!"

    final_cor, final_p_value, final_offset = 0, 1, 0

    letters = pssm1.keys()
    for offset in range(-(pssm1.length - min_overlap), pssm1.length - min_overlap + 1):

        if offset < 0:
            x1 = np.array([pssm1[base][:offset] for base in letters]).flatten()
            x2 = np.array([pssm2[base][-offset:] for base in letters]).flatten()
        elif offset > 0:
            x1 = np.array([pssm1[base][offset:] for base in letters]).flatten()
            x2 = np.array([pssm2[base][:-offset] for base in letters]).flatten()
        # no offset
        else:
            x1 = np.array([pssm1[base] for base in letters]).flatten()
            x2 = np.array([pssm2[base] for base in letters]).flatten()

        cor, p_value = measure(x1, x2)
        # if cor > final_cor:
        if p_value < final_p_value:
            final_cor, final_p_value, final_offset = cor, p_value, offset

    return final_cor, final_p_value, final_offset
