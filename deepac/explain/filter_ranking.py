import os
import re
import numpy as np
import matplotlib.pyplot as plt
import csv


def get_filter_ranking(args):
    """Plot contribution scores per filter and rank filters
    according to their pathogenicity/non-pathogenicity potential."""
    # create output directory
    out_dir = args.out_dir + "/" + args.mode
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # load true labels
    y_truth = np.load(args.true_label, mmap_mode='r')
    # load predicted labels
    y_pred = np.load(args.pred_label, mmap_mode='r')
    y_pred = (y_pred > 0.5).astype('int32')

    # nonzero scores per filter
    filter_scores = dict()
    # nonzero reads per filter
    read_ids = dict()
    for file in os.listdir(args.scores_dir):
        if file.endswith(".csv") and os.stat(args.scores_dir + "/" + file).st_size > 0:
            with open(args.scores_dir + "/" + file, 'r') as csvfile:
                reader = csv.reader(csvfile)
                c_filter = re.search("filter_[0-9]+", file).group()
                read_ids[c_filter] = []
                filter_scores[c_filter] = []
                for ind, row in enumerate(reader):
                    if ind % 3 == 0:
                        read_id = re.search("seq_[0-9]+", row[0]).group()
                        read_id = int(read_id.replace("seq_", ""))
                    elif ind % 3 == 2:
                        try:
                            # if score non-empty
                            score = float(row[0])
                        except ValueError:
                            continue
                        read_ids[c_filter].append(read_id)
                        filter_scores[c_filter].append(score)
                read_ids[c_filter] = np.array(read_ids[c_filter])
                filter_scores[c_filter] = np.array(filter_scores[c_filter])
                if args.mode == "rel_pred_class":
                    filter_scores[c_filter][y_pred[read_ids[c_filter]] == 0] *= -1
                if args.mode == "rel_true_class":
                    filter_scores[c_filter][y_truth[read_ids[c_filter]] == 0] *= -1

                if len(filter_scores[c_filter]) == 0:
                    print("Skipping " + c_filter + ' ... (no data)')
                    continue

                # plot distribution of contribution scores per filter (excluding zeros) as histograms
                print("Plotting the distribution of the contribution scores of " + c_filter + ' ...')
                data = []
                boxplot_labels = []
                tn = filter_scores[c_filter][np.array(y_truth[read_ids[c_filter]] == 0) &
                                             np.array(y_pred[read_ids[c_filter]] == 0)]
                tp = filter_scores[c_filter][np.array(y_truth[read_ids[c_filter]] == 1) &
                                             np.array(y_pred[read_ids[c_filter]] == 1)]
                fp = filter_scores[c_filter][np.array(y_truth[read_ids[c_filter]] == 0) &
                                             np.array(y_pred[read_ids[c_filter]] == 1)]
                fn = filter_scores[c_filter][np.array(y_truth[read_ids[c_filter]] == 1) &
                                             np.array(y_pred[read_ids[c_filter]] == 0)]

                if len(tp) >= 1:
                    plt.hist(tp, alpha=0.3, bins=100, label="TP", color="blue")
                    data.append(tp)
                    boxplot_labels.append("TP")
                if len(tn) >= 1:
                    plt.hist(tn, alpha=0.3, bins=100, label="TN", color="green")
                    data.append(tn)
                    boxplot_labels.append("TN")
                if len(fp) >= 1:
                    plt.hist(fp, alpha=0.3, bins=100, label="FP", color="red")
                    data.append(fp)
                    boxplot_labels.append("FP")
                if len(fn) >= 1:
                    plt.hist(fn, alpha=0.3, bins=100, label="FN", color="orange")
                    data.append(fn)
                    boxplot_labels.append("FN")
                plt.title('distribution of contribution scores of ' + c_filter+'\n(' + args.mode + ')')
                plt.xlabel("contribution score")
                plt.ylabel("#reads")
                plt.legend(loc='upper right',  prop={'size': 10})
                plt.savefig(out_dir + "/distr_contribution_scores_" +
                            c_filter + "_wo_zeros_4_classes_" + args.mode + ".png", dpi=300)
                plt.gcf().clear()

                # plot distribution of contribution scores per filter (excluding zeros) as boxplots
                plt.boxplot(data, labels=boxplot_labels)

                plt.title('distribution of contribution scores of ' + c_filter+'\n(' + args.mode + ')')
                plt.ylabel("contribution score")
                plt.savefig(out_dir + "/boxplots_contribution_scores_" +
                            c_filter + "_wo_zeros_4_classes_" + args.mode + ".png", dpi=300)
                plt.gcf().clear()

    print("Rank filters according to their pathogenicity/non-pathogenicity potential")
    if args.mode == "original":
        with open(out_dir + "/ranking_filter_4_classes_patho_filter_" + args.mode + ".txt", 'w') as csv_file:
            file_writer = csv.writer(csv_file, delimiter='\t')
            file_writer.writerow(["filter", "mean_score_wo_zeros", "mean_score_w_zeros", "min", "max", "sensitivity",
                                  "specificity", "accuracy", "TP", "FP", "TN", "FN", "perc_nonzero_reads"])

        with open(out_dir + "/ranking_filter_4_classes_nonpatho_filter_" + args.mode + ".txt", 'w') as csv_file:
            file_writer = csv.writer(csv_file, delimiter='\t')
            file_writer.writerow(["filter", "mean_score_wo_zeros", "mean_score_w_zeros", "min", "max", "sensitivity",
                                  "specificity", "accuracy", "TP", "FP", "TN", "FN", "perc_nonzero_reads"])

        for c_filter, scores in filter_scores.items():

            if len(filter_scores[c_filter]) == 0:
                print("Skipping " + c_filter + ' ... (no data)')
                continue

            mean_scores_wo_zeros = np.mean(scores)
            scores_all_reads = np.zeros(len(y_truth))
            scores_all_reads[read_ids[c_filter]] = scores
            mean_scores_w_zeros = np.mean(np.concatenate((scores, [0]*(len(y_truth) - len(scores)))))

            # filter acts in average as a pathogenic motif detector
            if mean_scores_wo_zeros >= 0:
                # number reads which are labeled as pathogenic
                # and for which the filter has a positive contribution score
                tp = float(np.sum(np.array(y_truth == 1) & np.array(scores_all_reads > 0)))
                # number reads which are labeled as nonpathogenic
                # but for which the filter has a positive contribution score
                fp = float(np.sum(np.array(y_truth == 0) & np.array(scores_all_reads > 0)))
                # number reads which are labeled as pathogenic
                # but for which the filter has a nonpositive contribution score
                fn = float(np.sum(np.array(y_truth == 1) & np.array(scores_all_reads <= 0)))
                # number reads which are labeled as nonpathogenic
                # and for which the filter has a nonpositive contribution score
                tn = float(np.sum(np.array(y_truth == 0) & np.array(scores_all_reads <= 0)))

                metrics = [mean_scores_wo_zeros, mean_scores_w_zeros, min(scores), max(scores), tp/(tp+fn), tn/(fp+tn),
                           (tp+tn)/(tp+fp+fn+tn), tp, fp, tn, fn, float(len(scores))/float(len(scores_all_reads))]
                with open(out_dir + "/ranking_filter_4_classes_patho_filter_" + args.mode + ".txt", 'a') as csv_file:
                    file_writer = csv.writer(csv_file, delimiter='\t')
                    file_writer.writerow([c_filter] + metrics)
            # filter acts in average as a nonpathogenic motif detector
            else:
                # number reads which are labeled as nonpathogenic
                # and for which the filter has a negative contribution score
                tp = float(np.sum(np.array(y_truth == 0) & np.array(scores_all_reads < 0)))
                # number reads which are labeled as pathogenic
                # but for which the filter has a negative contribution score
                fp = float(np.sum(np.array(y_truth == 1) & np.array(scores_all_reads < 0)))
                # number reads which are labeled as nonpathogenic
                # but for which the filter has a nonnegative contribution score
                fn = float(np.sum(np.array(y_truth == 0) & np.array(scores_all_reads >= 0)))
                # number reads which are labeled as pathogenic
                # and for which the filter has a nonnegative contribution score
                tn = float(np.sum(np.array(y_truth == 1) & np.array(scores_all_reads >= 0)))

                metrics = [mean_scores_wo_zeros, mean_scores_w_zeros, min(scores), max(scores), tp/(tp+fn), tn/(fp+tn),
                           (tp+tn)/(tp+fp+fn+tn), tp, fp, tn, fn, float(len(scores))/float(len(scores_all_reads))]
                with open(out_dir + "/ranking_filter_4_classes_nonpatho_filter_" + args.mode + ".txt", 'a') as csv_file:
                    file_writer = csv.writer(csv_file, delimiter='\t')
                    file_writer.writerow([c_filter] + metrics)
