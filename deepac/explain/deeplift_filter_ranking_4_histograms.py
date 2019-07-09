import argparse
import os
import csv
import re #regular expressions
import numpy as np
import matplotlib.pyplot as plt
import csv


'''
Plot contribution scores per filter and rank filters according to their pathogenicity/non-pathogenicity potential.
'''

#parse command line options
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", default = "original", choices=["original", "rel_true_class", "rel_pred_class"], help = "Use original filter scores or normalize scores relative to true or predicted classes.")
parser.add_argument("-f", "--scores_dir", required=True, help="Directory containing filter contribution scores (.csv)")
parser.add_argument("-y", "--true_label", required=True, help="File with true read labels (.npy)")
parser.add_argument("-p", "--pred_label", required=True, help="File with predicted read labels (.npy)")
parser.add_argument("-o", "--out_dir", required=True, help="Output directory")
args = parser.parse_args()


#create output directory
out_dir = args.out_dir + "/" + args.mode
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


#load true labels
y_truth = np.load(args.true_label, mmap_mode='r')
num_patho_reads = np.sum(y_truth == 1)
num_nonpatho_reads = np.sum(y_truth == 0)
#load predicted labels
y_pred = np.load(args.pred_label, mmap_mode='r')
y_pred = (y_pred > 0.5).astype('int32')

#nonzero scores per filter
filter_scores = dict()
#nonzero reads per filter
read_ids = dict()
for file in os.listdir(args.scores_dir):
    if file.endswith(".csv") and os.stat(args.scores_dir + "/" + file).st_size > 0:
        with open(args.scores_dir + "/" + file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            filter = re.search("filter_[0-9]+", file).group()
            read_ids[filter] = []
            filter_scores[filter] = []
            for ind, row in enumerate(reader):
                if ind % 3 == 0:
                    read_id = re.search("seq_[0-9]+", row[0]).group()
                    read_id = int(read_id.replace("seq_", ""))
                    read_ids[filter].append(read_id)
                elif ind % 3 == 2:
                    score = float(row[0])
                    filter_scores[filter].append(score)
            read_ids[filter] = np.array(read_ids[filter])
            filter_scores[filter] = np.array(filter_scores[filter])
            if args.mode == "rel_pred_class":
                filter_scores[filter][y_pred[read_ids[filter]] == 0] *= -1
            if args.mode == "rel_true_class":
                filter_scores[filter][y_truth[read_ids[filter]] == 0] *= -1

            #plot distribution of contribution scores per filter (excluding zeros) as histograms
            print("Plotting the distribution of the contribution scores of " + filter + ' ...')
            TN = filter_scores[filter][np.array(y_truth[read_ids[filter]] == 0) & np.array(y_pred[read_ids[filter]] == 0)]
            TP = filter_scores[filter][np.array(y_truth[read_ids[filter]] == 1) & np.array(y_pred[read_ids[filter]] == 1)]
            FP = filter_scores[filter][np.array(y_truth[read_ids[filter]] == 0) & np.array(y_pred[read_ids[filter]] == 1)]
            FN = filter_scores[filter][np.array(y_truth[read_ids[filter]] == 1) & np.array(y_pred[read_ids[filter]] == 0)]
            if len(TN) > 1: plt.hist(TN, alpha = 0.3, bins = 100, label = "TN", color = "green")
            if len(TP) > 1: plt.hist(TP, alpha = 0.3, bins = 100, label = "TP", color = "blue")
            if len(FP) > 1: plt.hist(FP, alpha = 0.3, bins = 100, label = "FP", color = "red")
            if len(FN) > 1: plt.hist(FN, alpha = 0.3, bins = 100, label = "FN", color = "orange")
            plt.title('distribution of contribution scores of '+filter+'\n(' + args.mode + ')')
            plt.xlabel("contribution score")
            plt.ylabel("#reads")
            plt.legend(loc='upper right',  prop={'size': 10})
            plt.savefig(out_dir + "/distr_contribution_scores_" + filter + "_wo_zeros_4_classes_"+ args.mode + ".png")
            plt.gcf().clear()

            #plot distribution of contribution scores per filter (excluding zeros) as boxplots
            data = [TP, FN, TN, FP]
            plt.boxplot(data, labels = ["TP", "FN", "TN", "FP"])
            #plt.title('distribution of contribution scores of '+filter+'\n(' + args.mode + ')')
            plt.xlabel("contribution score")
            plt.ylabel("#reads")
            plt.legend(loc='upper right',  prop={'size': 12})
            plt.savefig(out_dir + "/boxplots_contribution_scores_" + filter + "_wo_zeros_4_classes_"+ args.mode + ".png")
            plt.gcf().clear()


print("Rank filters according to their pathogenicity/non-pathogenicity potential")
if args.mode == "original":
    with open(out_dir + "/ranking_filter_4_classes_patho_filter_" + args.mode + ".txt", 'w') as csv_file:
        file_writer = csv.writer(csv_file, delimiter='\t')
        file_writer.writerow(["filter", "mean_score_wo_zeros", "mean_score_w_zeros", "min", "max", "sensitivity", "specificity", "accuracy", "TP", "FP", "TN", "FN", "perc_nonzero_reads"])

    with open(out_dir + "/ranking_filter_4_classes_nonpatho_filter_" + args.mode + ".txt", 'w') as csv_file:
        file_writer = csv.writer(csv_file, delimiter='\t')
        file_writer.writerow(["filter", "mean_score_wo_zeros", "mean_score_w_zeros", "min", "max", "sensitivity", "specificity", "accuracy", "TP", "FP", "TN", "FN", "perc_nonzero_reads"])

    for filter, scores in filter_scores.items():

        mean_scores_wo_zeros = np.mean(scores)
        scores_all_reads = np.zeros(len(y_truth))
        scores_all_reads[read_ids[filter]] = scores
        mean_scores_w_zeros = np.mean(np.concatenate((scores, [0]*(len(y_truth)- len(scores)))))


        #filter acts in average as a pathogenic motif detector
        if mean_scores_wo_zeros >= 0:
            #number reads which are labeled as pathogenic and for which the filter has a positive contribution score
            TP = float(np.sum(np.array(y_truth == 1) & np.array(scores_all_reads > 0)))
            #number reads which are labeled as nonpathogenic but for which the filter has a positive contribution score
            FP = float(np.sum(np.array(y_truth == 0) & np.array(scores_all_reads > 0)))
            #number reads which are labeled as pathogenic but for which the filter has a nonpositive contribution score
            FN = float(np.sum(np.array(y_truth == 1) & np.array(scores_all_reads <= 0)))
            #number reads which are labeled as nonpathogenic and for which the filter has a nonpositive contribution score
            TN = float(np.sum(np.array(y_truth == 0) & np.array(scores_all_reads <= 0)))

            metrics = [mean_scores_wo_zeros, mean_scores_w_zeros, min(scores), max(scores), TP/(TP+FN), TN/(FP+TN), (TP+TN)/(TP+FP+FN+TN), TP, FP, TN, FN, float(len(scores))/float(len(scores_all_reads))]
            with open(out_dir + "/ranking_filter_4_classes_patho_filter_" + args.mode + ".txt", 'a') as csv_file:
                file_writer = csv.writer(csv_file, delimiter='\t')
                file_writer.writerow([filter] + metrics)
        #filter acts in average as a nonpathogenic motif detector
        else:
            #number reads which are labeled as nonpathogenic and for which the filter has a negative contribution score
            TP = float(np.sum(np.array(y_truth == 0) & np.array(scores_all_reads < 0)))
            #number reads which are labeled as pathogenic but for which the filter has a negative contribution score
            FP = float(np.sum(np.array(y_truth == 1) & np.array(scores_all_reads < 0)))
            #number reads which are labeled as nonpathogenic but for which the filter has a nonnegative contribution score
            FN = float(np.sum(np.array(y_truth == 0) & np.array(scores_all_reads >= 0)))
            #number reads which are labeled as pathogenic and for which the filter has a nonnegative contribution score
            TN = float(np.sum(np.array(y_truth == 1) & np.array(scores_all_reads >= 0)))

            metrics = [mean_scores_wo_zeros, mean_scores_w_zeros, min(scores), max(scores), TP/(TP+FN), TN/(FP+TN), (TP+TN)/(TP+FP+FN+TN), TP, FP, TN, FN, float(len(scores))/float(len(scores_all_reads))]
            with open(out_dir + "/ranking_filter_4_classes_nonpatho_filter_" + args.mode + ".txt", 'a') as csv_file:
                file_writer = csv.writer(csv_file, delimiter='\t')
                file_writer.writerow([filter] + metrics)
