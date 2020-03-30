library(ggplot2)
library(gridExtra)
library(tidyverse)
library(xtable)
library(RColorBrewer)

# filter data
read_eval_data <- readRDS("read_eval_data.rds")

read_eval_data <- read_eval_data%>%
  filter(subread_length != "avg")%>%
  filter(set  %in% c("test_1","all_test_1") | 
           (set %in% c("test_1_test_2","all_test_1_all_test_2") & subread_length == 250))%>%
  group_by(training)%>%
  # keep only data where we have single and paired pred
  filter(n() == 20)%>%
  mutate(seq_cycle = ifelse(!is.na(subread_length_2),
                            as.integer(as.character(subread_length)) + as.integer(as.character(subread_length_2)) + 9,
                            as.integer(as.character(subread_length))))%>%
  mutate(seq_cycle = factor(x = seq_cycle,levels = sort(unique(seq_cycle))),
         pred_type = factor(pred_type,levels = c("single_read","paired_read"),ordered = T))%>%
  ungroup()


# plot func
plot_metric_vs_seqc <- function(data,metric,ylab,ylim){
  # plots the course of a metric on different subreadlengths/seq cycles (y=metric,x=seq_cycle) 
  # comparing individual classifiers
  # read lengths 25-250bp are single read, 283-508 are paired read
  #
  # Args:
  #   data: dataframe cont data
  #   metric: e.g. acc, auroc ...
  #   ylab: label y axis 
  #   ylim: limits y axis as vector
  #
  # Returns:
  #   plot
  # compute plot
  plot <- ggplot(data = get(data),
                 mapping = aes(seq_cycle,
                               get(metric), 
                               group=training,
                               color=training)) + 
    geom_point(size=3,shape=4) + 
    geom_line(size=0.8) + 
    theme_classic() + theme(axis.text = element_text(face = "bold",size = 11),
                            axis.title = element_text(face = "bold",size = 12),
                            legend.text =  element_text(face = "bold",size = 11),
                            legend.title  =  element_text(face = "bold",size = 12),
                            panel.grid.major = element_line(),
                            legend.position = "left",
                            strip.text = element_blank()) +
    scale_y_continuous(name = ylab, limits = ylim) +
    scale_x_discrete(name = "sequencing cycle",
                     breaks = sort(unique(read_eval_data_plot$seq_cycle))) + 
    facet_grid(.~pred_type,scales = "free") + scale_colour_brewer(name = "Tool:", palette = "Paired")
  return(plot)
}

# img plots----
trainings_plot_img <- c("cnn_150bp_d02_img-e003","lstm_25-250bp_d02_img-e009",
                        "img-cnn","img-lstm",
                        "cnn_250bp_d025_img","lstm_250bp_d02_img",
                        "PaPrBaG DNA_25-250","Blast_TestB" )

labels_plot_img <- c("CNN (ours)", "LSTM (ours)", 
                     "HiLive2+CNN", "HiLive2+LSTM",
                     "DeePaC (CNN)", "DeePaC (LSTM)",
                     "PaPrBaG", "BLAST")

read_eval_data_plot <- read_eval_data%>%
  filter(training %in% trainings_plot_img)%>%
  mutate(training = factor(training,levels = trainings_plot_img,labels = labels_plot_img))

acc_img_zoom <- plot_metric_vs_seqc("read_eval_data_plot","acc","accuracy",c(0.5,0.9))
acc_img_zoom

# latex table
read_eval_data_figure <- read_eval_data_plot%>%
  select(acc,precision,recall,training)%>%
  group_by(training)%>%
  summarise_if(is.numeric,function(x){round(mean(x)*100,3) })
xtable(read_eval_data_figure,digits = 1)

# plots vhdb ----
trainings_plot_vhdb <- c("cnn_25-250bp_d025_vhdb_all-e010","cnn_150bp_d025_vhdb_all-e014_cnn_25-250bp_d025_vhdb_all-e010",
                         "vhdb-cnn-mix","vhdb-ens",
                         "cnn_150bp_d025_vhdb_all-e014","lstm_250bp_d02_vhdb-e013_all",
                         "knn_vhdb","Blast_TestV",
                         "vhdb-hilive")

labels_plot_vhdb <- c("CNN (ours)", "ENS (ours)", 
                      "HiLive2+CNN", "HiLive2+ENS",
                      "DeePaC (150bp)", "DeePaC (LSTM)",
                      "kNN", "BLAST",
                      "HiLive2")

read_eval_data_plot <- read_eval_data%>%
  filter(training %in% trainings_plot_vhdb[1:8])%>%
  mutate(training = factor(training,levels = trainings_plot_vhdb,labels = labels_plot_vhdb))

acc_vhdb_zoom <- plot_metric_vs_seqc("read_eval_data_plot","acc","accuracy",c(0.5,0.9))
acc_vhdb_zoom

read_eval_data_plot <- read_eval_data%>%
  filter(training %in% trainings_plot_vhdb[c(1:4,9)])%>%
  mutate(training = factor(training,levels = trainings_plot_vhdb,labels = labels_plot_vhdb))

colors = brewer.pal(n=10,"Paired")

ppv_vhdb_zoom <- plot_metric_vs_seqc("read_eval_data_plot","precision","precision",c(.5,1))  + 
  scale_color_manual(name="Tool:",values = colors[c(1:4,9)])
ppv_vhdb_zoom


# aureus plots ----
trainings_plot_aureus <- c("aureus-cnn",
                           "aureus-lstm",
                           "aureus-hilive" )

labels_plot_aureus <- c("HiLive2+CNN", 
                        "HiLive2+LSTM",
                        "HiLive2")

read_eval_data_plot <- read_eval_data%>%
  filter(training %in% trainings_plot_aureus)%>%
  mutate(training = factor(training,levels = trainings_plot_aureus,labels = labels_plot_aureus))

colors = brewer.pal(n=10,"Paired")

recall_aureus <- plot_metric_vs_seqc("read_eval_data_plot","recall","recall",c(0,1)) + 
  scale_color_manual(name="Tool:",values = colors[c(3,4,9)])
recall_aureus


#save results-----
for(obj in ls()){
  plot = get(obj)
  if("ggplot" %in% class(plot)){
    ggsave(  filename = paste0("~/Dropbox/HPI_RKI/DeePaC/data_vizualization/plots/",obj,".jpg"),plot = plot,width = 9,height = 4)
  }
}