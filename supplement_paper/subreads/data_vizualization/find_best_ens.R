library(ggplot2)
library(gridExtra)
library(tidyverse)

read_eval_data <- readRDS("read_eval_data.rds")

#plots ----
# filter data
read_eval_data_plot <- read_eval_data%>%
  ungroup()%>%
  filter(pathogen_type=="img",
         type %in% c("cnn+cnn","cnn+lstm","lstm+cnn","lstm+lstm"))%>%
  filter(training == training[acc==max(acc[subread_length == "250"])]  )

# vhdb
# cnn_150bp_d025_vhdb_all-e014_cnn_250bp_d025_vhdb_all-e011-metrics.csv best 250bp value
# cnn_150bp_d025_vhdb_all-e014_cnn_25-250bp_d025_vhdb_all-e010-metrics.csv best avg

# img

# cnn_200bp_d02_img-e013_lstm_150bp_d025_img-e009-metrics.csv best avg
# cnn_200bp_d02_img-e013_lstm_200bp_d02_img-e010-metrics.csv best 250