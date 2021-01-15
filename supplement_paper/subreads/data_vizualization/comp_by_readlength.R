library(ggplot2)
library(gridExtra)
library(tidyverse)

read_eval_data <- readRDS("read_eval_data.rds")

#plots ----
# filter data
cnns_img <-  c("cnn_25-250bp_d02_img-e001",
              "cnn_50bp_d02_img-e008",
              "cnn_100bp_d02_img-e001",
              "cnn_150bp_d02_img-e003",
              "cnn_200bp_d02_img-e003",
              "cnn_250bp_d025_img")

lstms_img <- c("lstm_25-250bp_d02_img-e009",
               "lstm_50bp_d02_img-e009",
               "lstm_100bp_d02_img-e005",
               "lstm_150bp_d02_img-e010",
               "lstm_200bp_d02_img-e010",
               "lstm_250bp_d02_img")

cnns_vhdb <-  c("cnn_25-250bp_d025_vhdb_all-e010",
                "cnn_150bp_d025_vhdb_all-e014",
                "cnn_250bp_d025_vhdb_all-e011")

lstms_vhdb <- c("lstm_25-250bp_d02_vhdb_all-e001",
                "lstm_150bp_d025_vhdb_all-e005",
                "lstm_250bp_d02_vhdb_all-e013")


read_eval_data_plot <- read_eval_data%>%
  filter(training %in% lstms_vhdb,
         set == "all_test_1")%>%
  mutate(trained_on = factor(trained_on,levels = c("50bp","100bp","150bp","200bp","250bp","25-250bp"),ordered = T))

plot_metric_vs_subl <- function(data,metric,ylab,ylim,table=F){
  # plots the course of a metric on different subreadlengths (y=metric,x=length) and a table summerizing the plot 
  #
  # Args:
  #   data: plot dataframe
  #   metric: e.g. acc, auroc ...
  #   ylab: label y axis 
  #   ylim: limits y axis as vector
  #   table: show table (Bool)
  #
  # Returns:
  #   plot and opt. table
  # compute plot
  plot <- ggplot(
    data = get(data) %>%
      filter(!subread_length == "avg"),
    mapping = aes(
      subread_length,
      get(metric),
      group = training,
      shape = trained_on,
      color = trained_on
    )
  ) +
    geom_point(size = 4) + 
    geom_line(size = 0.8) +
    theme_classic() + theme(
      axis.text = element_text(face = "bold"),
      panel.grid.major = element_line(),
      legend.position = "top"
    ) +
    xlab("subread length [bp]") + 
    scale_y_continuous(name = ylab, limits = ylim) +
    scale_shape_discrete(name = paste0(get(data)$type[1], "\ntrained on:")) +
    scale_color_brewer(palette = "Dark2",
                       name = paste0(get(data)$type[1], "\ntrained on:"))
  
  
  if (table) {
    data_temp <-
      get(data)[, c(
        "subread_length",
        metric,
        "type",
        "trained_on",
        "training"
      )]
    colnames(data_temp)[colnames(data_temp) == metric] = "metric"
    
    data_temp <- data_temp %>%
      ungroup() %>%
      mutate(metric = round(metric, digits = 3)) %>%
      spread(subread_length, metric)
    
    colnames(data_temp)[colnames(data_temp) == "metric"] = metric
    
    table <- tableGrob(data_temp, rows = NULL)
    plot <- grid.arrange(plot, table)
  }
  return(plot)
}
  
acc <- plot_metric_vs_subl("read_eval_data_plot","acc","accuracy",c(0.45,0.9),table = T)
acc

spec <- plot_metric_vs_subl("read_eval_data_plot","spec","tnr",c(0,1),table = T)
spec

sens <- plot_metric_vs_subl("read_eval_data_plot","recall","tpr",c(0,1),table = T)
sens

ppv <-  plot_metric_vs_subl("read_eval_data_plot","precision","ppv",c(0,1),table = T)
ppv

auc <-  plot_metric_vs_subl("read_eval_data_plot","auroc","auroc",c(0.5,1),table = T)
auc

#save results-----
for(obj in ls()){
  plot = get(obj)
  if("ggplot" %in% class(plot)){
    ggsave(  filename = paste0("~/Dropbox/HPI_RKI/DeePaC/data_vizualization/plots/",obj,".jpg"),plot = plot,width = 6,height = 4)
  } else if ("grob" %in% class(plot)){
    ggsave(  filename = paste0("~/Dropbox/HPI_RKI/DeePaC/data_vizualization/plots/",obj,".jpg"),plot = plot,width = 15,height = 6)
  }
}

