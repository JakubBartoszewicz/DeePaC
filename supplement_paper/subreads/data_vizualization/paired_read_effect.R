library(ggplot2)
library(gridExtra)
library(tidyverse)
library(stringr)
library(gridExtra)

# data preparation
read_eval_data <- readRDS("read_eval_data.rds")
read_eval_data <- read_eval_data%>%
  filter(str_detect(set,"fl"))

read_eval_data$fragment_length <- str_extract(read_eval_data$set,"\\d*$")
read_eval_data$fragment_length <-  as.numeric(read_eval_data$fragment_length)

read_eval_data <- read_eval_data%>%
    group_by(training,fragment_length,subread_length)%>%
    mutate(acc_diff = acc[3]-acc[1],
           auroc_diff = auroc[3]-auroc[1],
           recall_diff = recall[3]-recall[1],
           precision_diff = precision[3]-precision[1],
           spec_diff = spec[3]-spec[1]
           )

plot_paired_read_eff <- function(metric,ylab,ylim=c(NA,1),table=FALSE,nn_type, read_lengths=c(50,100,150,200,250)){
  # plots the course of a metric on different fragments lengths (y=metric,x=fl) and a table summerizing the plot 
  # the metric is shown for paired and single reads
  #
  # Args:
  #   metric: e.g. acc, auroc ...
  #   ylab: label y axis 
  #   ylim: limits y axis as vector
  #   nn_type: either cnn or lstm
  #   table: show table (Bool)
  #   read_lengths
  # Returns:
  #   plot and opt. table
  # compute plot
  data_temp <- read_eval_data%>%
    filter(subread_length%in%read_lengths)%>%
    filter(type==nn_type)
  
  data_temp_paired <- data_temp%>%
    filter(pred_type=="paired_read")
  
  data_temp_single <- data_temp%>%
    filter(pred_type=="single_read")
  
  fragment_lengths=unique(data_temp$fragment_length)
  
plot <- 
  # paired read data
  ggplot(data = data_temp_paired,
         mapping = aes(x=log(fragment_length),
                       y=get(metric), 
                       group=paste(subread_length,pred_type),
                       color=subread_length,
                       shape=subread_length)) + 
    geom_point(size = 4,shape = 7) + 
    geom_line(size = 2) +
    scale_x_continuous(breaks = log(fragment_lengths),
                       labels = fragment_lengths) +
    # single read as ref
    geom_point(data =  data_temp_single,
               color="grey",show.legend = FALSE,shape = 6) +
    geom_line(data =  data_temp_single,
              color="grey", linetype =2,show.legend = FALSE) + 
    # plot design
    xlab("mean fragment length (bp)") + 
    ylab(ylab) + ylim(ylim) +
    theme_classic() +
    theme(axis.text.x = element_text(angle = 45,vjust = 0.5),
          axis.text = element_text(face = "bold",size = 14),
          axis.title  = element_text(face = "bold",size = 16),
          panel.grid.major = element_line(),
          legend.position = "top") +
    scale_color_brewer(palette = "Dark2")


  if(table){
  data_temp <- data_temp[,c("fragment_length",metric,"subread_length","pred_type","set")]
  colnames(data_temp) <- c(colnames(data_temp)[1],"metric",colnames(data_temp)[c(3:5)])
  
  # create table with plot data
  data_temp <- data_temp%>%
    mutate(set = str_extract(set,"test_1|test_2"))%>%
    ungroup()%>%
    mutate(metric=round(metric,digits = 3))%>%
    spread(fragment_length,metric)%>%
    select(-set)
  
  # devide table
  table_1 <- tableGrob(data_temp[,c(1,2,3:(round(length(data_temp)/2)))],rows = NULL,theme = ttheme_minimal(base_size = 16)) 
  table_2 <- tableGrob(data_temp[,c(1,2,(round(length(data_temp) /2)+1):length(data_temp))],rows = NULL,theme = ttheme_minimal(base_size = 16)) 
  
  # combine plot and table
  plot <- grid.arrange(plot,table_1,table_2,layout_matrix=rbind(c(1,1),
                                                                  c(1,1),
                                                                  c(2,2),
                                                                  c(3,3)))
} 
  return(plot)
}

# overview plots
cnn_overview <- plot_paired_read_eff(ylab = "accuracy",metric = "acc",ylim = c(0.5,0.9),table = FALSE,nn_type = "cnn") 
cnn_overview

lstm_overview <- plot_paired_read_eff(ylab = "accuracy",metric = "acc",ylim = c(0.5,0.9),table = FALSE,nn_type = "lstm") 
lstm_overview

# plot all metrics on all subread lengths
for(type in c("cnn","lstm")){
    for(read_length in c(50,100,150,200,250)){
      plot <- plot_paired_read_eff(ylab = "accuracy",metric = "acc",read_lengths = read_length,ylim = c(0.5,0.9),table = TRUE,nn_type = type)
      ggsave(filename = paste0("./",type,read_length,"_acc.jpeg"),plot = plot,width = 11, height = 12)
    }
}

for(type in c("cnn","lstm")){
  for(read_length in c(50,100,150,200,250)){
    plot <- plot_paired_read_eff(ylab = "auroc",metric = "auroc",read_lengths = read_length,ylim = c(0.6,1),table = TRUE,nn_type = type)
    ggsave(filename = paste0("./",type,read_length,"_auroc.jpeg"),plot = plot,width = 11, height = 12)
  }
}
  
for(type in c("cnn","lstm")){
  for(read_length in c(50,100,150,200,250)){
    plot <- plot_paired_read_eff(ylab = "precision",metric = "precision",read_lengths = read_length,ylim = c(0.5,1),table = TRUE,nn_type = type)
    ggsave(filename = paste0("./",type,read_length,"_precision.jpeg"),plot = plot,width = 11, height = 12)
    }
}
  
for(type in c("cnn","lstm")){
  for(read_length in c(50,100,150,200,250)){
    plot <- plot_paired_read_eff(ylab = "recall",metric = "recall",read_lengths = read_length,ylim = c(0.5,1),table = TRUE,nn_type = type)
    ggsave(filename = paste0("./",type,read_length,"_recall.jpeg"),plot = plot,width = 11, height = 12)
  }
}

for(type in c("cnn","lstm")){
  for(read_length in c(50,100,150,200,250)){
    plot <- plot_paired_read_eff(ylab = "spec",metric = "spec",read_lengths = read_length,ylim = c(0,1),table = TRUE,nn_type = type)
    ggsave(filename = paste0("./",type,read_length,"_spec.jpeg"),plot = plot,width = 11, height = 12)
  }
}
