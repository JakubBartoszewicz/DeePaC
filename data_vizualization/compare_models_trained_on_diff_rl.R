library(ggplot2)
library(gridExtra)
library(tidyverse)

read_eval_data <- readRDS("read_eval_data.rds")

#plots ----
# filter data
read_eval_data_plot <- read_eval_data%>%
  filter(seq_tech_train_data == "nanopore",
         type == "lstm",
         !subread_length %in% c("avg","mixed(50-250bp)"),
         pred_type == "single_read" & set == "test_nano_500" |
         pred_type == "single_read" & set == "test_2_nano")

plotDesign <- function(data,metric,ylab,ylim,table=F){
    plot <- ggplot(data = get(data),
         mapping = aes(subread_length,get(metric), group=training, 
                       shape=trained_on, 
                       color=trained_on)) + 
    geom_point(size=4) + geom_line(size=0.8) + 
    theme_classic() + theme(axis.text = element_text(face = "bold"),
                            panel.grid.major = element_line(),
                            legend.position = "left") +
    scale_y_continuous(name = ylab, limits = ylim) + 
    scale_shape_discrete(name = paste0(get(data)$type[1], "\ntrained on:")) + 
    scale_color_brewer(palette = "Dark2",name = paste0(get(data)$type[1], "\ntrained on:"))
  
  if(table){
    data_temp <- get(data)[,c("subread_length",metric,"type","trained_on")]
    colnames(data_temp)[colnames(data_temp) == metric] = "metric"
    
    data_temp <- data_temp%>%
      ungroup()%>%
      mutate(metric=round(metric,digits = 3))%>%
      spread(subread_length,metric)
    
    colnames(data_temp)[colnames(data_temp) == "metric"] = metric
    
    table <- tableGrob(data_temp,rows = NULL)
    plot <- grid.arrange(plot,table)
  }
  return(plot)
}
  
acc <- plotDesign("read_eval_data_plot","acc","accuracy",c(0.45,0.9),table = T)
acc

spec <- plotDesign("read_eval_data_plot","spec","tnr",c(0,0.9),table = T)
spec

sens <- plotDesign("read_eval_data_plot","recall","tpr",c(0.3,1),table = T)
sens

ppv <-  plotDesign("read_eval_data_plot","precision","ppv",c(0.4,0.9),table = T)
ppv

auc <-  plotDesign("read_eval_data_plot","auroc","auroc",c(0.5,1),table = T)
auc

#save results-----
for(obj in ls()){
  plot = get(obj)
  if("ggplot" %in% class(plot)){
    ggsave(  filename = paste0("~/Dropbox/HPI_RKI/data_vizualization/plots/compare_read_lengths_nanopore/250_single_vs_500_single_read_lstm_",obj,".jpg"),plot = plot,width = 6,height = 4)
  } else if ("grob" %in% class(plot)){
    ggsave(  filename = paste0("~/Dropbox/HPI_RKI/data_vizualization/plots/compare_read_lengths_nanopore/250_single_vs_500_single_read_lstm_",obj,".jpg"),plot = plot,width = 13,height = 6)
  }
}

