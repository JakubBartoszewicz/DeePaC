library(tidyverse)
library(gridExtra)
#first vs last delete--------
read_eval_data <- readRDS("read_eval_data.rds")

# new plots
models_kept = c("lstm_50bp_d02_img-e009", "lstm_150bp_d025_img-e009","lstm_250bp_d02_img")
colors= c("darkgrey","darkgrey","darkgrey","darkgrey","black")
levels =c("50bp","100bp","150bp","200bp","250bp")
read_eval_data_plot <- read_eval_data%>%
  filter(training %in% models_kept,
         set == "test_1")%>%
  mutate(trained_on = factor(trained_on, levels =levels))



plot_comp_to_baseline <- function(metric,ylab,ylim,colors,breaks){
  # plots the course of a metric over different subreadlengths (y=metric,x=length) and a table summerizing the plot 
  # function similar to comp_by_readlength.R (plot_metric_vs_subl), difference is the color encoding 
  #
  # Args:
  #   data: plot dataframe
  #   metric: e.g. acc, auroc ...
  #   ylab: label y axis 
  #   ylim: limits y axis as vector
  #   colors: colors used for diff training read lengths
  #   breaks: training read lengths
  # Returns:
  #   plot 
  #
  # compute plot
  ggplot(data = read_eval_data_plot%>%filter(subread_length != "avg"),
         mapping = aes(subread_length,get(metric), 
                       group=training,shape=trained_on, 
                       color=trained_on)) + 
    geom_point(size=5) + 
    geom_line(size=1.5,show.legend = FALSE) + 
    # design 
    ylim(ylim) +
    theme_bw() + 
    theme(legend.position = "top",
          axis.title = element_text(face = "bold",size = 12),
          axis.text = element_text(face = "bold",size = 12),
          axis.text.x = element_text(angle = 45,vjust = 0.5)) +
    scale_shape_discrete(name = "subread length (training):") + 
    scale_color_manual(name = "subread length (training):",
                       breaks = levels,
                       values = colors) +
    ylab(ylab) + xlab("subread length [bp]") + 
    guides(shape = guide_legend(nrow = 1))
}

acc <- plot_comp_to_baseline("acc","Accuracy",c(0.5,1),colors,levels)
acc

auroc <- plot_comp_to_baseline("auroc","AUROC",c(0.5,1),colors,levels)
auroc

tpr <- plot_comp_to_baseline("recall","recall",c(0,1),colors,levels)
tpr

tnr <- plot_comp_to_baseline("spec","spec",c(0,1),colors,levels)
tnr

ppv <- plot_comp_to_baseline("precision","precision",c(0,1),colors,levels)
ppv


# combine plots
acc_auroc <- grid.arrange(acc,auroc,nrow=1)
tpr_tnr_ppv <- grid.arrange(tpr+theme(legend.position = "none"),
                              tnr+theme(legend.position = "none"),
                              ppv+theme(legend.position = "none"),nrow=1)

# legend is best adjusted manualy (not with r)
figure <- grid.arrange(acc_auroc,tpr_tnr_ppv,layout_matrix=rbind(c(NA,1,1,1,1,NA),
                                                                 c(2,2,2,2,2,2)))
