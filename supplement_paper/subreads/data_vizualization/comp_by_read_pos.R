library(ggplot2)
library(gridExtra)
library(tidyverse)

read_eval_data <- readRDS("read_eval_data.rds")

#first vs last delete--------
read_eval_data_plot <- read_eval_data%>%
  filter(trained_on %in% c("200bp"),
         type == "lstm",
         !subread_length %in% c("mixed","avg"),
         set == "test_1")

plot_subread_pos <- function(metric,ylab,ylim){
  ggplot(data = read_eval_data_plot,
         mapping = aes(subread_length,get(metric),  
                       color=paste(trained_on,subread_pos),  
                       group=paste(trained_on,subread_pos,training))) +
    geom_point(size=4) + geom_line(size=0.8) +
    theme_classic() + theme(axis.text = element_text(face = "bold"),
                            panel.grid.major = element_line(),
                            legend.margin = margin(0,0,0,0),
                            legend.box.margin = margin(0,0,0,0),
                            legend.spacing = unit(0,"cm")) +
    scale_x_discrete(breaks = c("25bp","50bp","75bp","100bp","125bp","150bp","175bp","200bp","225bp","250bp"),
                     labels = c(seq(25,250,25)),
                     name = "subread length (bp)") +
    scale_y_continuous(name = ylab, limits = ylim) + theme(legend.position = "left") +
    scale_color_manual(values = c("green4","green","darkred"),name="LSTM\ntrained on:")
}

acc <- plot_subread_pos("acc","Accuracy",c(0.4,1))
acc

ggsave(filename = paste0("~/Documents/RKI/results/", "200FirstvsLast_lstm",".jpg"),plot = acc,width = 6,height = 4)