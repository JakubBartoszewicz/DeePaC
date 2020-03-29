library(ggplot2)
library(gridExtra)
library(tidyverse)

read_eval_data <- readRDS("read_eval_data.rds")

# filter data for plots (create envelope of best performing classifiers)
read_eval_data_plot <- read_eval_data%>%
  filter(!subread_length %in% c("mixed(50-250bp)","avg"),
         type %in% c("cnn","lstm","cnn+lstm","lstm+cnn", "random_forest"),
         set == "test_1",
         pred_type == "single_read")%>%
  group_by(type,subread_length)%>%
  # keep only the best classifiers
  filter(acc == max(acc))

plot_envelope <- function(metric,ylab,ylim,title){
  ggplot(data = read_eval_data_plot,
         mapping = aes(subread_length,get(metric),  color=type,  group=type, shape=trained_on)) +
    geom_point(size=4) + geom_line(size=0.8) +
    theme_classic() + theme(axis.text = element_text(face = "bold"),
                            panel.grid.major = element_line(),
                            legend.position = "top") +
    scale_y_continuous(name = ylab, limits = ylim) 
}

acc <- plot_envelope("acc","Accuracy", c(0.4,1))
auc <- plot_envelope("auroc","AUROC",c(0.4,1))
sens <- plot_envelope("recall","sens",c(0.4,1))
spec <- plot_envelope("spec","spec",c(0.4,1))
ppv <- plot_envelope("precision","precision",c(0.4,1))

# combine plots
comp <- grid.arrange(acc + scale_shape_discrete(guide=F),
                     auc + scale_color_discrete(guide=F),
                     sens+theme(legend.position = "none"),
                     spec+theme(legend.position = "none"),
                     ppv+theme(legend.position = "none"),layout_matrix = rbind(c(NA,1,1,2,2,NA),c(3,3,4,4,5,5))
)

ggsave(filename = paste0("plots/", "compare_model_types_enevlope",".jpg"),plot = comp,width = 11,height = 7.5)