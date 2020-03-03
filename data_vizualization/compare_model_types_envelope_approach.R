library(ggplot2)
library(gridExtra)
library(tidyverse)

single_read_data <- readRDS("single_read_data.rds")

# filter data for plots (create envelope of best performing classifiers)
single_read_data_plot <- single_read_data%>%
  filter(!subread_length %in% c("mixed(50-250bp)","avg"),
         type %in% c("cnn","lstm","cnn+lstm","random_forest"),
         set == "test_1",
         pred_type == "single_read")%>%
  group_by(type,subread_length)%>%
  # keep only the best classifiers
  filter(acc == max(acc))

plotDesign <- function(metric,ylab,ylim,title){
  ggplot(data = single_read_data_plot,
         mapping = aes(subread_length,get(metric),  color=type,  group=type, shape=trained_on)) +
    geom_point(size=4) + geom_line(size=0.8) +
    theme_classic() + theme(axis.text = element_text(face = "bold"),
                            panel.grid.major = element_line(),
                            legend.position = "left") +
    scale_y_continuous(name = ylab, limits = ylim) 
}

acc <- plotDesign("acc","Accuracy", c(0.4,1))
auc <- plotDesign("auroc","AUROC",c(0.4,1))
sens <- plotDesign("recall","sens",c(0.4,1))
spec <- plotDesign("spec","spec",c(0.4,1))
ppv <- plotDesign("precision","precision",c(0.4,1))

comp <- grid.arrange(acc,
                     auc+theme(legend.position = "none"),
                     sens+theme(legend.position = "none"),
                     spec+theme(legend.position = "none"),
                     ppv+theme(legend.position = "none"),layout_matrix = rbind(c(NA,1,1,2,2,NA),c(3,3,4,4,5,5))
)
ggsave(filename = paste0("plots/", "compare_model_types_enevlope",".jpg"),plot = comp,width = 11,height = 7.5)


