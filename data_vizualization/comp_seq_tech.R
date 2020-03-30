library(ggplot2)
library(gridExtra)
library(tidyverse)

# data preparation
read_eval_data <- readRDS("read_eval_data.rds")
test_set = "test_1"
read_eval_data_plot <- read_eval_data %>%
  filter(
    set == test_set,
    type %in% c("cnn", "lstm"),!subread_length %in%  c("avg", "mixed(50-250bp)"),
    trained_on == "250bp",
    pred_type == "single_read"
  )

plot_compare_seq_tech <- function(metric,ylab, ylim,title = "",table = FALSE) {
    # plots the course of a metric on different subreadlengths (y=metric,x=read length) 
    # comparing the classifiers trained with data from varying seq technology
    #
    # Args:
    #   metric: e.g. acc, auroc ...
    #   ylab: label y axis 
    #   ylim: limits y axis as vector
    #   title of the plot
    #   table: show table (Bool)
    #
    # Returns:
    #   plot and opt. table
    # compute plot
    plot <- ggplot(
      data = read_eval_data_plot,
      mapping = aes(
        subread_length,
        get(metric),
        color = seq_tech_train_data,
        group = training,
        shape = type
      )
    ) + xlab("subread length [bp]") + 
      geom_point(size = 4) + geom_line(size = 0.8) +
      theme_classic() + theme(
        axis.text = element_text(face = "bold"),
        panel.grid.major = element_line(),
        legend.position = "top"
      ) +
      scale_y_continuous(name = ylab, limits = ylim) + ggtitle(title)
    
    if (table) {
      data_temp <-
        read_eval_data_plot[, c("subread_length", metric, "type", "seq_tech_train_data")]
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

acc <-
  plot_compare_seq_tech(
    "acc",
    "accuracy",
    c(0.4, 1),
    title = paste("performance", test_set),
    table = TRUE
  )
acc
auroc <-
  plot_compare_seq_tech(
    "auroc",
    "auroc",
    c(0.4, 1),
    title = paste("performance", test_set),
    table = TRUE
  )
auroc
precision <-
  plot_compare_seq_tech(
    "precision",
    "precision",
    c(0.4, 1),
    title = paste("performance", test_set),
    table = TRUE
  )
precision
recall <-
  plot_compare_seq_tech(
    "recall",
    "recall",
    c(0.3, 1),
    title = paste("performance", test_set),
    table = TRUE
  )
recall
spec <-
  plot_compare_seq_tech("spec",
                              "spec",
                              c(0, 1),
                              title = paste("performance", test_set),
                              table = TRUE)
spec



#save results-----
for (obj in ls()) {
  plot = get(obj)
  if ("ggplot" %in% class(plot)) {
    ggsave(
      filename = paste0("~/Dropbox/HPI_RKI/DeePaC/data_vizualization/plots/compare_seq_tech/nanopore_vs_illumina_",obj,"_",test_set,".jpg"),
      plot = plot,
      width = 6,
      height = 4
    )
  } else if ("grob" %in% class(plot)) {
    ggsave(
      filename = paste0(
        "~/Dropbox/HPI_RKI/DeePaC/data_vizualization/plots/compare_seq_tech/nanopore_vs_illumina_", obj,"_",test_set,".jpg"),
      plot = plot,
      width = 9,
      height = 8
    )
  }
}
