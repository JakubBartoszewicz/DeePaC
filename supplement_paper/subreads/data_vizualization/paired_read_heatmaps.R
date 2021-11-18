# heatmap -----
library(tidyverse)
library(cowplot)
library(grid)

plot_heatmap <- function(data,metric,limits_scale){
  # plots a heatmap of a metric on different readlength comb 
  #
  # Args:
  #   data: dataframe, must contain col subread_length, subread_length_2 and a metric
  #   metric: e.g. acc, auroc ...
  #   limits_scale: limits for the color scale
  #
  # Returns:
  #   heatmap
  # compute plot
  heatmap <- ggplot() + 
    # no data and mapping defined in the ggplot command 
    # to enable the easy drawing of lines on the plot later on 
    geom_tile(data = data,
              mapping = aes(subread_length, 
                            subread_length_2, 
                            fill = 100*get(metric)) #,show.legend = TRUE
                            ) +
    scale_fill_gradientn(colours = c("red3","red" ,"yellow","green","darkgreen"),
                         values = c(0,0.25,.50,.75,1),
                         limits = limits_scale,
                         name = metric
    ) + xlab("length read 1 [bp]") + ylab("length read 2 [bp]") +
    geom_text(data = data,
              aes(subread_length,
                  subread_length_2,
                  label = sprintf("%2.1f",100 * round(get(metric),3)))) +
    theme_classic() +
    theme(
      axis.line = element_blank(),
      axis.ticks = element_blank(),
      axis.text = element_text(face = "bold",size = 12),
      axis.title = element_text(face = "bold",size = 14),
      legend.position = "none"
    )
  return(heatmap)
}

add_line_heatmap <- function(heatmap,data,metric,threshold,color,linetype,name){
  # draws a threshold line on a heatmap 
  #
  # Args:
  #   heatmap
  #   data: dataframe, must contain col subread_length, subread_length_2 and a metric
  #   metric e.g. acc.....
  #   threshold
  #   color of the line
  #   linetype 
  #   name: legend key to be used
  #
  # Returns:
  #   heatmap with the added line
  # get coordiantes for the threshold line
  data <- data %>%
    arrange(subread_length, subread_length_2) %>%
    group_by(subread_length) %>%
    mutate(borderBottom = ifelse((get(metric) >= threshold & lag(get(metric)) < threshold) |
                                   (get(metric) < threshold & lag(get(metric)) >= threshold),
                                 TRUE,
                                 FALSE)) %>%
    arrange(subread_length_2, subread_length) %>%
    group_by(subread_length_2) %>%
    mutate(borderLeft = ifelse((get(metric) >= threshold & lag(get(metric)) <  threshold) |
                                 (get(metric) <  threshold & lag(get(metric)) >= threshold) ,
                               TRUE,
                               FALSE)) %>%
    ungroup()

  x_borderBottom <- as.integer(data$subread_length[which(data$borderBottom)])
  y_borderBottom <- as.integer(data$subread_length_2[which(data$borderBottom)])
  x_borderLeft <- as.integer(data$subread_length[which(data$borderLeft)])
  y_borderLeft <- as.integer(data$subread_length_2[which(data$borderLeft)])


  # draw lines
  if(length(x_borderBottom) > 0 ){
    heatmap <- heatmap +
      geom_segment(aes(
        x = x_borderBottom - 0.5,
        xend = x_borderBottom + 0.5,
        y = y_borderBottom - 0.5,
        yend = y_borderBottom - 0.5,
        color = name,
        linetype = name),
        show.legend = TRUE
      )
  }

  if(length(x_borderLeft) > 0){
    heatmap <- heatmap +
      geom_segment(aes(
        x = x_borderLeft - 0.5,
        xend = x_borderLeft - 0.5,
        y = y_borderLeft - 0.5,
        yend = y_borderLeft + 0.5,
        color = name,
        linetype = name),
        show.legend = TRUE
      )
  }
  # add color and linetype
  heatmap <- heatmap +
    scale_color_manual(name="thresholds",values = color) +
    scale_linetype_manual(name="thresholds",values = linetype)
  
  return(heatmap)
}

read_eval_data <- readRDS(file = "read_eval_data_merged.rds")

trainings <- c("cnn_250bp_d025_img",
               "res18-25-250bp-img-d25-corr-e021.h5",
               "cnn_250bp_d025_vhdb_all-e011",
               "res18-25-250bp-vhdb-d25-e026.h5")

metric <- "acc"
for(train in trainings){
  read_eval_data_plot <- read_eval_data%>%
    filter(training == train,
           set %in% c("test_1_test_2","all_test_1_all_test_2"))%>%
    mutate_if(is.numeric, "round", digits = 3)
  
  heatmap <- plot_heatmap(read_eval_data_plot,metric,c(50,91))
  
  heatmap <- add_line_heatmap(heatmap = heatmap,threshold = 0.8,data = read_eval_data_plot,
                               color = "black",linetype = "solid",name = "80%",metric = metric)
  
  baseline <- read_eval_data%>%
    filter(training == train,
           set %in% c("test_1","all_test_1"),
           subread_length == 250)
  
  baseline <- as.numeric(baseline[,metric])
  
  # when adding a second line color and linetype of the previous defined line need to be defined again
  # => function should be optimized here
  heatmap <- add_line_heatmap(heatmap = heatmap,threshold = baseline,data = read_eval_data_plot,
                               color = c("black","red"),linetype = c("solid","dashed"),name = "baseline",metric = metric)
  
  ggsave(filename = paste0("plots/", train, "_",metric,".png"),
         plot = heatmap, width = 5, height = 5)
  ggsave(filename = paste0("plots/", train, "_",metric,".eps"),
         plot = heatmap, width = 5, height = 5, device="eps")
}

# legend <- cowplot::get_legend(heatmap)
# grid.newpage()
# grid.draw(legend)
# ggsave(filename = paste0("plots/", "legend_heatmap_color.png"), plot = legend, width = 0.75, height = 2)
# ggsave(filename = paste0("plots/", "legend_heatmap_color.eps"), plot = legend, width = 0.75, height = 2, device = "eps")