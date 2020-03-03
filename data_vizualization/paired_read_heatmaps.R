#heatmaps

#heatmap -----

library(tidyverse)

filesDeepac <- c(
  #"../deepac_analysis/eval/results_25-250bp_reads_cnn/builtinConfig/cnn_trained_on_50bp_reads/e007/paired/nn-fullcnn-metrics.csv",
  "../deepac_analysis/eval/results_25-250bp_reads_cnn/builtinConfig/cnn_trained_on_50bp_reads/e008/paired/nn-fullcnn-metrics.csv",
  "../deepac_analysis/eval/results_25-250bp_reads_cnn/builtinConfig/cnn_trained_on_100bp_reads/e001/paired/nn-fullcnn-metrics.csv",
  "../deepac_analysis/eval/results_25-250bp_reads_cnn/builtinConfig/cnn_trained_on_150bp_reads/e003/paired/nn-fullcnn-metrics.csv",
  "../deepac_analysis/eval/results_25-250bp_reads_cnn/builtinConfig/cnn_trained_on_150bp_reads_2/e003/paired/nn-fullcnn-metrics.csv",
  "../deepac_analysis/eval/results_25-250bp_reads_cnn/builtinConfig/cnn_trained_on_200bp_reads/e003/paired/nn-fullcnn-metrics.csv",
  "../deepac_analysis/eval/results_25-250bp_reads_cnn/builtinConfig/cnn_trained_on_250bp_reads/e000/paired/nn-fullcnn-metrics.csv",
  "../deepac_analysis/eval/results_25-250bp_reads_cnn/builtinConfig/cnn_trained_on_50-250bp_reads/e003/paired/nn-fullcnn-metrics.csv",
  
  "../deepac_analysis/eval/results_25-250bp_reads_lstm/builtinConfig/lstm_trained_on_50bp_reads/e007/paired/nn-fullcnn-metrics.csv",
  "../deepac_analysis/eval/results_25-250bp_reads_lstm/builtinConfig/lstm_trained_on_100bp_reads_3/e004/paired/nn-fullcnn-metrics.csv",
  "../deepac_analysis/eval/results_25-250bp_reads_lstm/builtinConfig/lstm_trained_on_150bp_reads/e009/paired/nn-fullcnn-metrics.csv",
  "../deepac_analysis/eval/results_25-250bp_reads_lstm/builtinConfig/lstm_trained_on_200bp_reads/e009/paired/nn-fullcnn-metrics.csv",
  "../deepac_analysis/eval/results_25-250bp_reads_lstm/builtinConfig/lstm_trained_on_250bp_reads/e000/paired/nn-fullcnn-metrics.csv",
  "../deepac_analysis/eval/results_25-250bp_reads_lstm/builtinConfig/lstm_trained_on_50-250bp_reads/e006/paired/nn-fullcnn-metrics.csv"
)

plotTtile <- c(
  "CNN trained with 50bp subreads",
  "CNN trained with 100bp subreads",
  "CNN trained with 150bp subreads",
  "CNN trained with 150bp subreads",
  "CNN trained with 200bp subreads",
  "CNN trained with 250bp subreads",
  "CNN trained with 50-250bp subreads",
  
  "LSTM trained with 50bp subreads",
  "LSTM trained with 100bp subreads",
  "LSTM trained with 150bp subreads",
  "LSTM trained with 200bp subreads",
  "LSTM trained with 250bp subreads",
  "LSTM trained with 50-250bp subreads"
)

addLineHeatMap <- function(heatmap,threshold,data,color,linetype,name){
  # get coordiantes
  data <- data %>%
    arrange(x, y) %>%
    group_by(x) %>%
    mutate(borderBottom = ifelse((acc >= threshold & lag(acc) < threshold) |
                                   (acc < threshold & lag(acc) >= threshold), 
                                 TRUE, 
                                 FALSE)) %>%
    arrange(y, x) %>%
    group_by(y) %>%
    mutate(borderLeft = ifelse((acc >= threshold & lag(acc) <  threshold) | 
                                 (acc <  threshold & lag(acc) >= threshold) ,
                               TRUE, 
                               FALSE)) %>%
    ungroup()
  
  
  x_borderBottom <- data$x[which(data$borderBottom)]
  y_borderBottom <- data$y[which(data$borderBottom)]
  x_borderLeft <- data$x[which(data$borderLeft)]
  y_borderLeft <- data$y[which(data$borderLeft)]
  
  
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
  
  return(heatmap)
}

j = 1

for (file in filesDeepac) {
  pairedPredictions <- read.csv2(file, sep = ",", dec = ".")
  pairedPredictions$type = str_extract(string = file, pattern = "[^/]*(cnn_|lstm_)[^/]*")
  pairedPredictions$e = str_extract(string = file, pattern = "e\\d{3}")
  pairedPredictions$type = as.factor(paste0(pairedPredictions$type, "_", pairedPredictions$e))
  
  
  pairedPredictions <- pairedPredictions %>%
    filter(set == "test_1_test_2") %>%
    mutate(read_1_length = sort(rep(seq(25, 250, 25), 10)),
           read_2_length = rep(seq(25, 250, 25), 10)) %>%
    mutate(x = rep(1:10, 10),
           y = sort(rep(1:10, 10))) %>%
    mutate_if(is.numeric, "round", digits = 3)
  
  readLengths = paste0(seq(25, 250, 25), "bp")
  
  accHeatMap <-
    ggplot() + 
    geom_tile(data = pairedPredictions,
              aes(x, y, fill = acc)) +
    scale_x_continuous(breaks = c(1:10),
                       labels = readLengths,
                       name = "length read 2") +
    scale_y_continuous(breaks = c(1:10),
                       labels = readLengths,
                       name = "length read 1") +
    scale_fill_gradientn(colours = c("darkred","red" ,"yellow","green","darkgreen"),values = c(0,0.25,0.5,0.75,1),
      limits = c(0.5, 0.9),
      name = "Accuracy"
    ) +
    geom_text(data = pairedPredictions,
              aes(x,y, label = acc)) + 
    ggtitle(unique(pairedPredictions$type)) +
    theme_classic() +
    theme(
      axis.line = element_blank(),
      axis.ticks = element_blank(),
      axis.ticks.length = unit(-.5, "cm"),
      axis.text = element_text(face = "bold")
    ) +
    ggtitle(plotTtile[j]) 

  if(str_extract(file,"lstm|cnn") == "lstm"){
    threshold <- 0.838
  }else{
    threshold <- 0.844
  }
  
  accHeatMap <- addLineHeatMap(heatmap = accHeatMap,threshold = 0.8,data = pairedPredictions,
                               color = "black",linetype = "solid",name = "80%-treshold")
  accHeatMap <- addLineHeatMap(heatmap = accHeatMap,threshold = threshold,data = pairedPredictions,
                               color = "red",linetype = "dashed",name = "baseline threshold")
  
  accHeatMap <- accHeatMap + 
    scale_color_manual(name="thresholds",values = c("black",c("red"))) + 
    scale_linetype_manual(name="thresholds",values = c(1,2))
  
  j = j + 1
  
  
  ggsave(filename = paste0("../results/", unique(pairedPredictions$type), "_acc.jpg"),
         plot = accHeatMap)
  
}


# diff paired vs single read --------

library(tidyverse)

filesDeepac <- c(
  #"../deepac_analysis/eval/results_25-250bp_reads_cnn/builtinConfig/cnn_trained_on_50bp_reads/e007/paired/nn-fullcnn-metrics.csv",
  "../deepac_analysis/eval/results_25-250bp_reads_cnn/builtinConfig/cnn_trained_on_50bp_reads/e008/paired/nn-fullcnn-metrics.csv",
  "../deepac_analysis/eval/results_25-250bp_reads_cnn/builtinConfig/cnn_trained_on_100bp_reads/e001/paired/nn-fullcnn-metrics.csv",
  "../deepac_analysis/eval/results_25-250bp_reads_cnn/builtinConfig/cnn_trained_on_150bp_reads/e003/paired/nn-fullcnn-metrics.csv",
  "../deepac_analysis/eval/results_25-250bp_reads_cnn/builtinConfig/cnn_trained_on_150bp_reads_2/e003/paired/nn-fullcnn-metrics.csv",
  "../deepac_analysis/eval/results_25-250bp_reads_cnn/builtinConfig/cnn_trained_on_200bp_reads/e003/paired/nn-fullcnn-metrics.csv",
  "../deepac_analysis/eval/results_25-250bp_reads_cnn/builtinConfig/cnn_trained_on_250bp_reads/e000/paired/nn-fullcnn-metrics.csv",
  "../deepac_analysis/eval/results_25-250bp_reads_cnn/builtinConfig/cnn_trained_on_50-250bp_reads/e003/paired/nn-fullcnn-metrics.csv",
  
  "../deepac_analysis/eval/results_25-250bp_reads_lstm/builtinConfig/lstm_trained_on_50bp_reads/e007/paired/nn-fullcnn-metrics.csv",
  "../deepac_analysis/eval/results_25-250bp_reads_lstm/builtinConfig/lstm_trained_on_100bp_reads_3/e004/paired/nn-fullcnn-metrics.csv",
  "../deepac_analysis/eval/results_25-250bp_reads_lstm/builtinConfig/lstm_trained_on_150bp_reads/e009/paired/nn-fullcnn-metrics.csv",
  "../deepac_analysis/eval/results_25-250bp_reads_lstm/builtinConfig/lstm_trained_on_200bp_reads/e009/paired/nn-fullcnn-metrics.csv",
  "../deepac_analysis/eval/results_25-250bp_reads_lstm/builtinConfig/lstm_trained_on_250bp_reads/e000/paired/nn-fullcnn-metrics.csv",
  "../deepac_analysis/eval/results_25-250bp_reads_lstm/builtinConfig/lstm_trained_on_50-250bp_reads/e006/paired/nn-fullcnn-metrics.csv"
)

plotTtile <- c(
  "CNN trained with 50bp subreads",
  "CNN trained with 100bp subreads",
  "CNN trained with 150bp subreads",
  "CNN trained with 150bp subreads",
  "CNN trained with 200bp subreads",
  "CNN trained with 250bp subreads",
  "CNN trained with 50-250bp subreads",
  
  "LSTM trained with 50bp subreads",
  "LSTM trained with 100bp subreads",
  "LSTM trained with 150bp subreads",
  "LSTM trained with 200bp subreads",
  "LSTM trained with 250bp subreads",
  "LSTM trained with 50-250bp subreads"
)

j = 1

for (file in filesDeepac) {
  pairedPredictions <- read.csv2(file, sep = ",", dec = ".")
  pairedPredictions$type = str_extract(string = file, pattern = "[^/]*(cnn_|lstm_)[^/]*")
  pairedPredictions$e = str_extract(string = file, pattern = "e\\d{3}")
  pairedPredictions$type = as.factor(paste0(pairedPredictions$type, "_", pairedPredictions$e))
  
  pairedPredictions$acc[pairedPredictions$set == "test_1_test_2"] = pairedPredictions$acc[pairedPredictions$set ==
                                                                                            "test_1_test_2"] -
    pairedPredictions$acc[pairedPredictions$set == "test_1"]
  
  pairedPredictions <- pairedPredictions %>%
    filter(set == "test_1_test_2") %>%
    mutate(read_1_length = sort(rep(seq(25, 250, 25), 10)),
           read_2_length = rep(seq(25, 250, 25), 10)) %>%
    mutate(x = rep(1:10, 10),
           y = sort(rep(1:10, 10))) %>%
    mutate_if(is.numeric, "round", digits = 3)
  
  
  pairedPredictions <- pairedPredictions %>%
    arrange(x, y) %>%
    group_by(x) %>%
    mutate(borderBottom = ifelse((acc >= 0.8 &
                                    lag(acc) < 0.8) | (acc < 0.8 & lag(acc) >= 0.8), TRUE, FALSE)) %>%
    arrange(y, x) %>%
    group_by(y) %>%
    mutate(borderLeft = ifelse((acc >= 0.8 &
                                  lag(acc) < 0.8) | (acc < 0.8 & lag(acc) >= 0.8) , TRUE, FALSE)) %>%
    ungroup()
  
  
  x_borderBottom <-
    pairedPredictions$x[which(pairedPredictions$borderBottom)]
  y_borderBottom <-
    pairedPredictions$y[which(pairedPredictions$borderBottom)]
  
  x_borderLeft <-
    pairedPredictions$x[which(pairedPredictions$borderLeft)]
  y_borderLeft <-
    pairedPredictions$y[which(pairedPredictions$borderLeft)]
  
  
  readLengths = paste0(seq(25, 250, 25), "bp")
  
  accHeatMap <-
    ggplot(pairedPredictions, aes(x, y, fill = acc, label = acc)) + geom_tile() +
    scale_x_continuous(breaks = c(1:10),
                       labels = readLengths,
                       name = "length read 2") +
    scale_y_continuous(breaks = c(1:10),
                       labels = readLengths,
                       name = "length read 1") +
    scale_fill_gradientn(colours = c("red","yellow","green","darkgreen"),values = c(0,0.2,0.4,1),
                         limits = c(-0.1, 0.35),
      name = "Diff in Accuracy"
    ) +
    geom_text() + ggtitle(unique(pairedPredictions$type)) +
    theme_classic() +
    theme(
      axis.line = element_blank(),
      axis.ticks = element_blank(),
      axis.ticks.length = unit(-.5, "cm"),
      axis.text = element_text(face = "bold")
    ) +
    ggtitle(plotTtile[j])
  
  if (length(x_borderBottom) > 0) {
    for (i in 1:length(x_borderBottom)) {
      accHeatMap <- accHeatMap +
        geom_segment(
          x = x_borderBottom[i] - 0.5,
          xend = x_borderBottom[i] + 0.5,
          y = y_borderBottom[i] - 0.5,
          yend = y_borderBottom[i] - 0.5,
          show.legend = TRUE
        )
      
    }
    
    
    for (i in 1:length(x_borderLeft)) {
      accHeatMap <- accHeatMap +
        geom_segment(
          x = x_borderLeft[i] - 0.5,
          xend = x_borderLeft[i] - 0.5,
          y = y_borderLeft[i] - 0.5,
          yend = y_borderLeft[i] + 0.5
        )
      
    }
    
  }
  
  
  j = j + 1
  
  
  ggsave(filename = paste0("../results/", unique(pairedPredictions$type), "_acc.jpg"),
         plot = accHeatMap)
  
}
    