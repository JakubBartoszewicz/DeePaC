library(ggplot2)
library(gridExtra)
library(tidyverse)
library(stringr)
library(gridExtra)

# load data
files = c("../deepac_analysis/pairedRead_effect/results/cnn_50bp.csv",
          "../deepac_analysis/pairedRead_effect/results/cnn_100bp.csv",
          "../deepac_analysis/pairedRead_effect/results/cnn_150bp.csv",
          "../deepac_analysis/pairedRead_effect/results/cnn_200bp.csv",
          "../deepac_analysis/pairedRead_effect/results/cnn_250bp.csv",
          "../deepac_analysis/pairedRead_effect/results/lstm_50bp.csv",
          "../deepac_analysis/pairedRead_effect/results/lstm_100bp.csv",
          "../deepac_analysis/pairedRead_effect/results/lstm_150bp.csv",
          "../deepac_analysis/pairedRead_effect/results/lstm_200bp.csv",
          "../deepac_analysis/pairedRead_effect/results/lstm_250bp.csv",
          "../deepac_analysis/pairedRead_effect/results/cnn_random_mixed_250bp.csv",
          "../deepac_analysis/pairedRead_effect/results/lstm_random_mixed_250bp.csv"
          )

pairedRead_data <- do.call(what = rbind, lapply(files, function(file){data <- read.csv(file,colClasses="character")
                                                                      data$filename <- str_extract(string=file,pattern="\\w*(?=.csv)")
                                                                      return(data)}  ))

pairedRead_data <- pairedRead_data[pairedRead_data$epoch!="epoch",]
num_cols = colnames(pairedRead_data)[!colnames(pairedRead_data) %in% c("set", "filename")]
  pairedRead_data <- pairedRead_data%>%
  mutate_at(.vars = num_cols,.funs = as.numeric)

pairedRead_data$fragment_length <- str_extract(pairedRead_data$set,"\\d*$")
pairedRead_data$fragment_length <-  as.numeric(pairedRead_data$fragment_length)
fragment_lengths <- unique(sort(pairedRead_data$fragment_length))


pairedRead_data <- pairedRead_data%>%
    group_by(epoch,filename)%>%
    mutate(acc_diff = acc[3]-acc[1],
           auroc_diff = auroc[3]-auroc[1],
           recall_diff = recall[3]-recall[1],
           precision_diff = precision[3]-precision[1],
           spec_diff = spec[3]-spec[1],
           read_pair = ifelse(str_count(set,"test") > 1,
                              yes = "both",
                              no = str_extract(set,"\\d")),
           read_length = str_extract(filename,"\\d*bp$"),
           type = str_extract(filename,"^\\w*(?=_)")
           )

# diff in performane
plotDesign3 <- function(ylab,y,nn_type,table=FALSE, ylim=c(NA,1), read_lengths=c("50bp","100bp","150bp","200bp","250bp")){
  data_temp <- pairedRead_data%>%
    filter(read_length%in%read_lengths)%>%
    filter(type==nn_type)%>%
    mutate(read_length = factor(read_length,levels = c("50bp","100bp","150bp","200bp","250bp")))
  
  data_temp_paired <- data_temp%>%
    filter(read_pair=="both")
  
  data_temp_single <- data_temp%>%
    filter(read_pair!="both")
  
plot <- ggplot(data = data_temp_paired,
         mapping = aes(x = log(fragment_length),
                       y=get(y), 
                       group=paste(filename,read_pair),
                       color=read_length,
                       shape=read_length)) + 
    geom_point(size = 4,shape = 7,color = "limegreen") + geom_line(size = 2,color = "limegreen") +
    scale_x_continuous(breaks = log(fragment_lengths),
                       labels = fragment_lengths) +
    xlab("mean fragment length (bp)") + 
    ylab(ylab) + ylim(ylim) +
    #single read ref
    geom_point(data =  data_temp_single,
               mapping = aes(x = log(fragment_length),y=get(y), 
                             group=paste(filename,read_pair)),color="grey",show.legend = FALSE,shape = 6) +
    geom_line(data =  data_temp_single,
             mapping = aes(x = log(fragment_length),y=get(y), 
                           group=paste(filename,read_pair)),color="grey", linetype =2,show.legend = FALSE) + 
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45,vjust = 0.5),
        axis.text = element_text(face = "bold",size = 14),
        axis.title  = element_text(face = "bold",size = 16),
        panel.grid.major = element_line(),
        legend.position = "top") +
  scale_color_brewer(palette = "Dark2")


data_temp <- data_temp[,c("fragment_length",y,"read_length","read_pair")]
colnames(data_temp) <- c(colnames(data_temp)[1],"metric",colnames(data_temp)[c(3,4)])

data_temp <- data_temp%>%
  ungroup()%>%
  mutate(metric=round(metric,digits = 3))%>%
  spread(fragment_length,metric)

if(table){
  table_1 <- tableGrob(data_temp[,c(1,2,3:(round(length(data_temp)/2)))],rows = NULL,theme = ttheme_minimal(base_size = 16)) 
  table_2 <- tableGrob(data_temp[,c(1,2,(round(length(data_temp) /2)+1):length(data_temp))],rows = NULL,theme = ttheme_minimal(base_size = 16)) 
  
  figure <- grid.arrange(plot,table_1,table_2,layout_matrix=rbind(c(1,1),
                                                                  c(1,1),
                                                                  c(2,2),
                                                                  c(3,3)))
} else {
  return(plot)
}

}
cnn_overview <- plotDesign3(ylab = "accuracy",y = "acc",ylim = c(0.5,0.9),table = FALSE,nn_type = "cnn") 
lstm_overview <- plotDesign3(ylab = "accuracy",y = "acc",ylim = c(0.5,0.9),table = FALSE,nn_type = "lstm") 

ggsave(filename = "../results/paired_read_performance/cnn_overview.jpg",plot = cnn_overview,width = 8, height = 8)
ggsave(filename = "../results/paired_read_performance/lstm_overview.jpg",plot = lstm_overview,width = 8, height = 8)

plot <- plotDesign3(ylab = "acc",y = "acc",ylim = c(0.75,1),table = TRUE,nn_type = "lstm_random_mixed",read_lengths = "250bp") 
ggsave(filename = paste0("../results/paired_read_performance/","lstm250bp_precision.jpeg"),
         plot = plot,
         width = 11, 
         height = 7)





for(type in c("cnn","lstm")){
    for(read_length in c("50bp","100bp","150bp","200bp","250bp")){
      plot <- plotDesign3(ylab = "accuracy",y = "acc",read_lengths = read_length,ylim = c(0.5,0.9),table = TRUE,nn_type = type)
      ggsave(filename = paste0("../results/paired_read_performance/",type,read_length,"_acc.jpeg"),plot = plot,width = 11, height = 12)
    }
}

for(type in c("cnn","lstm")){
  for(read_length in c("50bp","100bp","150bp","200bp","250bp")){
    plot <- plotDesign3(ylab = "auroc",y = "auroc",read_lengths = read_length,ylim = c(0.6,1),table = TRUE,nn_type = type)
    ggsave(filename = paste0("../results/paired_read_performance/",type,read_length,"_auroc.jpeg"),plot = plot,width = 11, height = 12)
  }
}
  
for(type in c("cnn","lstm")){
  for(read_length in c("50bp","100bp","150bp","200bp","250bp")){
    plot <- plotDesign3(ylab = "precision",y = "precision",read_lengths = read_length,ylim = c(0.5,1),table = TRUE,nn_type = type)
    ggsave(filename = paste0("../results/paired_read_performance/",type,read_length,"_precision.jpeg"),plot = plot,width = 11, height = 12)
    }
}
  
for(type in c("cnn","lstm")){
  for(read_length in c("50bp","100bp","150bp","200bp","250bp")){
    plot <- plotDesign3(ylab = "recall",y = "recall",read_lengths = read_length,ylim = c(0.5,1),table = TRUE,nn_type = type)
    ggsave(filename = paste0("../results/paired_read_performance/",type,read_length,"_recall.jpeg"),plot = plot,width = 11, height = 12)
  }
}

for(type in c("cnn","lstm")){
  for(read_length in c("50bp","100bp","150bp","200bp","250bp")){
    plot <- plotDesign3(ylab = "spec",y = "spec",read_lengths = read_length,ylim = c(0,1),table = TRUE,nn_type = type)
    ggsave(filename = paste0("../results/paired_read_performance/",type,read_length,"_spec.jpeg"),plot = plot,width = 11, height = 12)
  }
}
