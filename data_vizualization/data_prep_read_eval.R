library(ggplot2)
library(gridExtra)
library(tidyverse)

# file paths -----
filesDeepac <- list.files(path = c("/home/uli/Dropbox/HPI_RKI/model_evaluation_subreads/DeePaC/results/lstm",
                                   "/home/uli/Dropbox/HPI_RKI/model_evaluation_subreads/DeePaC/results/cnn"),
                          pattern = ".csv",full.names = TRUE)

# deepac
# read data
deepac <- do.call(rbind,lapply(filesDeepac, function(file){
  data = read.csv2(file,sep =",",dec = ".",as.is = TRUE)
  data$training = str_extract(string = file,pattern = "(cnn_|lstm_).*(?=-metrics)")
  data$seq_tech_train_data = ifelse(str_detect(string = file,pattern = "nanopore"),"nanopore","illumina")
  data <- data[data$epoch!="epoch",]
  return(data)}  ))

# remove duplicates from paired pred
deepac <- deepac[!duplicated(deepac),]

# ensemble data (read seperatly because epoch column is faulty)
files_ens <- list.files(path = "/home/uli/Dropbox/HPI_RKI/model_evaluation_subreads/DeePaC/results/ensemble/",
                        pattern = ".csv",full.names = TRUE)

deepac_ens <- do.call(rbind,lapply(files_ens, function(file){
  data = read.csv2(file,sep =",",dec = ".",as.is = TRUE)
  data$training = str_extract(string = file,pattern = "(cnn_|lstm_).*(?=-metrics)")
  data <- data[data$epoch!="epoch",]
  data$seq_tech_train_data = ifelse(str_detect(string = file,pattern = "nanopore"),"nanopore","illumina")
  data <- data%>%
    group_by(set)%>%
    mutate(epoch = 1:n())%>%
    ungroup()
  return(data)}  ))

# remove duplicates from paired pred
deepac_ens <- deepac_ens[!duplicated(deepac_ens),]

deepac <- rbind(deepac,deepac_ens)

cols_numeric <- colnames(deepac)[3:15]
deepac <- deepac%>%
  mutate_at(cols_numeric,as.numeric)%>%
  # add column indicating if its a paired or single read pred
  mutate(pred_type = ifelse(set %in% c("test_1_test_2",
                                       "test_2_nano_test_3_nano",
                                       "test_2_nano_test_4_nano",
                                       "test_3_nano_test_4_nano"), 
                            "paired_read", "single_read"))

# epochs encode for subread length
levels = 1:22
labels = seq(25,500,25)
labels = append(labels,c("mixed(50-250bp)","avg"))

colnames(deepac)[colnames(deepac) == "epoch"] <- "subread_length"
deepac$subread_length <- factor(deepac$subread_length,levels = levels,labels = labels,ordered = TRUE)


# extract info from training column e.g. training read length, ann type ...
deepac <- deepac%>%
  rowwise()%>%
  mutate(trained_on = str_extract(training,"\\d+-*\\d*bp"),
         trained_on = factor(trained_on,levels = c("50bp","100bp","150bp","200bp","250bp","50-250bp","500bp")),
         subread_pos = ifelse(str_detect(training,"last"),"end","start"),
         type = paste(unlist(str_extract_all(training,"cnn|lstm")),collapse = "+"))


#paprbag
filesPaPrBag <- list.files(path = "/home/uli/Dropbox/HPI_RKI/model_evaluation_subreads/PaPrBag/",pattern = ".csv",full.names = TRUE)


# add paprbag data
paprbag <- do.call(rbind,lapply(filesPaPrBag, function(file){
  data = read.csv2(file,sep =";")
  data$training = str_extract(string = file,pattern = "DNA_\\d+")
  data$training = paste0("PaPrBaG ", data$training)
  data$training = as.factor(data$training)
  data$seq_tech_train_data = ifelse(str_detect(string = file,pattern = "nanopore"),"nanopore","illumina")
  return(data)}  ))
colnames(paprbag) <- tolower(colnames(paprbag))
colnames(paprbag)[1] <- "subread_length"
paprbag$subread_length <- factor(paprbag$subread_length)
colnames(paprbag)[6] <- "recall"
colnames(paprbag)[7] <- "spec"
colnames(paprbag)[8] <- "precision"
colnames(paprbag)[19] <- "auroc"

paprbag$subread_pos <- "start"
paprbag$type <- "random_forest"
paprbag$set <- "test_1"
paprbag$pred_type <- "single_read"


paprbag <- paprbag%>%
  mutate(trained_on = str_extract(training,"\\d+-*\\d*"),
         trained_on = paste0(trained_on,"bp"),
         trained_on = factor(trained_on,levels = c("50bp","100bp","150bp","200bp","250bp","50-250bp")))

# drop cols not in deepac eval
paprbag <- paprbag[,colnames(paprbag) %in% colnames(deepac)]
deepac <- deepac[,colnames(deepac) %in% colnames(paprbag)]


# combine paprbag and deepac data
read_eval_data <- rbind(deepac,paprbag)%>%
  ungroup()

# calc average over all read lengths
read_eval_data_avg_part_1 <- read_eval_data%>%
  group_by(training,set)%>%
  summarise_if(is.numeric,mean)%>%
  mutate(subread_length = "avg")

read_eval_data_avg_part_2 <- read_eval_data%>%
  group_by(training,set)%>%
  mutate(subread_length = "avg")%>%
  summarise_if(function(x){!is.numeric(x)},unique)

read_eval_data_avg <- merge(read_eval_data_avg_part_1,read_eval_data_avg_part_2,by = c("training","set","subread_length"))
read_eval_data <- rbind(read_eval_data,read_eval_data_avg)
rm(read_eval_data_avg,read_eval_data_avg_part_1,read_eval_data_avg_part_2)

# calc average over 50-250bp length (valuies equal to that test set)
read_eval_data_avg_part_1 <- read_eval_data%>%
  group_by(training,set)%>%
  filter(subread_length %in% seq(50,250,25))%>%
  summarise_if(is.numeric,mean)%>%
  mutate(subread_length = "mixed(50-250bp)")

read_eval_data_avg_part_2 <- read_eval_data%>%
  group_by(training,set)%>%
  mutate(subread_length = "mixed(50-250bp)")%>%
  summarise_if(function(x){!is.numeric(x)},unique)

read_eval_data_avg <- merge(read_eval_data_avg_part_1,read_eval_data_avg_part_2,by = c("training","set","subread_length"))
read_eval_data <- rbind(read_eval_data,read_eval_data_avg)
rm(read_eval_data_avg,read_eval_data_avg_part_1,read_eval_data_avg_part_2)

# save as rds
saveRDS(read_eval_data,file = "read_eval_data.rds")

