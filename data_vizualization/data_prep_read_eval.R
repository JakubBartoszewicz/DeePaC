library(tidyverse)
# this script prepares the data from the evaluation of the different tools (DeePaC, PaPrBaG) for plotting 
# all the data is combined into one dataframe to simplify plotting and data analysis


# functions ----
rbind_own <- function(data_1,data_2){
  # rbind dataframes with different number of columns
  # add columns present in data_1 to data_2 
  # remove additional columns from data_2 
  #
  # Args:
  #   data_1: dataframe
  #   data_2: dataframe
  #
  # Returns:
  #   combination of data_1 and data_2
  for(colname in colnames(data_1)){
    if(!colname %in% colnames(data_2)){
      data_2[,colname] = NA
    }
  }
  for(colname in colnames(data_2)){
    if(!colname %in% colnames(data_1)){
      data_2 <- data_2%>%
        select(-all_of(colname))
    }
  }
  return(rbind(data_1,data_2))
}

read_deepac <- function(file){
  # read csv files created with deepac eval function
  #
  # Args:
  #   file: path to csv file
  #
  # Returns:
  #   dataframe
  data = read.csv2(file,sep =",",dec = ".",as.is = TRUE)
  if("epoch" %in% colnames(data)){
    data <- data[data$epoch!="epoch",]
  }
  cols_numeric <- colnames(data)[3:15]
  data <- data%>%mutate_at(cols_numeric,as.numeric)
  data$training = str_extract(string = file,pattern = "[^/\\\\]+(?=.csv$)")
  data$training = str_replace(data$training,"-metrics","")
  data$seq_tech_train_data = ifelse(str_detect(string = file,pattern = "nano"),"nanopore","illumina")
  data$pathogen_type = str_extract(file,"vhdb|img|aureus")
  data$dropout_rate = str_extract(file,"(?<=d)\\d{2,3}")
  return(data)}

# deepac----
# file paths
files_deepac <- list.files(path = c("/home/uli/Dropbox/HPI_RKI/DeePaC/model_evaluation_subreads/DeePaC/results"),
                          pattern = ".csv",full.names = TRUE)
# read data
deepac <- do.call(rbind,lapply(files_deepac,read_deepac))

# ensemble classifier data
files_ens <- list.files(path = "/home/uli/Dropbox/HPI_RKI/DeePaC/model_evaluation_subreads/DeePaC/results/ensemble",
                        pattern = ".csv",full.names = TRUE)

deepac_ens <- do.call(rbind,lapply(files_ens, read_deepac))

deepac_ens <- deepac_ens%>%
    group_by(set,training)%>%
    mutate(epoch = 1:n())%>%
    ungroup()

read_eval_data <- rbind(deepac,deepac_ens)

deepac <- deepac%>%
  # add column indicating if its a paired or single read pred
  mutate(pred_type = ifelse(set %in% c("test_1_test_2",
                                       "test_2_nano_test_3_nano",
                                       "test_2_nano_test_4_nano",
                                       "test_3_nano_test_4_nano"), 
                            "paired_read", "single_read"))

# epochs encode for subread length
levels = 1:21
labels = seq(25,500,25)
labels = append(labels,c("avg"))
colnames(deepac)[colnames(deepac) == "epoch"] <- "subread_length"
deepac$subread_length <- factor(deepac$subread_length,levels = levels,labels = labels,ordered = TRUE)
deepac$subread_length_2 <- factor(NA,levels = levels,labels = labels,ordered = TRUE)
# all paired pred already present are synchronous
deepac$subread_length_2[deepac$pred_type=="paired_read"] = deepac$subread_length[deepac$pred_type=="paired_read"]


# add asynchronous paired read data
files_deepac_paired_as <- list.files("/home/uli/Dropbox/HPI_RKI/DeePaC/model_evaluation_subreads/DeePaC/results/paired_asynchron/",".csv",full.names = T,recursive = T) 
deepac_paired_as <- do.call(rbind,lapply(files_deepac_paired_as,read_deepac))
deepac_paired_as <- deepac_paired_as%>%
  filter(set %in% c("test_1_test_2","test_2","all_test_1_all_test_2","all_test_2"))%>%
  group_by(training)%>%
  mutate(subread_length = sort(rep(seq(25,250,25),20)),
         subread_length = as.factor(subread_length),
         subread_length_2 = rep(c(NA,25,NA,50,NA,75,NA,100,NA,125,NA,150,NA,175,NA,200,NA,225,NA,250),10),
         subread_length_2 = as.factor(subread_length_2))%>%
  select(!epoch)%>%
  ungroup()

deepac_paired_as$pred_type=c("single_read","paired_read")

read_eval_data <- rbind(read_eval_data,deepac_paired_as)

# remove duplicates from paired pred and ensemble pred
deepac <- deepac[!duplicated(deepac[c(1:6,8:21)]),]

# extract info from training column e.g. training read length, ann type ...
deepac <- deepac%>%
  rowwise()%>%
  mutate(trained_on = paste(unlist(str_extract_all(training,"\\d+-*\\d*bp")),collapse = "+"),
         subread_pos = ifelse(str_detect(training,"last"),"end","start"),
         type = paste(unlist(str_extract_all(training,"cnn|lstm")),collapse = "+"))


# paprbag-----
filesPaPrBag <- list.files(path = "/home/uli/Dropbox/HPI_RKI/DeePaC/model_evaluation_subreads/PaPrBag/",pattern = ".csv",full.names = TRUE,recursive = T)

# add paprbag data
paprbag <- do.call(rbind,lapply(filesPaPrBag, function(file){
  data = read.csv2(file,sep =";")
  data$training = str_extract(string = file,pattern = "DNA_\\d+-*\\d+")
  data$training = paste0("PaPrBaG ", data$training)
  data$training = as.factor(data$training)
  data$seq_tech_train_data = ifelse(str_detect(string = file,pattern = "nanopore"),"nanopore","illumina")
  data$pathogen_type = str_extract(file,"vhdb|img")
  data$set <- str_extract(file,"test_*\\d*")
  return(data)}  ))
colnames(paprbag) <- tolower(colnames(paprbag))
colnames(paprbag)[1] <- "subread_length"
paprbag$subread_length <- factor(paprbag$subread_length)
colnames(paprbag)[6] <- "recall"
colnames(paprbag)[7] <- "spec"
colnames(paprbag)[8] <- "precision"
colnames(paprbag)[19] <- "auroc"

paprbag$pred_type <- "single_read"
paprbag$pred_type[paprbag$set=="test"] <- "paired_read" 
paprbag$set[paprbag$set=="test"] = "test_1_test_2"
paprbag$subread_length_2 <- NA
paprbag$subread_length_2[paprbag$set == "test_1_test_2"] = seq(25,250,25)
paprbag$subread_length[paprbag$set == "test_1_test_2"] = 250
paprbag$subread_pos <- "start"
paprbag$type <- "random_forest"
paprbag$dropout_rate  <- NA

paprbag <- paprbag%>%
  mutate(trained_on = str_extract(training,"\\d+-*\\d*"),
         trained_on = paste0(trained_on,"bp"))


# combine paprbag and deepac data
read_eval_data <- rbind_own(deepac,paprbag)%>%
  ungroup()

# blast----
files_blast <- list.files(path = "/home/uli/Dropbox/HPI_RKI/DeePaC/model_evaluation_subreads/Blast/",pattern = ".csv",full.names = T,recursive = T)

blast <- do.call(rbind,lapply(files_blast, function(file){
  data <- read.csv2(file,sep = ";",as.is = T)
  if(str_detect(file,"ext")){
    data$subread_length <- c(250,str_extract(file,"\\d+"),250)
    data$subread_length_2 <-c(NA,NA,str_extract(file,"\\d+"))
  }else{
    data$subread_length <- rep(str_extract(file,"\\d+"))  
    data$subread_length_2 <- c(NA,NA,str_extract(file,"\\d+"))  }  
  data$training <- str_extract(file,"Blast_Test\\w{1}")
  data$type <- "blast"
  data$pathogen_type <- str_extract(file,"vhdb|img")
  colnames(data) <- tolower(colnames(data))
  return(data)
  }  ))

blast <- blast%>%
  rename(set="x")

blast$set[blast$set=="test.L"] = "test_1"
blast$set[blast$set=="test.R"] = "test_2"
blast$set[blast$set=="test"] = "test_1_test_2"
blast$subread_pos <- "start"
blast$pred_type <- "single_read"
blast$pred_type[blast$set=="test_1_test_2"] <- "paired_read"
blast <- blast%>%
  rename(spec = "tnr",
         recall = "tpr",
         precision = "ppv")%>%
  mutate(acc = total.acc,
         recall = total.tpr,
         spec = total.tnr,
         precision= total.ppv)

blast <- blast[!duplicated(blast),]

read_eval_data <- rbind_own(read_eval_data,blast)

# hilive-----
# files have been formatted by hand/using a script (hilive parser)
hilive_files <- list.files("/home/uli/Dropbox/HPI_RKI/DeePaC/model_evaluation_subreads/HiLive/",pattern = ".csv",full.names = T,recursive = T)

hilive <- do.call(rbind,lapply(hilive_files, read_deepac))

colnames(hilive) <- tolower(colnames(hilive))

hilive <- hilive%>%
  select(-acc)%>%
  rename(acc = "acc.all",
         recall = "tpr.all",
         precision= "ppv.all",
         spec="tnr.all")

hilive <- hilive%>%
  mutate_at(colnames(hilive)[1:20],as.numeric)

read_eval_data <- rbind_own(read_eval_data,hilive)

# kNN ----
knn_files <- list.files("/home/uli/Dropbox/HPI_RKI/DeePaC/model_evaluation_subreads/kNN/", pattern = ".csv", full.names = T,recursive = T)

knn <-  do.call(rbind,lapply(knn_files, read_deepac))
knn <- knn[knn$epoch!="epoch",]
knn <- knn%>%
  rename(subread_length = "epoch")%>%
  mutate(subread_length = ifelse(!str_detect(set, "ext"),str_extract(set,"\\d\\d+"),250),
         pred_type = ifelse(!str_detect(set, "ext|250_250"),"single_read","paired_read"))%>%
  mutate(subread_length_2 = ifelse(str_detect(set,"ext"),str_extract(set,"\\d\\d+"),NA  ))%>%
  mutate(type = "knn")

knn$subread_length[knn$set %in% c("all_test_1","all_test_2")] = 250
knn$subread_length_2[knn$set %in% c("all_test_250_250")] = 250
knn$set[str_detect(knn$set,"test_1")] = "all_test_1"
knn$set[str_detect(knn$set,"test_ext|all_test_250_250")] = "all_test_1_all_test_2"
cols_numeric <- colnames(knn)[3:15]
knn <- knn%>%
  mutate_at(cols_numeric,as.numeric)

read_eval_data <- rbind_own(read_eval_data,knn)


# calc avg ----
# calc average over all read lengths (single read data)
read_eval_data_avg_part_1 <- read_eval_data%>%
  filter(pred_type !="paired_read")%>%
  group_by(training,set)%>%
  summarise_if(is.numeric,mean)%>%
  mutate(subread_length = "avg")

read_eval_data_avg_part_2 <- read_eval_data%>%
  filter(pred_type !="paired_read")%>%
  group_by(training,set)%>%
  mutate(subread_length = "avg")%>%
  summarise_if(function(x){!is.numeric(x)},unique)

read_eval_data_avg <- merge(read_eval_data_avg_part_1,read_eval_data_avg_part_2,by = c("training","set","subread_length"))
read_eval_data <- rbind(read_eval_data,read_eval_data_avg)
rm(read_eval_data_avg,read_eval_data_avg_part_1,read_eval_data_avg_part_2)

# save as rds
saveRDS(read_eval_data,file = "read_eval_data.rds")

