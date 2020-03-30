library(readODS)

# split the odt file containing the HiLive results into seperate csvs 
# the csvs are supposed to be somehwat similar to deepac eval outout
# (easier to pass them to data pred script that way )
file <- "/home/uli/Dropbox/HPI_RKI/DeePaC/model_evaluation_subreads/HiLive/hybrid-performance.ods"
sheets <- ods_sheets(path = file)
for(sheet in sheets){
  data <- read_ods(path = file,sheet = sheet,skip = 1,formula_as_formula = F)
  data <- data[c(1:10,12:21),-c(1,22,23)]
  data$subread_length <- c(seq(25,250,25),rep(250,10))
  data$subread_length_2 <- c(rep(NA,10),seq(25,250,25))
  data$type <- sheet
  data$pred_type <- c(rep("single_read",10),rep("paired_read",10))
  data$set <- c(rep("test_1",10),rep("test_1_test_2",10))
  write.csv(x = data,file= paste0("/home/uli/Dropbox/HPI_RKI/DeePaC/model_evaluation_subreads/HiLive/",tolower(sheet),".csv" ),sep =",",dec = ".",row.names = F )
}

