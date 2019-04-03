#---------- Setup ----------
library(ranger)
library(paprbag)
library(doParallel)
source("./SCRATCH_NOBAK/PaPrBaG/Scripts/Functions_scripts.R")
load("./SCRATCH_NOBAK/PaPrLive_2018/Data/config-dna.rda")
FeatureSetName <- "DNA"
subreadLengths <- c(250)
ForestSetFolders <- c("DNA_250_10e7")
Cores <- 16
folds <- c(1)
Path2Stats <- "./SCRATCH_NOBAK/PaPrLive_2018/Data/results"
DatasetName <- "balanced.fold"
PathMode <- "test"

get.mean.preds <- function(x, ForestSetFolder) {
      test1 <- readRDS(file.path(Path2Stats,ForestSetFolder,paste("PredictionData_labelled_",x,"bp_", PathMode, "_1.rds", sep="")))
      test2 <- readRDS(file.path(Path2Stats,ForestSetFolder,paste("PredictionData_labelled_",x,"bp_", PathMode, "_2.rds", sep="")))

      test <- test1
      test$ML.FALSE <- rowMeans(cbind(test1$ML.FALSE, test2$ML.FALSE))
      test$ML.TRUE <- rowMeans(cbind(test1$ML.TRUE, test2$ML.TRUE))

      saveRDS(object = test, file = file.path(Path2Stats,ForestSetFolder,paste("PredictionData_labelled_",x,"bp_", PathMode, ".rds", sep="")))
}


#---------- Test -------------
sapply(ForestSetFolders, function(ForestSetFolder){
  #---------- Prepare Data ----------
  sapply(subreadLengths, get.mean.preds, ForestSetFolder= ForestSetFolder)
    
  #---------- Compare stats ----------
  PredictionData <- lapply(subreadLengths, function(x) {readRDS(file.path(Path2Stats,ForestSetFolder,paste("PredictionData_labelled_",x,"bp_", PathMode, ".rds", sep="")))})
  all.results <- lapply(PredictionData, Compute.Stats.byRead)
  all.results <- do.call(rbind.data.frame, all.results)
  rownames(all.results) <- subreadLengths
  write.csv2(x = all.results, file = file.path(Path2Stats, paste("stats-comparison-",ForestSetFolder,"-",PathMode,".csv", sep="")))
})

#---------- Visualise ----------
pattern <- "stats-comparison-"
Measures <- c("ACC", "AUC", "MCC", "F1", "TPR", "PPV")
Draw.Plots(subreadLengths, Path2Stats, pattern, Measures)
