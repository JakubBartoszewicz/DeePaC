#---------- Setup ----------
library(ranger)
library(paprbag)
library(doParallel)
source("~/PaPrLive/Scripts/Functions_Jakub.R")
load("~/PaPrLive/Data/config-dna.rda")
FeatureSetName <- "DNA"
subreadLengths <- c(250)
ForestSetFolders <- c("DNA_250")
Cores <- 12
folds <- 1:5
Path2Stats <- "~/PaPrLive/Data/bal_results"

#---------- Test -------------
sapply(ForestSetFolders, function(ForestSetFolder){
  #---------- Predict ----------
  for (subreadLength in subreadLengths){
    sapply(folds, Predict.subreadSet.fold, Feature.Configuration = Feature.Configuration, FeatureSetName = FeatureSetName, Cores = Cores, Path2Classifiers="~/PaPrLive/Data/classifiers_for_bal", ForestSetFolder=ForestSetFolder, subreadLength=subreadLength, feat.save.folder="balanced_test")
  }
  
  #---------- Prepare Data ----------
  sapply(subreadLengths, Prepare.Data, folds = folds, TrainingFolder=ForestSetFolder,TestReadFolder="classifiers_for_bal/TrainingData", ResultFolder="bal_results")
  
  
  #---------- Compare stats ----------
  PredictionData <- lapply(subreadLengths, function(x) {readRDS(file.path(Path2Stats,ForestSetFolder,paste("PredictionData_labelled_",x,"bp.rds", sep="")))})
  all.results <- lapply(PredictionData, Compute.Stats.byRead)
  all.results <- do.call(rbind.data.frame, all.results)
  rownames(all.results) <- subreadLengths
  write.csv2(x = all.results, file = file.path(Path2Stats, paste("stats-comparison-",ForestSetFolder,".csv", sep="")))
})

#---------- Visualise ----------
pattern <- "stats-comparison-"
Measures <- c("TPR", "TNR","ACC", "F1", "MCC", "I", "AUC")
Draw.Plots(subreadLengths, Path2Stats, pattern, Measures)
