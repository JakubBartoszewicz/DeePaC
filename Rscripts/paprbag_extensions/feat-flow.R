#---------- Setup ----------
library(ranger)
library(paprbag)
library(doParallel)
source("./SCRATCH_NOBAK/PaPrLive_2018/Scripts/Functions_scripts.R")
load("./SCRATCH_NOBAK/PaPrLive_2018/Data/config-all.rda")
FeatureSetName <- "ALL"
subreadLengths.all <- c(250)
subreadLengths <- c(250)
ForestSetFolders <- c("ALL_250_10e7")
Cores <- 16
folds <- c(1)
Path2Stats <- "./SCRATCH_NOBAK/PaPrLive_2018/Data/results"
DatasetName = "10e7.fold"
PathMode <- "val"
do.Prep <- FALSE

#---------- Test -------------
sapply(ForestSetFolders, function(ForestSetFolder){
  #---------- Predict ----------
  if(do.Prep) {
      for (subreadLength in subreadLengths){
        sapply(folds, Predict.featSet.fold, Feature.Configuration = Feature.Configuration, FeatureSetName = FeatureSetName, Cores = Cores, Path2Classifiers="./SCRATCH_NOBAK/PaPrLive_2018/Data/classifiers", ForestSetFolder=ForestSetFolder, subreadLength=subreadLength, Path2Data="./SCRATCH_NOBAK/PaPrLive_2018/Data/", DatasetName=DatasetName, path.mode=PathMode)
      }
      
      #---------- Prepare Data ----------
      sapply(subreadLengths.all, Prepare.Data, TrainingFolder=ForestSetFolder, ProjectFolder="SCRATCH_NOBAK/PaPrLive_2018", IMGPath="./IMG_1_folds_170418_sizes.rds", path.mode=PathMode, n.folds=1)
  }
  
  #---------- Compare stats ----------
  PredictionData <- lapply(subreadLengths.all, function(x) {readRDS(file.path(Path2Stats,ForestSetFolder,paste("PredictionData_labelled_",x,"bp_", PathMode, ".rds", sep="")))})
  all.results <- lapply(PredictionData, Compute.Stats.byRead, PathMode=PathMode, FeatureSetName=FeatureSetName)
  all.results <- do.call(rbind.data.frame, all.results)
  rownames(all.results) <- subreadLengths.all
  write.csv2(x = all.results, file = file.path(Path2Stats, paste("stats-comparison-",ForestSetFolder,"-", PathMode,".csv", sep="")))
})

#---------- Visualise ----------
pattern <- "stats-comparison-"
Measures <- c("TPR", "TNR","ACC", "F1", "MCC", "I", "AUC")
Draw.Plots(subreadLengths.all, Path2Stats, pattern, Measures)
