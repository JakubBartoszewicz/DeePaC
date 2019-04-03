#---------- Setup ----------
library(ranger)
library(paprbag)
library(doParallel)
source("~/PaPrLive/Scripts/Functions_scripts.R")
load("~/PaPrLive/Data/config-paper.rda")
FeatureSetName <- "ALL"
subreadLengths <- c(250)
ForestSetFolders <- c("ALL_250")
Cores <- 1
folds <- c(1)
Path2Stats <- "~/PaPrLive/Data/results"

#---------- Test -------------
sapply(ForestSetFolders, function(ForestSetFolder){
  #---------- Predict ----------
  for (subreadLength in subreadLengths){
    StartTime <- proc.time()[3]
    sapply(folds, function(f){
      StartTime <- proc.time()[3]
      Predict.featSet.fold(f, Feature.Configuration = Feature.Configuration, FeatureSetName = FeatureSetName, 
           Cores = Cores, Path2Classifiers="~/PaPrLive/Data/classifiers", ForestSetFolder=ForestSetFolder, subreadLength=subreadLength,
           savePredictions = F, verbose = T)
      EndTime <- proc.time()[3]
      print(paste("Prediction for", FeatureSetName, subreadLength, "took",EndTime - StartTime) )
    })
  }
})

