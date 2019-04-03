#---------- Setup ----------
library(paprbag)
library(ranger)
library(doParallel)
source("~/PaPrLive/Scripts/Functions_scripts.R")

load("~/PaPrLive/Data/config-paper.rda")
FeatureSetNames <- c("ALL")
subreadLengths <- c(75)
folds <- c(1)
Cores.Feat.Table <- c(1,1,1,1,1)

print(paste("Extracting features on", Cores.Feat.Table[1], Cores.Feat.Table[2], Cores.Feat.Table[3], Cores.Feat.Table[4], Cores.Feat.Table[5], "cores"))

#---------- Extract -----------
sapply(FeatureSetNames, function(FeatureSetName){
  for (subreadLength in subreadLengths){
    sapply(folds, function(f){
      StartTime <- proc.time()[3]
      Extract.Save.Features(fold=f, Feature.Configuration = Feature.Configuration, Cores.Feat = Cores.Feat.Table[f], FeatureSetName = FeatureSetName,
                            subreadLength = subreadLength, path.mode = "training", create.table = F, saveFeatures = F, verbose = F)
      EndTime <- proc.time()[3]
      print(paste("Feature creation for", FeatureSetName, subreadLength, "took",EndTime - StartTime) )
    })
  }
})
