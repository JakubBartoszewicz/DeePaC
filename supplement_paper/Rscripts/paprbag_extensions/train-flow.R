#---------- Setup ----------
library(paprbag)
library(ranger)
library(doParallel)
library(foreach)
source("./SCRATCH_NOBAK/PaPrLive_2018/Scripts/Functions_scripts.R")

load("./SCRATCH_NOBAK/PaPrLive_2018/Data/config-all.rda")
FeatureSetNames <- c("ALL")
subreadLengths <- c(250)
Cores.Train <- 32
folds <- c(1)
Cores.Feat.Table <- c(8,4,4,4,4)
do.Features <- TRUE
createFeatures <- TRUE
options(warn=1)

print(paste("Training on", Cores.Train, "cores"))
print(paste("Extracting features on", paste(Cores.Feat.Table, collapse=" "), "cores"))



#---------- Train -----------
sapply(FeatureSetNames, function(FeatureSetName){
  for (subreadLength in subreadLengths){
    if(do.Features){sapply(folds, function(f){
      Extract.Save.Features(fold=f, Feature.Configuration = Feature.Configuration, Cores.Feat = Cores.Feat.Table[f], FeatureSetName = FeatureSetName,
                            subreadLength = subreadLength, path.mode = "training", create.table = T, Path2Data="./SCRATCH_NOBAK/PaPrLive_2018/Data/", DatasetName="10e7.fold", createFeatures = createFeatures)
    })};
    sapply(folds, Train.fold, Feature.Configuration = Feature.Configuration, FeatureSetName = FeatureSetName,
           Cores.Train = Cores.Train, subreadLength=subreadLength, extract.features = F, Path2Data="./SCRATCH_NOBAK/PaPrLive_2018/Data/", DatasetName="10e7.fold")
  }
})
