source("./SCRATCH_NOBAK/PaPrLive_ROGAL/Scripts/Functions_scripts.R")
Path2Stats <- "./SCRATCH_NOBAK/PaPrLive_ROGAL/Data/results"
ForestSetFolders <- c("ALL_250_10e7", "DNA_250_10e7")
Paired <- T
PathMode <- "test" 
LeftSet <- "test_1"
RightSet <- "test_2"

sapply(ForestSetFolders, function(ForestSetFolder){
    predlist.L <- readRDS(paste0("SCRATCH_NOBAK/PaPrLive_ROGAL/Data/classifiers/TrainingData/",ForestSetFolder,"/fold1/Analysis_250_",LeftSet,"/AllPredictions_list.rds"))
    species.L <- data.frame(Labels = sapply(predlist.L,function(x){attr(x,"OSlabel")}), ML.FALSE = sapply(predlist.L, function(x){mean(x$ML.FALSE)}), ML.TRUE = sapply(predlist.L, function(x){mean(x$ML.TRUE)}))
    stats.L <- Compute.Stats.byRead(species.L, FeatureSetName = stringr::str_replace(string = ForestSetFolder, pattern = "_.*", replacement = ""))
    if (Paired) {
        predlist.R <- readRDS(paste0("SCRATCH_NOBAK/PaPrLive_ROGAL/Data/classifiers/TrainingData/",ForestSetFolder,"/fold1/Analysis_250_",RightSet,"/AllPredictions_list.rds"))
        species.R <- data.frame(Labels = sapply(predlist.R,function(x){attr(x,"OSlabel")}), ML.FALSE = sapply(predlist.R, function(x){mean(x$ML.FALSE)}), ML.TRUE = sapply(predlist.R, function(x){mean(x$ML.TRUE)}))
        predlist <- lapply(1:length(predlist.L), function(i){result <- predlist.L[[i]]; result$ML.FALSE <- rowMeans(cbind(result$ML.FALSE, predlist.R[[i]]$ML.FALSE)); result$ML.TRUE <- rowMeans(cbind(result$ML.TRUE, predlist.R[[i]]$ML.TRUE)); return(result)})
        species <- data.frame(Labels = sapply(predlist,function(x){attr(x,"OSlabel")}), ML.FALSE = sapply(predlist, function(x){mean(x$ML.FALSE)}), ML.TRUE = sapply(predlist, function(x){mean(x$ML.TRUE)}))    
        stats.R <- Compute.Stats.byRead(species.R, FeatureSetName = stringr::str_replace(string = ForestSetFolder, pattern = "_.*", replacement = ""))
        stats <- Compute.Stats.byRead(species, FeatureSetName = stringr::str_replace(string = ForestSetFolder, pattern = "_.*", replacement = ""))   
        all.results <- do.call(rbind, list(stats.L, stats.R, stats))  
        write.csv2(x = all.results, file = file.path(Path2Stats, paste("species-comparison-",ForestSetFolder,"-",PathMode,".csv", sep="")))        
    } else {
        write.csv2(x = stats.L, file = file.path(Path2Stats, paste("species-comparison-",ForestSetFolder,"-",PathMode,".csv", sep="")))
    }  
})
