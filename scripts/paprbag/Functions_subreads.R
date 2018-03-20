#--------------- Training -------------------
Extract.Save.Features <- function(fold, Feature.Configuration, Cores.Feat, FeatureSetName, subreadLength = 250, path.mode=c("training", "test", "balanced_test"), create.table = F, saveFeatures = T, verbose = T){
  Feature.Configuration$Path2ReadFiles <- paste("~/PaPrLive/Data/", path.mode, "/Illumina_250bp_SE.fold", fold, sep="")
  if (saveFeatures)
    Feature.Configuration$savePath <- file.path(Feature.Configuration$Path2ReadFiles, paste("Features", FeatureSetName, subreadLength, sep="_"))
  else
    Feature.Configuration$savePath <- NULL
  Feature.Configuration$subreadLength <- subreadLength
  Feature.Configuration$Cores <- Cores.Feat
  Feature.Configuration$verbose <- verbose
  print(paste("### Processing", subreadLength, "bp subreads, fold", fold,"###"))
  do.call(UpdateFeatures.Subreads, Feature.Configuration)
  gc()
  if(create.table){
    data("Labels")
    
    Create.TrainingDataSet (Path2Files = Feature.Configuration$savePath, pattern="Features",OSlabels = Labels, savePath = file.path(Feature.Configuration$Path2ReadFiles,"TrainingData", paste(FeatureSetName, subreadLength, sep="_")))
    gc()
  }
}


Train.fold <- function(fold, Feature.Configuration, FeatureSetName, Cores.Train, subreadLength, Cores.Feat = Cores.Train, verbose=T, extract.features = T){
  if(extract.features){
    Extract.Save.Features(fold=fold, Feature.Configuration=Feature.Configuration, Cores.Feat=Cores.Feat, FeatureSetName=FeatureSetName, subreadLength=subreadLength, path.mode="training", create.table = T, verbose=verbose)
    gc()
  }

  Feature.Configuration$Path2ReadFiles <- paste("~/PaPrLive/Data/training/Illumina_250bp_SE.fold", fold, sep="")
  dir <- file.path(Feature.Configuration$Path2ReadFiles, "TrainingData", paste(FeatureSetName, subreadLength, sep="_"))
  Path2FeatureFile <- file.path(dir,"FeatureTable.rds")
  Path2LabelFile <- file.path(dir,"ReadLabel_OS.rds")
  savePath <- file.path(dir,paste("ranger", FeatureSetName, subreadLength, sep="_"))
  # Run with path information
  Run.Training (Path2FeatureFile = Path2FeatureFile, Path2LabelFile = Path2LabelFile, savePath = savePath, verbose = T, num.threads = Cores.Train)
  gc()
}
#--------------- Visualise -------------------

Draw.Plots <- function(subreadLengths, Path2Stats="./PaPrLive/Data/results", pattern="stats-comparison-",
                       Measures = NULL, ordering = NULL, doEnvelope = TRUE, plot.width = 640, plot.height = 480, plot.lwd = 3, plot.cex =1){
  require(RColorBrewer)
  if(is.null(Measures))
    Measures <- c("TPR", "TNR","ACC", "F1", "MCC", "I", "AUC")
  ReadFiles <- list.files(Path2Stats, pattern = pattern, full.names = T)
  Labels <- toupper(sapply(list.files(Path2Stats, pattern = pattern, full.names = F),
                   function(x){gsub(pattern = ".csv", replacement="", x=
                                      gsub(pattern = pattern, replacement="", x=x))}))
  if(!is.null(ordering)&&length(ordering)>1)
  {
    ReadFiles <- ReadFiles[ordering]
    Labels <- Labels[ordering]
  }
  stats.data <- lapply(ReadFiles, read.csv2)
  names(stats.data) <- Labels
  nlines <- length(ReadFiles)
  dir.create(file.path(Path2Stats, "plots"))
  if (doEnvelope)
    Labels <- c(Labels, "Envelope")
  
  
  colors <- c(brewer.pal(n = nlines, name = "Paired"), "#333333")
  linetype <- c(rep.int(1, nlines), 2)
  plotchar <- c(rep.int(20, nlines), 21)
  
  #pdf
  pdf(file.path(Path2Stats, "plots", "report.pdf"))
  sapply(Measures, function(m){
    yrange <- range(sapply(stats.data, function(x){x[m]}))
    xrange <- range(subreadLengths)
    plot(xrange,yrange,type="n", xlab="subread length", ylab=m)
    if (doEnvelope)
      envelope <- sapply(1:length(subreadLengths), function(read){
        max(sapply(1:nlines, function(line){
          stats.data[[line]][read,m]
        }), na.rm = TRUE)
      })
    
    sapply(1:nlines, function(l){
      lines(x = stats.data[[l]][,1], y = stats.data[[l]][,m], type = "o", lwd=1.5,
            lty=linetype[l], col=colors[l], pch=plotchar[l])
    })
    
    if (doEnvelope)
      lines(x = subreadLengths, y = envelope, type = "o", lwd=1.5,
            lty=linetype[length(Labels)], col=colors[length(Labels)], pch=plotchar[length(Labels)])
    
    #title("")
    legend(xrange[1],yrange[2], Labels, cex=0.8, col=colors,
           pch=plotchar, lty=linetype, title="Forest")
  })
  dev.off()
  
  #pngs
  sapply(Measures, function(m){
    png(file.path(Path2Stats, "plots", paste(m, ".png", sep="")), width = plot.width, height = plot.height)
    yrange <- range(sapply(stats.data, function(x){x[m]}))
    xrange <- range(subreadLengths)
    plot(xrange,yrange,type="n", xlab="subread length", ylab=m)
    if (doEnvelope)
      envelope <- sapply(1:length(subreadLengths), function(read){
        max(sapply(1:nlines, function(line){
          stats.data[[line]][read,m]
        }), na.rm = TRUE)
      })
    
    sapply(1:nlines, function(l){
      lines(x = stats.data[[l]][,1], y = stats.data[[l]][,m], type = "o", lwd=plot.lwd,
            lty=linetype[l], col=colors[l], pch=plotchar[l])
    })
    
    if (doEnvelope)
      lines(x = subreadLengths, y = envelope, type = "o", lwd=plot.lwd,
            lty=linetype[length(Labels)], col=colors[length(Labels)], pch=plotchar[length(Labels)])
    
    #title("")
    legend(xrange[1],yrange[2], Labels, cex=plot.cex, col=colors,
           pch=plotchar, lty=linetype, title="Forest")
    dev.off()
  })
}
#--------------- Stats -------------------
# (C) C. Deneke
Compute.BasicStats <- function(myTable, handle.NA = T, expand.Table = T){
  
  
  if(is.table(myTable)) {
    
    if(expand.Table != T) {
      if(!any(grepl("FALSE",rownames(myTable)) ) | !any(grepl("TRUE",rownames(myTable)) ) | !any(grepl("TRUE",colnames(myTable)) ) | !any(grepl("FALSE",colnames(myTable)) ) ) stop("The table does not have entries TRUE and FALSE")
    } else {
      # handle missing T & F rows, cols
      if(!any(grepl("FALSE",rownames(myTable)) )) {
        myTable <- as.table(rbind(myTable,"FALSE"=rep(0,ncol(myTable))) )
      }
      if(!any(grepl("TRUE",rownames(myTable)) )) {
        myTable <- as.table(rbind("TRUE"=rep(0,ncol(myTable)),myTable) )
      }
      if(!any(grepl("FALSE",colnames(myTable)) )) {
        myTable <- as.table(cbind(myTable,"FALSE"=rep(0,ncol(myTable))) )
      }
      if(!any(grepl("TRUE",colnames(myTable)) )) {
        myTable <- as.table(cbind("TRUE"=rep(0,ncol(myTable)),myTable) )
      }
    }
    
    
    if(handle.NA == T & any(is.na(rownames(myTable) ) )){
      NA_row <- which(is.na(rownames(myTable) ) )  
      missingP <- myTable[NA_row,"TRUE"]
      missingN <- myTable[NA_row,"FALSE"]
      
    } else {
      missingP <- 0
      missingN <- 0
    }
    
    TP <- myTable['TRUE','TRUE']
    TN <- myTable['FALSE','FALSE']
    FP <- myTable['TRUE','FALSE'] + missingN
    FN <- myTable['FALSE','TRUE'] + missingP
  } else if(is.numeric(myTable)  ) {
    if(! (any(grepl("TP",names(myTable) )) & any(grepl("TN",names(myTable) ) ) & any(grepl("FP",names(myTable) ) ) & any(grepl("FN",names(myTable) ) ) )) stop("The contingincy matrix has missing columns")
    TP = myTable["TP"]
    TN = myTable["TN"]
    FP = myTable["FP"]
    FN = myTable["FN"]
  }
  
  
  # sensitivity or true positive rate (TPR)
  sensitivity = TP/(TP+FN)
  specificity  = TN/(TN+FP)
  precision = TP/(TP+FP)
  # negative predictive value (NPV)
  NPV = TN / (TN + FN)
  # fall-out or false positive rate (FPR)
  FPR = FP / (FP + TN)
  # false negative rate (FNR)
  FNR = FN / (TP + FN)
  # false discovery rate (FDR)
  FDR = FP / (TP + FP)
  # accuracy (ACC)
  ACC = (TP + TN) / (TP + FP + FN + TN)
  # F1 score
  F1 = 2 * TP / (2 * TP + FP + FN)
  # Informedness
  Info = sensitivity + specificity - 1
  # MCC
  MCC_denominator <- sqrt( ( as.numeric(TP)+ FP) * (as.numeric(TP) +  FN ) * (  as.numeric(TN) +  FP ) * (  as.numeric(TN) +  FN ) )
  if(MCC_denominator == 0) MCC_denominator <- 1
  MCC = (as.numeric(TP) * TN - as.numeric(FP) * FN) / MCC_denominator
  # DOR (Diagnostic odds ratio)
  DOR = (TP/FP)/(FN/TN)
  
  
  return(data.frame(TP = TP, TN =TN, FP = FP, FN =FN, TPR = sensitivity, TNR = specificity, PPV = precision, ACC = ACC, F1 = F1, MCC = MCC, I = Info, DOR = DOR, NPV = NPV, FPR = FPR, FNR = FNR, FDR = FDR))
  
}

# --- (C) C. Deneke

Compute.AUC <- function(Predictions,Labels){
  require(ROCR)
  Prediction_rocr <- ROCR::prediction(Predictions,Labels)  
  AUC = performance( Prediction_rocr, "auc" )@y.values[[1]]
  return(AUC)
}

#(C) Jakub Bartoszewicz based on Carlus Deneke
Compute.Stats.byRead <- function(JoinedPredictionData, threshold = 0.5){
  
  myTable <- table(Prediction = ifelse(JoinedPredictionData[,"ML.TRUE"] > threshold,T,F) , Labels = JoinedPredictionData$Labels, useNA = 'always') 
  
  if(nrow(myTable) < 3) print(paste("Error for Column",Column,"Table:",paste(myTable,collapse = ",") ))
  MissingPredictions <- sum(myTable[3,])
  Stats <- Compute.BasicStats(myTable)
  Stats <- cbind(Stats, MissingPredictions = MissingPredictions)
  Stats <- cbind(Stats,AUC = Compute.AUC(Predictions = JoinedPredictionData[,"ML.TRUE"],Labels = JoinedPredictionData$Labels) )

  #rownames(Stats) <- strsplit(colnames(JoinedPredictionData)[SelectedColumns],".",fixed = T)[1]
  
  return(data.frame(Stats) )
}

#--------------- Data aggregation & preparation -------------------
Prepare.Fold<- function(subreadLength, TrainingFolder, FoldFolder="fold1", HomeFolder = "/home/bartoszewiczj", ProjectFolder = "PaPrLive",
                        WorkingDirectory = "Data",TestReadFolder = "classifiers/TrainingData", ResultFolder = "results"){
  
  # Prepare data for final analysis (C) Carlus Deneke, severe mod by Jakub Bartoszewicz
  
  
  Path2Results <- file.path(HomeFolder,ProjectFolder,WorkingDirectory,ResultFolder,TrainingFolder)
  if(!file.exists(Path2Results)) dir.create(Path2Results)

  Path <- file.path(HomeFolder,ProjectFolder,WorkingDirectory,TestReadFolder,TrainingFolder,FoldFolder,paste("Analysis_",subreadLength,sep=""),"AllPredictions_list.rds") 
  if(!file.exists(Path)) stop(paste("Analysis files are missing:", Path))
  
  AnalysisFiles <- do.call(c,lapply(c(Path), readRDS) )
  FileName <- paste("PredictionData_unlabelled_",subreadLength,"bp_", FoldFolder,".rds",sep="")
  
  saveRDS(AnalysisFiles,file.path(Path2Results,FileName))
  
  #Add labels
  # ---
  
  Labels <- sapply(AnalysisFiles, function(x) attr(x,"OSlabel") )
  
  Predictions_joined <- Join.PredictionStats (Predictions = AnalysisFiles,Labels = Labels)
  saveRDS(Predictions_joined,file.path(Path2Results, paste("PredictionData_labelled_",subreadLength,"bp_", FoldFolder,".rds", sep="") ))
}

Prepare.Data<- function(subreadLength, folds, TrainingFolder, HomeFolder = "/home/bartoszewiczj", ProjectFolder = "PaPrLive",
                        WorkingDirectory = "Data",TestReadFolder = "classifiers/TrainingData", ResultFolder = "results"){
  
  # Prepare data for final analysis (C) Carlus Deneke, severe mod by Jakub Bartoszewicz
  

  Path2Results <- file.path(HomeFolder,ProjectFolder,WorkingDirectory,ResultFolder,TrainingFolder)
  dir.create(Path2Results)
  
  
  # After prediction, bowtie, ...
  # call Aggregate for each fold
  sapply(folds, function(x){Aggregate.Predictions(WorkingDirectory, TestReadFolder, TrainingFolder, ModelFolder = paste("fold",x, sep = ""), subreadLength)})
  
  
  # ---
  # combine all folds
  
  # fold classifier paths
  ModelFolders_all <- list.dirs(file.path(HomeFolder,ProjectFolder,WorkingDirectory,TestReadFolder, TrainingFolder), recursive = F, full.names = F)
  # ensure only folds
  ModelFolders_all <- grep("fold",ModelFolders_all, value=T)
  Paths <- file.path(HomeFolder,ProjectFolder,WorkingDirectory,TestReadFolder,TrainingFolder,ModelFolders_all,paste("Analysis_",subreadLength,sep=""),"AllPredictions_list.rds") 
  Paths.Exist <- file.exists(Paths) 
  if(!all(Paths.Exist) ) stop(paste("Analysis files are missing:", Paths[!Paths.Exist]))
  
  AnalysisFiles <- do.call(c,lapply(Paths, readRDS) )
  FileName <- paste("PredictionData_unlabelled_",subreadLength,"bp.rds",sep="")
  
  saveRDS(AnalysisFiles,file.path(Path2Results,FileName))
  
  #Add labels
  # ---
  
  Labels <- sapply(AnalysisFiles, function(x) attr(x,"OSlabel") )
  
  Predictions_joined <- Join.PredictionStats (Predictions = AnalysisFiles,Labels = Labels)
  saveRDS(Predictions_joined,file.path(Path2Results, paste("PredictionData_labelled_",subreadLength,"bp.rds", sep="") ))
}

Join.PredictionStats <- function(Predictions,Labels){
  
  # join predictions
  if(!identical(names(Predictions),names(Labels))) stop("Names of predictions and labels do not match")
  
  library(plyr)
  Prediction_table <- rbind.fill(Predictions)
  Labels_reads <- unlist(lapply(1:length(Predictions),function(i) rep(Labels[i],nrow(Predictions[[i]])) ) )
  Prediction_table <- data.frame(Labels=Labels_reads,Prediction_table )
  
  return(Prediction_table)
}

Aggregate.Predictions <- function(WorkingDirectory, TestReadFolder, TrainingFolder, ModelFolder, subreadLength, HomeFolder="/home/bartoszewiczj", ProjectFolder ="PaPrLive" ){

  if(.Platform$OS.type == 'unix') dir.exists <- function(x) file.exists(x)
  
  # load some stuff
  IMGdata <- readRDS(file.path(HomeFolder,ProjectFolder,"Data/img/IMG_all_genomes_040615_taxontable6678_HumanOnly_final.rds") )
  
  # -------------------
  
  PredictionFolder <- file.path(HomeFolder,ProjectFolder,WorkingDirectory,TestReadFolder,TrainingFolder,ModelFolder,paste("Prediction_", subreadLength, sep=""))
  if(!dir.exists(PredictionFolder)) stop(paste("PredictionFolder",PredictionFolder,"does NOT exist"))  
  
  Path2Files <- file.path(HomeFolder,ProjectFolder,WorkingDirectory,TestReadFolder)
  PredictionFiles <- list.files(PredictionFolder,pattern = "rds$",full.names = T)
  
  AnalysisFolder <- file.path(Path2Files,TrainingFolder,ModelFolder,paste("Analysis_",subreadLength,sep=""))
  dir.create(AnalysisFolder)
  
  # Print Configuration
  print(paste("Processing data in ",file.path(HomeFolder,ProjectFolder,WorkingDirectory,TestReadFolder,TrainingFolder,ModelFolder)))
  
  print("Processing folders:")
  print(
    cbind(c("WorkingDirectory","TestReadFolder","TrainingFolder","ModelFolder"),
          c(WorkingDirectory,TestReadFolder,TrainingFolder,ModelFolder) )
  )
  
  print(paste("Output will be written to:",AnalysisFolder))
  
  
  # -----------------------
  # Settings
  
  Cores <- 5
  if(exists("Override.Outputname") ) Override_name <- readline("Choose desired outputname: ")
  
  # -----------------------
  
  library(plyr)
  library(stringr)
  library(foreach)
  if(Cores > 1) {
    library(doParallel)
    registerDoParallel(Cores)
  }
  
  
  AllPredictions_list  <-  foreach(i = 1:length(PredictionFiles)) %dopar% {
    
    print(paste("Processing file",i))
    
    CurrentID <- str_extract(PredictionFiles[i],"PRJ[A-Z]{2}[0-9]+")
    
    # Get filenames
    myPredictionFile <- PredictionFiles[i]
    if(!file.exists(myPredictionFile)) stop("Prediction file missing. That should not happen.")
    
    # read in data
    PredictionTable <- readRDS(myPredictionFile)
    AllPredictionsTable <- data.frame(ML = PredictionTable)
    
    # look-up label
    OSlabel <- IMGdata$Pathogenic[IMGdata$Bioproject.Accession == CurrentID]
    
    attr(AllPredictionsTable,"OSlabel") <- OSlabel[1]
    attr(AllPredictionsTable,"Bioproject.Accession") <- CurrentID
    
    return(AllPredictionsTable)
  }
  
  names(AllPredictions_list) <- str_extract(PredictionFiles,"PRJ[A-Z]{2}[0-9]+")
  
  
  if(!exists("Override_name")) OutputFile <- "AllPredictions_list.rds" else {
    OutputFile <- Override_name
  }
  
  saveRDS(AllPredictions_list,file.path(AnalysisFolder,OutputFile) )
}

#--------------- Reads2Predictions -------------------


Predict.subreadSet.fold <- function(fold, Feature.Configuration, FeatureSetName, Path2Classifiers, ForestSetFolder, subreadLength = 250, Cores = 5, saveOutput = T, verbose = T, feat.save.folder="test"){
  Feature.Configuration$Path2ReadFiles <- paste("~/PaPrLive/Data/balanced_test/Illumina_250bp_SE.fold", fold, sep="")
  if(!saveOutput)
    Feature.Configuration$Path2ReadFiles <- paste("~/PaPrLive/Data/training/Illumina_250bp_SE.fold", fold, sep="")
  Feature.Configuration$savePath <- file.path(Feature.Configuration$Path2ReadFiles, paste("Features", FeatureSetName, subreadLength, sep="_"))
  Extract.Save.Features(fold=fold, Feature.Configuration=Feature.Configuration, Cores.Feat=Cores, FeatureSetName=FeatureSetName, subreadLength=subreadLength, path.mode=feat.save.folder, saveFeatures = saveOutput, verbose=verbose)
  Prediction.workflow(Path2Files=Path2Classifiers,TrainingFolder=ForestSetFolder, ChooseForest=paste("fold",fold, sep = ""), Path2TestFiles=Feature.Configuration$savePath, PredictionFolderName = paste("Prediction_",subreadLength,sep=""), Cores = Cores, savePredictions = saveOutput, verbose = verbose)
  gc()
}

Predict.featSet.fold <- function(fold, Feature.Configuration, FeatureSetName, Path2Classifiers, ForestSetFolder, subreadLength = 250, Cores = 5, savePredictions = T, verbose = T){
  Feature.Configuration$Path2ReadFiles <- paste("~/PaPrLive/Data/test/Illumina_250bp_SE.fold", fold, sep="")
  if(!savePredictions)
    Feature.Configuration$Path2ReadFiles <- paste("~/PaPrLive/Data/training/Illumina_250bp_SE.fold", fold, sep="")
  Feature.Configuration$savePath <- file.path(Feature.Configuration$Path2ReadFiles, paste("Features", FeatureSetName, subreadLength, sep="_"))
  Feature.Configuration$subreadLength <- subreadLength
  Feature.Configuration$Cores <- Cores
  Feature.Configuration$verbose <- verbose
  print(paste("### Processing", subreadLength, "bp subreads, fold", fold,"###"))
  Prediction.workflow(Path2Files=Path2Classifiers,TrainingFolder=ForestSetFolder, ChooseForest=paste("fold",fold, sep = ""), Path2TestFiles=Feature.Configuration$savePath, PredictionFolderName = paste("Prediction_",subreadLength,sep=""), Cores = Cores, savePredictions = savePredictions, verbose = verbose)
  gc()
}

#' @title Update all subread features in a specified folder (C) Carlus Deneke, mod by Jakub Bartoszewicz
#' @description Update all features in a specified folder. Save features in subfolder "Features". Backup old features if desired.
#' @param Path2ReadFiles A path to the location of the read files
#' @param savePath Path where features should be saved to (default: Path2ReadFiles/Features)
#' @param Cores Multicore functionality for linux systems.
#' @param Backup Do you want to backup old features
#' @param AAindex_Selection Parameter for Features
#' @param pattern File extension of read files. Used as regex filter (default "fasta$|mason$")
#' @param verbose Write a logfile ("FeatureCreation.log") and verbose output
#' @param ... Parameters passed to function CreateFeaturesFromReads
#' @return Returns TRUE if completed
#' @author Carlus Deneke
#' @seealso Wraps \code{\link{CreateFeaturesFromReads}}
#' @importFrom foreach %dopar%
#' @export
UpdateFeatures.Subreads <- function(Path2ReadFiles, savePath = file.path(Path2ReadFiles,"Features"), subreadLength = 250, Cores = 1, Backup = F, pattern = "fasta$|mason$", verbose = T,...){
  
  require(foreach)
  require(Biostrings)
  
  if(Cores > 1) {
    #require(doParallel)
    doParallel::registerDoParallel(cores=Cores)
  }
  
  if(verbose == T){
    print(paste("A log-file is recorded under",file.path(Path2ReadFiles,"FeatureCreation.log") ))
    con = file(file.path(Path2ReadFiles,"FeatureCreation.log"), open = "a")
    sink(file=con, append=T, split=T)
    print(paste("New run for folder",Path2ReadFiles,"on",date() ))
  }
  
  if(verbose == T) StartTime <- proc.time()[3]
  
  ReadFiles <- list.files(Path2ReadFiles, pattern = pattern, full.names = T)
  
  # loop over all read files
  FeatureCheck <- foreach::foreach(i = 1: length(ReadFiles)) %dopar% {
    
    CurrentReadFile <- ReadFiles[i]
    print(paste("Creating features for file",CurrentReadFile,":",i))
    
    # Load data
    ReadData <-  Biostrings::readDNAStringSet(CurrentReadFile)
    
    # Simulate incomplete reads (Jakub)
    minReadLength <- min(Biostrings::width(ReadData))
    if(minReadLength < subreadLength) {warning(paste("Subreads cannot be longer than the shortest read:", minReadLength)); return(0)}
    if(subreadLength < minReadLength) {ReadData <- Biostrings::DNAStringSet(ReadData, end = subreadLength)}
    
    # Obtain features
    Features <- CreateFeaturesFromReads (Reads = ReadData, ...)
    
    # check feature qualitiy
    Check <- paprbag:::CheckFeatures(Features)
    if(!Check) {warning(paste("Feature check negative for file", CurrentReadFile)); return(0)}
    
    # Backup Features
    if (!is.null(savePath))
      paprbag:::SaveFeatureFile(Filename=CurrentReadFile, Features=Features, savePath = savePath, Backup = Backup)
    
    # Write Log
    if(verbose == T) print(paste("New feature set created for", CurrentReadFile,"on",date() ))
    if(verbose == T) print(paste("FeatureTable has dim",nrow(Features),",",ncol(Features),"and contains groups:",paste(attr(Features,"FeatureGroups"),collapse = ",") ))
    if(verbose == T) print(paste("Feature table contains the following features:",paste(colnames(Features), collapse = ",") ))
    if(verbose == T) if(Backup) print(paste("Backup of old feature version stored in subfolder Backup"))
    if(verbose == T) cat("\n")
    
    return(T)
  }
  
  if(verbose == T) EndTime <- proc.time()[3]
  
  if(verbose == T) print(paste("Feature creation took",EndTime - StartTime) )
  if(verbose == T) cat("#################\n")
  
  if(verbose == T) sink()
  if(verbose == T) close(con)
  
  return(T)
  
} # end function



# --- (C) Carlus Deneke, unmodded
#' Wrapper function that predicts set of test data for different forests
#' @details Works with randomForest and bigrf predict function. Predicts all combinations of forests in TrainingFolder and test features in TestDataFolder
#' @return Returns T if completed, saves predictions in subfolder "Predictions" in each trained model's folder 
Prediction.workflow <- function(Path2Files,TrainingFolder,Path2TestFiles, Labels = NULL, ChooseForest = NULL, PredictionFolderName = NULL,Cores = 1, savePredictions = T, verbose =F){
  
  require(foreach, quietly = T)
  # Register Cores
  if(Cores > 1) {
    require(doParallel, quietly = T)
    registerDoParallel(Cores)
  }  
    
  
  if(is.null(PredictionFolderName)) {
    PredictionFolderName  <- "Prediction"
  } else {
    PredictionFolderName  <- PredictionFolderName
  }
  
  
  ForestDirs <- list.dirs(file.path(Path2Files,"TrainingData",TrainingFolder), recursive = F )
  FeatureFiles <- list.files(Path2TestFiles, full.names = T, pattern = '^Features_' )  
    
  
  if(!is.null(ChooseForest) ) {
    ForestDirs <- ForestDirs[sapply(ChooseForest, function(x) grep(x,ForestDirs)[1])]
    if(is.na(ForestDirs)) stop("No forest recognized from variable ChooseForest")
  }
  
  Check_list <- foreach(CurrentForest = ForestDirs) %do% {
    if(verbose) print(paste("Processing forest",CurrentForest))
    rf <- readRDS(file.path(CurrentForest,"randomForest.rds"))
    
    if(savePredictions){
      if(file.exists(file.path(CurrentForest,PredictionFolderName))) {
        myPromt <- readline(prompt = "Prediction folder already exists. Do you want to continue? Typy y: ")
        if(myPromt != 'y') stop("Prediction folder already exists. You chose to abort execution.")
      }
      dir.create(file.path(CurrentForest,PredictionFolderName))
    }
    
    Check <- foreach(CurrentFeatureSet = FeatureFiles) %dopar% {
      
      if(verbose) print(paste("Processing feature set",CurrentFeatureSet))
      
      TestSet <- readRDS(CurrentFeatureSet)
      
      if(class(rf) == "randomForest") {
        require(randomForest, quietly = T)
        Prediction <- predict(rf,TestSet, type=ifelse(rf$type == "regression","response","prob") )
        
      } else if (class(rf) == "bigcforest") {
        require(bigrf, quietly = T)
        if(Cores > 1) registerDoSEQ()
        Prediction <- predict(rf, TestSet, y=NULL, printerrfreq=100L, printclserr=TRUE, cachepath=NULL, trace=0L)
        if(Cores > 1) registerDoParallel(Cores)   
        
        # convert, keep only normalized votes
        Prediction <- Prediction@testvotes/rowSums(Prediction@testvotes)
      } else if (class(rf) == "ranger") {
        require(ranger, quietly = T)
        Prediction <- predict(rf, dat = TestSet)$predictions
                
      } else stop("Class of random forest not recognized")
      
      # Saving
      # in training folder, seperate for every training run and rf-setting
      if(savePredictions){
        saveRDS(Prediction,file.path(CurrentForest,PredictionFolderName,sub("Features","Prediction",tail(strsplit(CurrentFeatureSet,"/")[[1]],1) )) )
      }
      return(1)
      
    } 
      return(Check)
    } # end for-loops
  
  
  registerDoSEQ()
  
  return( all(unlist(Check_list) == 1 ) )
  
} # end function
