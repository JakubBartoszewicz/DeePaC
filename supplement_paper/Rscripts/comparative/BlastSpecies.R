P <- 96
N <- 797
Paired <- T
SetName <- "test"
Path.L <- "SCRATCH_NOBAK/Benchmark_virS/PathogenReads/OtherData/HP_NHP/testData/Reads/Left_250bp_fold1/Blast/"
Path.R <- "SCRATCH_NOBAK/Benchmark_virS/PathogenReads/OtherData/HP_NHP/testData/Reads/Right_250bp_fold1/Blast/"

accept.anything <- Vectorize(function (x,y) {
    if (is.na(x)) return (y)
    if (is.na(y)) return (x)
    if (x==!y) return (NA)
    if (x==y) return (x)
})

get.performance <- function (prediction, y, P=0, N=0) {
    TP <- sum(prediction==T & y==T, na.rm = TRUE)
    TN <- sum(prediction==F & y==F, na.rm = TRUE)    
    FP <- sum(prediction==T & y==F, na.rm = TRUE)    
    FN <- sum(prediction==F & y==T, na.rm = TRUE)
    
    if (P<0) P = TP + FN
    if (N<0) N = TN + FP
    
    # sensitivity or true positive rate / recall (TPR)
    sensitivity = TP/(TP+FN)
    # specificity
    specificity = TN/(TN+FP)
    # precision
    precision = TP/(TP+FP)
    # accuracy (ACC)
    ACC = (TP + TN) / (TP + FP + FN + TN)
    # F1 score
    F1 = 2 * precision * sensitivity / (precision + sensitivity)
    # MCC
    MCC_denominator <- sqrt( ( as.numeric(TP)+ FP) * (as.numeric(TP) +  FN ) * (  as.numeric(TN) +  FP ) * (  as.numeric(TN) +  FN ) )
    if(MCC_denominator == 0) MCC_denominator <- 1
    MCC = (as.numeric(TP) * TN - as.numeric(FP) * FN) / MCC_denominator
    
    # sensitivity or true positive rate / recall (TPR)
    total.sensitivity = TP/P
    # specificity
    total.specificity = TN/N
    # precision
    total.precision = precision
    # accuracy (ACC)
    total.ACC = (TP + TN) / (P + N)
    # F1 score
    total.F1 = 2 * total.precision * total.sensitivity / (total.precision + total.sensitivity)
    # MCC
    total.MCC = MCC
    
    predictions = length(prediction)/(P+N)
    
    return(data.frame(TP=TP,TN=TN,FP=FP,FN=FN,TPR = sensitivity, TNR=specificity, PPV = precision, ACC = ACC, F1 = F1, MCC = MCC, total.TPR = total.sensitivity, total.TNR=total.specificity, total.PPV = total.precision, total.ACC = total.ACC, total.F1 = total.F1, total.MCC = total.MCC, predictions = predictions))
}

for (trainingSet in c("AllTrainingGenomes")){
    file.paths.L <- list.files(path = paste0(Path.L, trainingSet), pattern = "matched\\.rds$", full.names = T)
    file.data.L <- lapply(file.paths.L, readRDS)
    #merged.L <- do.call("rbind", file.data)
    #merged.L$read <- rownames(merged.L)
   
    specnames.L <-  make.names(sapply(file.data.L, function(x){return(x$QuerySpecies[1])}), unique=TRUE)
    for (i in 1:length(specnames.L)){
    	file.data.L[[i]]$QuerySpecies <- specnames.L[i]
    }

    species.L <- lapply(file.data.L, function(x){label <- sum(x$MatchedLabel) >= sum(!(x$MatchedLabel)); return(data.frame(MatchedLabel=label, QueryLabel=x$QueryLabel[1], QuerySpecies = x$QuerySpecies[1]))})
    species.L <- do.call("rbind", species.L)

    rownames(species.L) <- specnames.L
    
    if (Paired){
        file.paths.R <- list.files(path = paste0(Path.R, trainingSet), pattern = "matched\\.rds$", full.names = T)
        file.data.R <- lapply(file.paths.R, readRDS)

        specnames.R <- make.names(sapply(file.data.R, function(x){return(x$QuerySpecies[1])}), unique=TRUE)
        for (i in 1:length(specnames.R)){
            file.data.R[[i]]$QuerySpecies <- specnames.R[i]
        }

	species.R <- lapply(file.data.R, function(x){label <- sum(x$MatchedLabel) >= sum(!(x$MatchedLabel)); return(data.frame(MatchedLabel=label, QueryLabel=x$QueryLabel[1], QuerySpecies = x$QuerySpecies[1]))})
        species.R <- do.call("rbind", species.R)


	rownames(species.R) <- specnames.R

	species.join <-  merge(species.L, species.R, by = "QuerySpecies", all = TRUE, suffixes = c(".L", ".R"))
        species.join$Prediction <- accept.anything(species.join$MatchedLabel.L, species.join$MatchedLabel.R)
        species.pred <- species.join[!is.na(species.join$Prediction),]   
        species.pred$QueryLabel.L[is.na(species.pred$QueryLabel.L)] <- species.pred$QueryLabel.R[is.na(species.pred$QueryLabel.L)]
    }
    
    test.L <- get.performance(species.L$MatchedLabel[!is.na(species.L$MatchedLabel)], species.L$QueryLabel[!is.na(species.L$MatchedLabel)], P, N)
    
    if (Paired){
        test.R <- get.performance(species.R$MatchedLabel[!is.na(species.R$MatchedLabel)], species.R$QueryLabel[!is.na(species.R$MatchedLabel)], P, N) 
        test <- get.performance(species.pred$Prediction, species.pred$QueryLabel.L, P, N)
        results <- do.call("rbind", list(test.L,test.R,test))  
        rownames(results) <- c("test.L","test.R","test")
    } else {
        results <- test.L
        rownames(results) <- c("test.L")
    }
    write.csv2(results, file = paste0("Blast_Species_",SetName,"_", trainingSet, ".csv"))    
}


