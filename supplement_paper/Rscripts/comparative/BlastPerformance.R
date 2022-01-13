P <- 625017
N <- 1875082
#target class = NA for binary classification
target.class <- 3
sra <- F
Paired <- T
SetName <- "multi3"
Path.L <- "SCRATCH_NOBAK/Benchmark_multi/PathogenReads/OtherData/HP_NHP/testData/Reads/Left_250bp_fold1/Blast/"
Path.R <- "SCRATCH_NOBAK/Benchmark_multi/PathogenReads/OtherData/HP_NHP/testData/Reads/Right_250bp_fold1/Blast/"

accept.anything <- Vectorize(function (x,y) {
    if (is.na(x)) return (y)
    if (is.na(y)) return (x)
    if (x==!y) return (NA)
    if (x==y) return (x)
})

get.performance <- function (prediction, y, P=0, N=0, target.class=NA) {
    if (!is.na(target.class)) {
        prediction <- lapply(prediction, as.numeric)
        y <- lapply(y, as.numeric)
        prediction <- prediction==target.class
        y <- y==target.class
    }

    TP <- sum(prediction==T & y==T, na.rm = TRUE)
    TN <- sum(prediction==F & y==F, na.rm = TRUE)
    FP <- sum(prediction==T & y==F, na.rm = TRUE)
    FN <- sum(prediction==F & y==T, na.rm = TRUE)

    if (P<0) P <- TP + FN
    if (N<0) N <- TN + FP

    # sensitivity or true positive rate / recall (TPR)
    sensitivity <- TP/(TP+FN)
    #specificity
    specificity <- TN/(TN+FP)
    # precision
    precision <- TP/(TP+FP)
    # accuracy (ACC)
    ACC <- (TP + TN) / (TP + FP + FN + TN)
    # F1 score
    F1 <- 2 * precision * sensitivity / (precision + sensitivity)
    # MCC
    MCC_denominator <- sqrt( ( as.numeric(TP)+ FP) * (as.numeric(TP) +  FN ) * (  as.numeric(TN) +  FP ) * (  as.numeric(TN) +  FN ) )
    if(MCC_denominator == 0) MCC_denominator <- 1
    MCC <- (as.numeric(TP) * TN - as.numeric(FP) * FN) / MCC_denominator

    # sensitivity or true positive rate / recall (TPR)
    total.sensitivity <- TP/P
    # specificity
    total.specificity <- TN/N
    # precision
    total.precision <- precision
    # accuracy (ACC)
    total.ACC <- (TP + TN) / (P + N)
    # F1 score
    total.F1 <- 2 * total.precision * total.sensitivity / (total.precision + total.sensitivity)
    # MCC
    total.MCC <- MCC

    predictions <- length(prediction)/(P+N)

    return(data.frame(TP=TP, TN=TN, FP=FP, FN=FN, TPR = sensitivity, TNR=specificity,PPV = precision, ACC = ACC, F1 = F1, MCC = MCC, total.TPR = total.sensitivity, total.TNR=total.specificity,total.PPV = total.precision, total.ACC = total.ACC, total.F1 = total.F1, total.MCC = total.MCC, predictions = predictions))
}

for (trainingSet in c("AllTrainingGenomes")){
    file.paths <- list.files(path = paste0(Path.L, trainingSet), pattern = "matched\\.rds$", full.names = T)
    file.data <- lapply(file.paths, readRDS)
    merged.L <- do.call("rbind", file.data)
    merged.L$read <- rownames(merged.L)
    if (sra){
        merged.L$read <- gsub('.$', '', merged.L$read)
        merged.L$QuerySpecies <- "SRA"
    }

    if (Paired){
        file.paths <- list.files(path = paste0(Path.R, trainingSet), pattern = "matched\\.rds$", full.names = T)
        file.data <- lapply(file.paths, readRDS)
        merged.R <- do.call("rbind", file.data)
        merged.R$read <- rownames(merged.R)

        if (sra){
            merged.R$read <- gsub('.$', '', merged.R$read)
            merged.R$QuerySpecies <- "SRA"
        }
        merged.join <- merge(merged.L, merged.R, by = "read", all = TRUE, suffixes = c(".L", ".R"))

        merged.join$QuerySpecies <- merged.join$QuerySpecies.L
        merged.join$QuerySpecies[is.na(merged.join$QuerySpecies)] <- merged.join$QuerySpecies.R[is.na(merged.join$QuerySpecies)]
        merged.join$QueryLabel <- merged.join$QueryLabel.L
        merged.join$QueryLabel[is.na(merged.join$QueryLabel)] <- merged.join$QueryLabel.R[is.na(merged.join$QueryLabel)]
        merged.join$Prediction <- accept.anything(merged.join$MatchedLabel.L, merged.join$MatchedLabel.R)
        merged.pred <- merged.join[!is.na(merged.join$Prediction),]
    }

    test.L <- get.performance(merged.L$MatchedLabel[!is.na(merged.L$MatchedLabel)], merged.L$QueryLabel[!is.na(merged.L$MatchedLabel)], P, N, target.class)

    if (Paired){
        test.R <- get.performance(merged.R$MatchedLabel[!is.na(merged.R$MatchedLabel)], merged.R$QueryLabel[!is.na(merged.R$MatchedLabel)], P, N, target.class)
        test <- get.performance(merged.pred$Prediction, merged.pred$QueryLabel, P, N, target.class)
        results <- do.call("rbind", list(test.L,test.R,test))
        rownames(results) <- c("test.L","test.R","test")
    } else {
        results <- test.L
        rownames(results) <- c("test.L")
    }
    write.csv2(results, file = paste0("Blast_",SetName,"_", trainingSet, ".csv"))
}


