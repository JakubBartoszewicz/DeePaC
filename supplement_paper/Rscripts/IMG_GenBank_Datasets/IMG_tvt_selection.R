set.seed(0)
library(stringr)

date.postfix.img <- "_fungi"
random.best <- FALSE
ambiguous.species <- FALSE

# no of folds to generate
k <- 9
validation_percent <- 0.1
test_percent <- 0.1
fold_names <- paste0("fold", 1:k)
download_format = ".fna"
# no of folds to download
k_download <- 1
# generate urls
urls <- TRUE

IMG.all <- readRDS(paste0("IMG_assemblies",date.postfix.img,".rds"))
if (!ambiguous.species) {
  IMG.all$Ambiguous <- FALSE
}
IMG.all$assembly_level <- factor(IMG.all$assembly_level,levels(IMG.all$assembly_level), ordered = TRUE)
IMG.all$assembly_accession <- as.character(IMG.all$assembly_accession)

Species_HP <- as.character(unique(IMG.all$Species[which(IMG.all$Pathogenic == TRUE & IMG.all$Ambiguous == FALSE)]))
Species_NP <- as.character(unique(IMG.all$Species[which(IMG.all$Pathogenic == FALSE & IMG.all$Ambiguous == FALSE)]))
Species <- c(Species_HP, Species_NP)

### tvt
IMG.all[,fold_names] <- ""
IMG.all$subset <- ""

Species_HP_test_manual <- c("Candida auris", "Aspergillus fumigatus")
Species_NP_test_manual <- c("Pyricularia oryzae", "Batrachochytrium dendrobatidis")
Species_HP_test <- sample(Species_HP, ceiling(test_percent*length(Species_HP))-length(Species_HP_test_manual))
Species_HP_test <- c(Species_HP_test_manual, Species_HP_test)
Species_NP_test <- sample(Species_NP, ceiling(test_percent*length(Species_NP))-length(Species_NP_test_manual))
Species_NP_test <- c(Species_NP_test_manual, Species_NP_test)

IMG.all[IMG.all$Species %in% Species_HP_test,fold_names] <- "test"
IMG.all[IMG.all$Species %in% Species_NP_test,fold_names] <- "test"

Species_test <- c(Species_HP_test,Species_NP_test)

Species_HP_done <- Species_HP_test
Species_NP_done <- Species_NP_test

SampleStrains <- function(SpeciesList, LabelsList, IMGdata, Prefix, fold=1){

  # get strains
  Accessions <- lapply(SpeciesList, function(QuerySpecies){
    IMGdata[IMGdata$Species == QuerySpecies,"assembly_accession"]
  })
  names(Accessions) <- SpeciesList

  RandomStrain.Accession <- sapply(Accessions, function(Species) {
    Species[sample(1:length(Species),1)]
  })
  IMGdata$subset[match(RandomStrain.Accession,IMGdata$assembly_accession)] <- "selected"

  if (fold<2){
    saveRDS(RandomStrain.Accession,file.path(paste(Prefix,"Strains.rds", sep="") ))
  } else {
    saveRDS(RandomStrain.Accession,file.path(paste(Prefix,"Strains_", fold, ".rds", sep="") ))
  }

  # save label information
  names(LabelsList) <- RandomStrain.Accession
  if (fold<2){
    saveRDS(LabelsList, file.path(paste(Prefix,"Labels.rds", sep="")  ))
  } else {
    saveRDS(LabelsList, file.path(paste(Prefix,"Labels_", fold, ".rds", sep="") ))
  }

  return(IMGdata)
}

SampleBestStrains <- function(){
  ### sample strains
  assembly_level <- sapply(Species, function(s){min(IMG.all$assembly_level[IMG.all$Species == s])})
  names(assembly_level) <- Species

  IMG.all$subset[as.character(IMG.all$assembly_level) == as.character(assembly_level[IMG.all$Species])] <- "candidate"
  IMG.all$subset[!(as.character(IMG.all$assembly_level) == as.character(assembly_level[IMG.all$Species]))] <- "other"

  selected_assemblies <- sapply(Species, function(s){candidate_assemblies <- IMG.all[IMG.all$Species == s & IMG.all$subset == "candidate", "assembly_accession"]; return(candidate_assemblies[sample(1:length(candidate_assemblies),1)]) })
  IMG.all$subset[IMG.all$assembly_accession %in% selected_assemblies] <- "selected"
}

if(!random.best & !ambiguous.species){
  Labels_test <- c(rep(T,length(Species_HP_test)),rep(F,length(Species_NP_test)))
  IMG.all <- SampleStrains(Species_test, Labels_test, IMG.all, "Test")
}

Species_HP_trainval <- setdiff(Species_HP,Species_HP_done)
Species_NP_trainval <- setdiff(Species_NP,Species_NP_done)

fold_val_sizes_HP <- sapply(1:k, function(i) {floor(length(Species_HP_trainval)/k)})
if(length(Species_HP_trainval) %% k > 0){
  fold_val_sizes_HP[1:(length(Species_HP_trainval) %% k)] <- fold_val_sizes_HP[1:(length(Species_HP_trainval) %% k)] + 1
}
fold_val_sizes_NP <- sapply(1:k, function(i) {floor(length(Species_NP_trainval)/k)})
if(length(Species_NP_trainval) %% k > 0){
  fold_val_sizes_NP[1:(length(Species_NP_trainval) %% k)] <- fold_val_sizes_NP[1:(length(Species_NP_trainval) %% k)] + 1
}

for (i in 1:k) {
  Species_HP_trainval <- setdiff(Species_HP,Species_HP_done)
  Species_NP_trainval <- setdiff(Species_NP,Species_NP_done)

  Species_HP_training <- sample(Species_HP_trainval, length(Species_HP_trainval) - fold_val_sizes_HP[i])
  Species_NP_training <- sample(Species_NP_trainval, length(Species_NP_trainval) - fold_val_sizes_NP[i])
  Species_HP_validation <- setdiff(Species_HP_trainval,Species_HP_training)
  Species_NP_validation <- setdiff(Species_NP_trainval,Species_NP_training)

  IMG.all[IMG.all$Species %in% Species_HP_training,fold_names[i]] <- "train"
  IMG.all[IMG.all$Species %in% Species_NP_training,fold_names[i]] <- "train"

  IMG.all[IMG.all$Species %in% Species_HP_validation,fold_names[i]] <- "val"
  IMG.all[IMG.all$Species %in% Species_NP_validation,fold_names[i]] <- "val"

  if (i<k){
    # if validation here, training in other folds
    IMG.all[IMG.all$Species %in% Species_HP_validation,fold_names[(i+1):k]] <- "train"
    IMG.all[IMG.all$Species %in% Species_NP_validation,fold_names[(i+1):k]] <- "train"
  }

  Species_HP_done <- c(Species_HP_done, Species_HP_validation)
  Species_NP_done <- c(Species_NP_done, Species_NP_validation)

  if(!random.best & !ambiguous.species & i==1){
    Species_training <- c(Species_HP_training,Species_NP_training)
    Labels_training <- c(rep(T,length(Species_HP_training)),rep(F,length(Species_NP_training)))
    Species_validation <- c(Species_HP_validation,Species_NP_validation)
    Labels_validation <- c(rep(T,length(Species_HP_validation)),rep(F,length(Species_NP_validation)))
    IMG.all <- SampleStrains(Species_training, Labels_training, IMG.all, "Training")
    IMG.all <- SampleStrains(Species_validation, Labels_validation, IMG.all, "Validation")
  }
}

### sample strains
if(random.best){
  SampleBestStrains()
}

if (ambiguous.species) {
  IMG.all$subset <- "selected"
}

IMG.all[, grep(pattern = "\\.orig", x = colnames(IMG.all))] <- NULL
selected <- IMG.all[IMG.all$subset == "selected",]
# only folds to download
if(k>1 & k_download<k){
  for (fold_name in fold_names[(1+k_download):length(fold_names)]) {
    selected[,fold_name] <- NULL
  }
}

# Save data for backup
if (!ambiguous.species) {
  IMG.all$Ambiguous <- NULL
}
saveRDS(IMG.all, paste0("IMG_all_folds", date.postfix.img, ".rds"))
saveRDS(selected, paste0("IMG_", k_download, "_folds", date.postfix.img, ".rds"))


if(urls){
  # Save urls for downloading
  urls.test.HP <- sapply(as.character(selected$ftp_path[selected$fold1=="test" & selected$Pathogenic]), function(f){name <- unlist(strsplit(as.character(f), split = "/")); name <- name[length(name)]; return(paste0(f, "/", name, "_genomic", download_format, ".gz"))})
  writeLines(urls.test.HP, con = paste0("urls.test.HP", download_format, ".txt"))

  urls.test.NP <- sapply(as.character(selected$ftp_path[selected$fold1=="test" & !selected$Pathogenic]), function(f){name <- unlist(strsplit(as.character(f), split = "/")); name <- name[length(name)]; return(paste0(f, "/", name, "_genomic", download_format, ".gz"))})
  writeLines(urls.test.NP, con = paste0("urls.test.NP", download_format, ".txt"))

  for (i in 1:k_download) {
    urls.train.HP <- sapply(as.character(selected$ftp_path[selected[,paste0("fold", i)]=="train" & selected$Pathogenic]), function(f){name <- unlist(strsplit(as.character(f), split = "/")); name <- name[length(name)]; return(paste0(f, "/", name, "_genomic", download_format, ".gz"))})
    urls.val.HP <- sapply(as.character(selected$ftp_path[selected[,paste0("fold", i)]=="val" & selected$Pathogenic]), function(f){name <- unlist(strsplit(as.character(f), split = "/")); name <- name[length(name)]; return(paste0(f, "/", name, "_genomic", download_format, ".gz"))})
    writeLines(urls.train.HP, con = paste0("urls.train.HP.", fold_names[i], download_format, ".txt"))
    writeLines(urls.val.HP, con = paste0("urls.val.HP.", fold_names[i], download_format, ".txt"))

    urls.train.NP <- sapply(as.character(selected$ftp_path[selected[,paste0("fold", i)]=="train" & !selected$Pathogenic]), function(f){name <- unlist(strsplit(as.character(f), split = "/")); name <- name[length(name)]; return(paste0(f, "/", name, "_genomic", download_format, ".gz"))})
    urls.val.NP <- sapply(as.character(selected$ftp_path[selected[,paste0("fold", i)]=="val" & !selected$Pathogenic]), function(f){name <- unlist(strsplit(as.character(f), split = "/")); name <- name[length(name)]; return(paste0(f, "/", name, "_genomic", download_format, ".gz"))})
    writeLines(urls.train.NP, con = paste0("urls.train.NP.", fold_names[i], download_format, ".txt"))
    writeLines(urls.val.NP, con = paste0("urls.val.NP.", fold_names[i], download_format, ".txt"))
  }

  # Save urls for downloading ALL TRAINING STRAINS
  for (i in 1:k_download) {
    urls.all.train.HP <- sapply(as.character(IMG.all$ftp_path[IMG.all[,paste0("fold", i)]=="train" & IMG.all$Pathogenic]), function(f){name <- unlist(strsplit(as.character(f), split = "/")); name <- name[length(name)]; return(paste0(f, "/", name, "_genomic", download_format, ".gz"))})
    writeLines(urls.all.train.HP, con = paste0("urls.all.train.HP.", fold_names[i], download_format, ".txt"))

    urls.all.train.NP <- sapply(as.character(IMG.all$ftp_path[IMG.all[,paste0("fold", i)]=="train" & !IMG.all$Pathogenic]), function(f){name <- unlist(strsplit(as.character(f), split = "/")); name <- name[length(name)]; return(paste0(f, "/", name, "_genomic", download_format, ".gz"))})
    writeLines(urls.all.train.NP, con = paste0("urls.all.train.NP.", fold_names[i], download_format, ".txt"))
  }
}

