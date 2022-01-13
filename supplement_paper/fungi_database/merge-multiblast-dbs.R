bac <- readRDS("multiblast/IMG_1_folds_170418_sizes.rds")
virn <- readRDS("multiblast/VHDB_1_folds_all_nhuman.rds")
virp <- readRDS("multiblast/VHDB_1_folds_human.rds")
fung <- readRDS("multiblast/TrainValTest_fungi.rds")
columns <- c("assembly_accession", "Species", "Pathogenic")
bac$Pathogenic <- ifelse(bac$Pathogenic, 1, 0)
bac$Pathogenic <- ifelse(bac$Pathogenic, 1, 0)
virn$Pathogenic <- 0
virp$Pathogenic <- 2
fung <- fung[fung$Pathogenic,]
fung$Pathogenic <- 3

all.multi <- rbind(bac[,columns], virn[,columns], virp[,columns], fung[,columns])
