set.seed(0)

bacpacstest.NP <- read.table("bacpacs_NP_test.txt", colClasses = "character")
bacpacstest.HP <- read.table("bacpacs_HP_test.txt", colClasses = "character")
bacpacstrain.NP <- read.table("bacpacs_NP.txt", colClasses = "character")
bacpacstrain.HP <- read.table("bacpacs_HP.txt", colClasses = "character")
colnames(bacpacstest.HP) <- "assembly_accession"
colnames(bacpacstest.NP) <- "assembly_accession"
colnames(bacpacstrain.HP) <- "assembly_accession"
colnames(bacpacstrain.NP) <- "assembly_accession"

bacpacstest.NP$Pathogenic <- F
bacpacstest.HP$Pathogenic <- T
bacpacstrain.NP$Pathogenic <- F
bacpacstrain.HP$Pathogenic <- T

bacpacstest.NP$fold1 <- "test"
bacpacstest.HP$fold1 <- "test"
bacpacstrain.NP$fold1 <- ""
bacpacstrain.HP$fold1 <- ""

patric <- do.call("rbind", list(bacpacstest.NP, bacpacstest.HP, bacpacstrain.NP, bacpacstrain.HP))

ids_HP_trainval <- bacpacstrain.HP$assembly_accession   
ids_NP_trainval <- bacpacstrain.NP$assembly_accession  

fold_val_size_HP <- floor(length(ids_HP_trainval)/10) + 1
fold_val_size_NP <- floor(length(ids_NP_trainval)/10) + 1

ids_HP_training <- sample(ids_HP_trainval, length(ids_HP_trainval) - fold_val_size_HP)
ids_NP_training <- sample(ids_NP_trainval, length(ids_NP_trainval) - fold_val_size_NP)
ids_HP_validation <- setdiff(ids_HP_trainval,ids_HP_training)
ids_NP_validation <- setdiff(ids_NP_trainval,ids_NP_training)

patric[patric$assembly_accession %in% ids_HP_training,"fold1"] <- "train"
patric[patric$assembly_accession %in% ids_NP_training,"fold1"] <- "train"

patric[patric$assembly_accession %in% ids_HP_validation,"fold1"] <- "val"
patric[patric$assembly_accession %in% ids_NP_validation,"fold1"] <- "val"

saveRDS(patric, "bacpacs_folddata.rds")