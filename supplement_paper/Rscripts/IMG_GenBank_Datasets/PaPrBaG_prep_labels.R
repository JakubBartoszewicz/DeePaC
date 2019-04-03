date.postfix.img <- "_170418"

IMG_clean <- readRDS(paste0("IMG_1_folds", date.postfix.img,".rds"))

postfix.left <- ""
postfix.right <- ""


TrainingSet <- IMG_clean[IMG_clean$fold1=="train",]
TrainingLabels <- setNames(object = TrainingSet$Pathogenic, nm = paste0(TrainingSet$assembly_accession, postfix.left))
saveRDS(object = TrainingLabels, file = "TrainingLabels.rds")
TrainingStrains <- setNames(object = paste0(TrainingSet$assembly_accession, postfix.left), nm = TrainingSet$Species)
saveRDS(object = TrainingStrains, file = "TrainingStrains.rds")

TestSet <- IMG_clean[IMG_clean$fold1=="test",]

if(postfix.left == postfix.right) {
    TestLabels <- setNames(object = TestSet$Pathogenic, nm = paste0(TestSet$assembly_accession, postfix.left))
    saveRDS(object = TestLabels, file = "TestLabels.rds")
    TestStrains <- setNames(object = paste0(TestSet$assembly_accession, postfix.left), nm = TestSet$Species)
    saveRDS(object = TestStrains, file = "TestStrains.rds")
} else {
    TestLabels1 <- setNames(object = TestSet$Pathogenic, nm = paste0(TestSet$assembly_accession, postfix.left))
    TestLabels2 <- setNames(object = TestSet$Pathogenic, nm = paste0(TestSet$assembly_accession, postfix.right))
    saveRDS(object = TestLabels1, file = "TestLabels_left.rds")
    saveRDS(object = TestLabels2, file = "TestLabels_right.rds")
    TestStrains1 <- setNames(object = paste0(TestSet$assembly_accession, postfix.left), nm = TestSet$Species)
    saveRDS(object = TestStrains1, file = "TestStrains_left.rds")
    TestStrains2 <- setNames(object = paste0(TestSet$assembly_accession, postfix.right), nm = TestSet$Species)
    saveRDS(object = TestStrains2, file = "TestStrains_right.rds")
}

ValidationSet <- IMG_clean[IMG_clean$fold1=="val",]
ValidationLabels <- setNames(object = ValidationSet$Pathogenic, nm = paste0(ValidationSet$assembly_accession, postfix.left))
saveRDS(object = ValidationLabels, file = "ValidationLabels.rds")
ValidationStrains <- setNames(object = paste0(ValidationSet$assembly_accession, postfix.left), nm = ValidationSet$Species)
saveRDS(object = ValidationStrains, file = "ValidationStrains.rds")