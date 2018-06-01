source("MasonWrapper.R")

Do.TrainingData <- F
Do.TestData <- T
HomeFolder <- "~/my_wiss"
ProjectFolder <- "data/Folddata"
IMGFile <- "IMG_all_genomes_040615_taxontable6678_filtered_ruled_curated_HPNHPonly.rds"
TrainingStrainsFile <- "TrainingStrains_fold1.rds"
TestStrainsFile <- "TestStrains_fold1.rds"
TrainingLabelsFile <- "TrainingLabels_fold1.rds"
TestLabelsFile <- "TestLabels_fold1.rds"

IMGdata <- readRDS(file.path(HomeFolder,ProjectFolder,IMGFile))
if(grepl("curated",IMGFile)) IMGdata$Genome.Size. <- as.numeric(IMGdata$Genome.Size.) else IMGdata$Genome.Size. <- as.numeric(IMGdata$Genome.Size)

TrainingReadNumber = 1e07 # per class across all genomes
TestReadNumber = 5e05 # per class across all genomes
training.Fix.Coverage = F
test.Fix.Coverage = F
ReadLength = 250
pairedEnd = F
FastaFileLocation <- "~/SCRATCH_NOBAK/IMGdata"
TrainingTargetDirectory <- "~/SCRATCH_NOBAK/trainingReads"
TestTargetDirectory <- "~/SCRATCH_NOBAK/testReads"

# Simulate training reads
# simulate for each class
if(Do.TrainingData == T){
  RandomTrainingStrain_Bioproject.Accession <- readRDS(file.path(HomeFolder,ProjectFolder,TrainingStrainsFile))
  GroupMembers_training <- match(RandomTrainingStrain_Bioproject.Accession,IMGdata$Bioproject.Accession)
  Labels_training <- readRDS(file.path(HomeFolder,ProjectFolder,TrainingLabelsFile))
  
  GroupMembers_training_HP <-  GroupMembers_training[IMGdata$Pathogenic[GroupMembers_training] == T]
  GroupMembers_training_NP <-  GroupMembers_training[IMGdata$Pathogenic[GroupMembers_training] == F]
  Check.train_HP <- Simulate.Reads.fromMultipleGenomes (Members = GroupMembers_training_HP, TotalReadNumber = TrainingReadNumber, Proportional2GenomeSize = T, Fix.Coverage = training.Fix.Coverage, ReadLength = ReadLength, pairedEnd = pairedEnd, FastaFileLocation = FastaFileLocation, IMGdata = IMGdata, TargetDirectory = TrainingTargetDirectory)
  Check.train_NP <- Simulate.Reads.fromMultipleGenomes (Members = GroupMembers_training_NP, TotalReadNumber = TrainingReadNumber, Proportional2GenomeSize = T, Fix.Coverage = training.Fix.Coverage, ReadLength = ReadLength, pairedEnd = pairedEnd, FastaFileLocation = FastaFileLocation, IMGdata = IMGdata, TargetDirectory = TrainingTargetDirectory)  
}

# Simulate test reads
if(Do.TestData == T){
  RandomTestStrain_Bioproject.Accession <- readRDS(file.path(HomeFolder,ProjectFolder,TestStrainsFile))
  GroupMembers_test <- match(RandomTestStrain_Bioproject.Accession,IMGdata$Bioproject.Accession)
  Labels_test <- readRDS(file.path(HomeFolder,ProjectFolder,TestLabelsFile))

  GroupMembers_test_HP <-  GroupMembers_test[IMGdata$Pathogenic[GroupMembers_test] == T]
  GroupMembers_test_NP <-  GroupMembers_test[IMGdata$Pathogenic[GroupMembers_test] == F]
  Check.test_HP <- Simulate.Reads.fromMultipleGenomes (Members = GroupMembers_test_HP, TotalReadNumber = TestReadNumber, Proportional2GenomeSize = T, Fix.Coverage = test.Fix.Coverage, ReadLength = ReadLength, pairedEnd = pairedEnd, FastaFileLocation = FastaFileLocation, IMGdata = IMGdata, TargetDirectory = TestTargetDirectory)
  Check.test_NP <- Simulate.Reads.fromMultipleGenomes (Members = GroupMembers_test_NP, TotalReadNumber = TestReadNumber, Proportional2GenomeSize = T, Fix.Coverage = test.Fix.Coverage, ReadLength = ReadLength, pairedEnd = pairedEnd, FastaFileLocation = FastaFileLocation, IMGdata = IMGdata, TargetDirectory = TestTargetDirectory)
}