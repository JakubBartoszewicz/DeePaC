source("SimulationWrapper.R")

Workers <- 72

Do.TrainingData <- T
Do.ValidationData <- T
Do.TestData <- T
Do.Balance <- T
Do.Balance.test <- T
Do.GetSizes <- T
IMG.Sizes <- F
Do.Clean <- F
Cleaned <- T
Simulator <- "Mason"

ReadLength <- 250
MeanFragmentSize <- 600
FragmentStdDev <- 60
ReadMargin <- 10

TotalTrainingReadNumber <- 2e07
TotalValidationReadNumber <- 25e05
TotalTestReadNumber <- 25e05
Proportional2GenomeSize <- T

pairedEnd <- F
test.pairedEnd <- T

FastaFileLocation <- "~/SCRATCH_NOBAK/vhdb_assembled"
test.FastaFileLocation <- "~/SCRATCH_NOBAK/vhdb_test"
TrainingTargetDirectory <- "~/SCRATCH_NOBAK/CHORDATA/trainingReads"
ValidationTargetDirectory <- "~/SCRATCH_NOBAK/CHORDATA/validationReads"
TestTargetDirectory <- "~/SCRATCH_NOBAK/CHORDATA/testReads"
FastaExtension <- "fa"
FilenamePostfixPattern <- "\\."

HomeFolder <- "~/"
ProjectFolder <- "folddata/"
IMGFile <- "VHDB_1_folds_chordata_nhuman.rds"
IMGFile.new <- "VHDB_1_folds_chordata_nhuman_sizes.rds"

if (Do.Clean){

    FastaFiles <- system(paste0("find ", file.path(FastaFileLocation), " -type f -name '*", FastaExtension, "'"), intern=T)
    # ignore old temp files
    FastaFiles <- FastaFiles[!grepl("\\.temp\\.", FastaFiles)]
    
    library(foreach)  
    cat(paste("###Cleaning###\n"))
    
    Check <- foreach(f = FastaFiles) %do% { 
        cat(paste(f, "\n"))
        tempFasta <- sub(paste0("[.]",FastaExtension),paste0(".temp.",FastaExtension),f)
        # 6 std devs in NEAT
        if (pairedEnd){
            status = system(paste("bioawk -cfastx '{if(length($seq) > ", MeanFragmentSize + 6 * FragmentStdDev + ReadMargin," ) {print \">\"$name \" \" $comment;print $seq}}'",f,">",tempFasta ) )
        } else {
            status = system(paste("bioawk -cfastx '{if(length($seq) > ", ReadLength + ReadMargin," ) {print \">\"$name \" \" $comment;print $seq}}'",f,">",tempFasta ) )
        }
        if(status != 0){
            cat(paste("ERROR\n"))
        }
        system(paste("cat", tempFasta, ">", f))
        file.remove(tempFasta)    
    }
    
    test.FastaFiles <- system(paste0("find ", file.path(test.FastaFileLocation), " -type f -name '*", FastaExtension, "'"), intern=T)
    # ignore old temp files
    test.FastaFiles <- test.FastaFiles[!grepl("\\.temp\\.", test.FastaFiles)]
    
    Check <- foreach(f = test.FastaFiles) %do% {
        cat(paste(f, "\n"))
        tempFasta <- sub(paste0("[.]",FastaExtension),paste0(".temp.",FastaExtension),f) 
        # 6 std devs in NEAT
        if (test.pairedEnd){
            status = system(paste("bioawk -cfastx '{if(length($seq) > ", MeanFragmentSize + 6 * FragmentStdDev + ReadMargin," ) {print \">\"$name \" \" $comment;print $seq}}'",f,">",tempFasta ) )
        } else {
            status = system(paste("bioawk -cfastx '{if(length($seq) > ", ReadLength + ReadMargin," ) {print \">\"$name \" \" $comment;print $seq}}'",f,">",tempFasta ) )
        }
        if(status != 0){
            cat(paste("ERROR\n"))
        }
        system(paste("cat", tempFasta, ">", f))
        file.remove(tempFasta)    
    }
    cat(paste("###Cleaning done###\n"))
}
 
if (Do.GetSizes) {
    IMGdata <- readRDS(file.path(HomeFolder,ProjectFolder,IMGFile))
    calcSize <- function(x){
        if (x$fold1 == "test"){
            file.loc <- test.FastaFileLocation
        }
        else{
            file.loc <- FastaFileLocation
        }
        return (as.numeric(system(paste0("find ", file.loc, " -type f -name '", x$assembly_accession, "*' | xargs grep -v \">\" | wc | awk '{print $3-$1}'"), intern=T)))
    }

    IMGdata$Genome.Size <- sapply(1:nrow(IMGdata), function(i){calcSize(IMGdata[i,c("assembly_accession", "fold1")])})
    saveRDS(IMGdata, file.path(HomeFolder,ProjectFolder,IMGFile.new))
} else {
    IMGdata <- readRDS(file.path(HomeFolder,ProjectFolder,IMGFile))
}

if (IMG.Sizes) {
    IMGdata$Genome.Size <- as.numeric(IMGdata$Genome.Size.....assembled)
}

if (Do.Balance) {
    TrainingReadNumber <- TotalTrainingReadNumber / 2 # per class across all genomes
    ValidationReadNumber <- TotalValidationReadNumber / 2 # per class across all genomes
    training.Fix.Coverage <- F
    validation.Fix.Coverage <- F
} else {
    TrainingReadNumber <- TotalTrainingReadNumber * ReadLength / sum(IMGdata$Genome.Size[IMGdata$fold1 == "train"]) # coverage
    ValidationReadNumber = TotalValidationReadNumber * ReadLength / sum(IMGdata$Genome.Size[IMGdata$fold1 == "val"]) # coverage
    training.Fix.Coverage <- T
    validation.Fix.Coverage <- T
}
if (Do.Balance.test) {
    TestReadNumber <- TotalTestReadNumber / 2 # per class across all genomes
    test.Fix.Coverage <- F
} else {
    TestReadNumber = TotalTestReadNumber * ReadLength / sum(IMGdata$Genome.Size[IMGdata$fold1 == "test"]) # coverage
    test.Fix.Coverage <- T
}

Simulate.Dataset <- function(SetName, ReadNumber, Proportional2GenomeSize, Fix.Coverage, ReadLength, pairedEnd, FastaFileLocation, IMGdata, TargetDirectory, MeanFragmentSize = 600, FragmentStdDev = 60, Workers = 1, Simulator = c("Neat", "Mason", "Mason2"), Cleaned = T, FastaExtension = ".fna", FilenamePostfixPattern = "_"){
      
    dir.create(file.path(TargetDirectory), showWarnings = FALSE)
    
    GroupMembers <- IMGdata[IMGdata$fold1 == SetName,]
    
    GroupMembers_HP <- which(GroupMembers$Pathogenic)
    GroupMembers_NP <- which(!GroupMembers$Pathogenic)
    if (length(GroupMembers_HP) > 0 ){
        Check.train_HP <- Simulate.Reads.fromMultipleGenomes (Members = GroupMembers_HP, TotalReadNumber = ReadNumber, Proportional2GenomeSize = Proportional2GenomeSize, Fix.Coverage = Fix.Coverage, ReadLength = ReadLength, pairedEnd = pairedEnd, FastaFileLocation = FastaFileLocation, IMGdata = GroupMembers, TargetDirectory = file.path(TargetDirectory, "pathogenic"), FastaExtension = FastaExtension, MeanFragmentSize = MeanFragmentSize, FragmentStdDev = FragmentStdDev, Workers = Workers, Simulator = Simulator, Cleaned = Cleaned, FilenamePostfixPattern = FilenamePostfixPattern)
    }
    if (length(GroupMembers_NP) > 0 ){
        Check.train_NP <- Simulate.Reads.fromMultipleGenomes (Members = GroupMembers_NP, TotalReadNumber = ReadNumber, Proportional2GenomeSize = Proportional2GenomeSize, Fix.Coverage = Fix.Coverage, ReadLength = ReadLength, pairedEnd = pairedEnd, FastaFileLocation = FastaFileLocation, IMGdata = GroupMembers, TargetDirectory = file.path(TargetDirectory, "nonpathogenic"), FastaExtension = FastaExtension, MeanFragmentSize = MeanFragmentSize, FragmentStdDev = FragmentStdDev, Workers = Workers, Simulator = Simulator, Cleaned = Cleaned, FilenamePostfixPattern = FilenamePostfixPattern)
    }
}


# Simulate test reads
if(Do.TestData == T){
    cat("###Simulating test data...###")
    Simulate.Dataset("test", TestReadNumber, Proportional2GenomeSize, test.Fix.Coverage, ReadLength, test.pairedEnd, test.FastaFileLocation, IMGdata, TestTargetDirectory, Workers = Workers, Simulator = Simulator, Cleaned = Cleaned, FastaExtension = FastaExtension, FilenamePostfixPattern = FilenamePostfixPattern)
    cat("###Done!###")
}

# Simulate validation reads
# simulate for each class
if(Do.ValidationData == T){
    cat("###Simulating validation data...###")
    Simulate.Dataset("val", ValidationReadNumber, Proportional2GenomeSize, validation.Fix.Coverage, ReadLength, pairedEnd, FastaFileLocation, IMGdata, ValidationTargetDirectory, Workers = Workers, Simulator = Simulator, Cleaned = Cleaned, FastaExtension = FastaExtension, FilenamePostfixPattern = FilenamePostfixPattern)
    cat("###Done!###")
}

# Simulate training reads
# simulate for each class
if(Do.TrainingData == T){
    cat("###Simulating training data...###")
    Simulate.Dataset("train", TrainingReadNumber, Proportional2GenomeSize, training.Fix.Coverage, ReadLength, pairedEnd, FastaFileLocation, IMGdata, TrainingTargetDirectory, Workers = Workers, Simulator = Simulator, Cleaned = Cleaned, FastaExtension = FastaExtension, FilenamePostfixPattern = FilenamePostfixPattern)
    cat("###Done!###")
}
