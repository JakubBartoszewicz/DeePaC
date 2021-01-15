source("SimulationWrapper.R")    
    Workers <- 2
    
    Do.TrainingData <- F
    Do.ValidationData <- F
    Do.TestData <- T
    Do.Balance <- T
    Do.Balance.test <- T
    Do.GetSizes <- T
    IMG.Sizes <- F
    Do.Clean <- T
    Simulator <- "DeepSimulator"
    # Only affects Mason v0.x
    AllowNsFromGenome <- F
    
    ReadLength <- 1000
    MeanFragmentSize <- 600
    FragmentStdDev <- 60
    ReadMargin <- 10
    
    TotalTrainingReadNumber <- 2e07
    TotalValidationReadNumber <- 25e05
    TotalTestReadNumber <- 125e04
    Proportional2GenomeSize <- T
    
    pairedEnd <- F
    test.pairedEnd <- F
    
    FastaFileLocation <- "~/SCRATCH_NOBAK/read_simulation_nanopore/assemblies_raw/"
    test.FastaFileLocation <- "~/SCRATCH_NOBAK/read_simulation_nanopore/assemblies_raw/test/"
    val.FastaFileLocation <- "~/SCRATCH_NOBAK/read_simulation_nanopore/assemblies_raw/val/"
    train.FastaFileLocation <- "~/SCRATCH_NOBAK/read_simulation_nanopore/assemblies_raw/train/"
    TrainingTargetDirectory <- "/home/genskeu/SCRATCH_NOBAK/read_simulation_nanopore/simulated_reads/train"
    ValidationTargetDirectory <- "/home/genskeu/SCRATCH_NOBAK/read_simulation_nanopore/simulated_reads/val"
    TestTargetDirectory <- "/home/genskeu/SCRATCH_NOBAK/read_simulation_nanopore/simulated_reads/test_1000bp"
    
    FastaExtension <- "fna"
    FilenamePostfixPattern <- "_"
    
    HomeFolder <- "~/SCRATCH_NOBAK"
    ProjectFolder <- "read_simulation_nanopore"
    IMGFile <- "IMG_1_folds_170418.rds"
    IMGFile.new <- "IMG_1_folds_170418_new.rds"
    
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
                min.contig <- MeanFragmentSize + 6 * FragmentStdDev + ReadMargin
            } else {
                min.contig <- ReadLength + ReadMargin
            }
            status = system(paste("bioawk -cfastx '{if(length($seq) > ", min.contig," ) {print \">\"$name \" \" $comment;print $seq}}'",f,">",tempFasta ) )
    
            if(status != 0){
                cat(paste("ERROR\n"))
            }
            if (file.info(tempFasta)$size > 0){
                system(paste("cat", tempFasta, ">", f))
            } else {
                cat(paste0("WARNING: all contigs of ", basename(f), " are shorter than ", min.contig, ". Using the original file.\n"))
            }
            file.remove(tempFasta)
        }
    
        FastaFiles <- system(paste0("find ", file.path(FastaFileLocation), " -type f -name '*", FastaExtension, "'"), intern=T)
        # ignore old temp files
        FastaFiles <- FastaFiles[!grepl("\\.temp\\.", FastaFiles)]
    
        Check <- foreach(f = FastaFiles) %do% {
            cat(paste(f, "\n"))
            tempFasta <- sub(paste0("[.]",FastaExtension),paste0(".temp.",FastaExtension),f)
            # 6 std devs in NEAT
            if (test.pairedEnd){
                min.contig <- MeanFragmentSize + 6 * FragmentStdDev + ReadMargin
            } else {
                min.contig <- ReadLength + ReadMargin
            }
            status = system(paste("bioawk -cfastx '{if(length($seq) > ", min.contig," ) {print \">\"$name \" \" $comment;print $seq}}'",f,">",tempFasta ) )
    
            if(status != 0){
                cat(paste("ERROR\n"))
            }
            if (file.info(tempFasta)$size > 0){
                system(paste("cat", tempFasta, ">", f))
            } else {
                cat(paste0("WARNING: all contigs of ", basename(f), " are shorter than ", min.contig, ". Using the original file.\n"))
            }
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
    
    Simulate.Dataset <- function(SetName, ReadNumber, Proportional2GenomeSize, Fix.Coverage, ReadLength, pairedEnd, FastaFileLocation, IMGdata, TargetDirectory, MeanFragmentSize = 600, FragmentStdDev = 60, Workers = 1, Simulator = c("Neat", "Mason", "Mason2"), FastaExtension = ".fna", FilenamePostfixPattern = "_", ReadMargin = 10, AllowNsFromGenome = F){
    
        dir.create(file.path(TargetDirectory), showWarnings = FALSE)
    
        GroupMembers <- IMGdata[IMGdata$fold1 == SetName,]
    
        GroupMembers_HP <- which(GroupMembers$Pathogenic)
        GroupMembers_NP <- which(!GroupMembers$Pathogenic)
        if (length(GroupMembers_HP) > 0 ){
            Check.train_HP <- Simulate.Reads.fromMultipleGenomes (Members = GroupMembers_HP, TotalReadNumber = ReadNumber, Proportional2GenomeSize = Proportional2GenomeSize, Fix.Coverage = Fix.Coverage, ReadLength = ReadLength, pairedEnd = pairedEnd, FastaFileLocation = FastaFileLocation, IMGdata = GroupMembers, TargetDirectory = file.path(TargetDirectory, "pathogenic"), FastaExtension = FastaExtension, MeanFragmentSize = MeanFragmentSize, FragmentStdDev = FragmentStdDev, Workers = Workers, Simulator = Simulator, FilenamePostfixPattern = FilenamePostfixPattern, ReadMargin = ReadMargin, AllowNsFromGenome = AllowNsFromGenome)
        }
        if (length(GroupMembers_NP) > 0 ){
            Check.train_NP <- Simulate.Reads.fromMultipleGenomes (Members = GroupMembers_NP, TotalReadNumber = ReadNumber, Proportional2GenomeSize = Proportional2GenomeSize, Fix.Coverage = Fix.Coverage, ReadLength = ReadLength, pairedEnd = pairedEnd, FastaFileLocation = FastaFileLocation, IMGdata = GroupMembers, TargetDirectory = file.path(TargetDirectory, "nonpathogenic"), FastaExtension = FastaExtension, MeanFragmentSize = MeanFragmentSize, FragmentStdDev = FragmentStdDev, Workers = Workers, Simulator = Simulator, FilenamePostfixPattern = FilenamePostfixPattern, ReadMargin = ReadMargin, AllowNsFromGenome = AllowNsFromGenome)
        }
    }
    
    
    # Simulate test reads
    if(Do.TestData == T){
        cat("###Simulating test data...###")
        Simulate.Dataset("test", TestReadNumber, Proportional2GenomeSize, test.Fix.Coverage, ReadLength, test.pairedEnd, test.FastaFileLocation, IMGdata, TestTargetDirectory, Workers = Workers, Simulator = Simulator, FastaExtension = FastaExtension, FilenamePostfixPattern = FilenamePostfixPattern, ReadMargin = ReadMargin, AllowNsFromGenome = AllowNsFromGenome, MeanFragmentSize = MeanFragmentSize, FragmentStdDev = FragmentStdDev)
        cat("###Done!###")
    }
    
    # Simulate validation reads
    # simulate for each class
    if(Do.ValidationData == T){
        cat("###Simulating validation data...###")
        Simulate.Dataset("val", ValidationReadNumber, Proportional2GenomeSize, validation.Fix.Coverage, ReadLength, pairedEnd, val.FastaFileLocation, IMGdata, ValidationTargetDirectory, Workers = Workers, Simulator = Simulator, FastaExtension = FastaExtension, FilenamePostfixPattern = FilenamePostfixPattern, ReadMargin = ReadMargin, AllowNsFromGenome = AllowNsFromGenome, MeanFragmentSize = MeanFragmentSize, FragmentStdDev = FragmentStdDev)
        cat("###Done!###")
    }
    
    # Simulate training reads
    # simulate for each class
    if(Do.TrainingData == T){
        cat("###Simulating training data...###")
        Simulate.Dataset("train", TrainingReadNumber, Proportional2GenomeSize, training.Fix.Coverage, ReadLength, pairedEnd, train.FastaFileLocation, IMGdata, TrainingTargetDirectory, Workers = Workers, Simulator = Simulator, FastaExtension = FastaExtension, FilenamePostfixPattern = FilenamePostfixPattern, ReadMargin = ReadMargin, AllowNsFromGenome = AllowNsFromGenome, MeanFragmentSize =  MeanFragmentSize, FragmentStdDev = FragmentStdDev)
        cat("###Done!###")
    }

