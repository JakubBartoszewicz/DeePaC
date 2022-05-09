# ---
# Functions for simulating reads, based on simulation with Mason by Carlus Deneke (modified: NEAT, mason2, recurrent filesearch, genbank accessions, bugfixes, fragment distrib, coverage settings)
Simulate.Reads <- function(InputFastaFile = NULL, ReadCoverage = NULL, ReadLength = 250, pairedEnd = F, TargetDirectory = NULL, MeanFragmentSize = 600, FragmentStdDev = 60, ReadMargin = 10, Simulator = c("Neat", "Mason", "Mason2"), ReadNumber = NULL, Cleaned = T, AllowNsFromGenome = F) {
    # run read simulator

    # make sure you are on a linux machine
    # needs mason or NEAT + bioawk

    # MULTI-THREADING in mason (not used):
    #  When using multi-threading, each thread gets its own random number generator (RNG). The RNG of thread i is
    #  initialized with the value of --seed plus i.

    # MeanFragmentSize should be between 300 and 800 for paired end /short-insert library simulation. More than 1kb is not feasible https://www.ecseq.com/support/ngs/what-is-mate-pair-sequencing-useful-for
    # should not exceed 600 http://dnatech.genomecenter.ucdavis.edu/illumina-sequencing-all-about-libraries/

    if(!file.exists(InputFastaFile)) stop(paste("Please submit a valid input Fasta file.", InputFastaFile, "does not exist."))

    FilePrefix <- tools::file_path_sans_ext(tail(strsplit(InputFastaFile,"/")[[1]],1))
    #FilePrefix <- paste0(strsplit(FilePrefix,"_")[[1]][1:2], collapse="_")
    FileExtension <- ".fq"

    InputFileExtension <- tools::file_ext(tail(strsplit(InputFastaFile,"/")[[1]],1))

    PathPrefix <- file.path(TargetDirectory,paste0(FilePrefix))
    if (AllowNsFromGenome) {
        aNg.option <- "-aNg"
    } else {
        aNg.option <- ""
    }

    if(Simulator == "Neat"){
        OutputFile.Left <- file.path(TargetDirectory,paste0(FilePrefix,"_read1.fq"))
        OutputFile.Right <- file.path(TargetDirectory,paste0(FilePrefix,"_read2.fq"))

        if(file.exists(OutputFile.Left) & file.info(OutputFile.Left)$size > 0) {
            return(1)
        } else {
            # clean after any previous errors
            if(file.exists(OutputFile.Left)){
                file.remove(OutputFile.Left)
            }

            cat(paste0("###SIMULATING ACCESSION: ", FilePrefix, "###\n"))
            cat(paste0("###COVERAGE: ", ReadCoverage, "###\n"))

            if(!Cleaned){
                # remove contigs smaller than fragment length + create temp fasta
                tempFasta <- sub(paste0("[.]",InputFileExtension),paste0(".temp.",InputFileExtension),InputFastaFile)
                cat(paste0("###creating temp", tempFasta, "###\n"))
                if (pairedEnd){
                    system(paste("bioawk -cfastx '{if(length($seq) > ", MeanFragmentSize + 6 * FragmentStdDev + ReadMargin," ) {print \">\"$name \" \" $comment;print $seq}}'",InputFastaFile,">",tempFasta ) )
                } else {
                    system(paste("bioawk -cfastx '{if(length($seq) > ", ReadLength + ReadMargin," ) {print \">\"$name \" \" $comment;print $seq}}'",InputFastaFile,">",tempFasta ) )
                }
                InputFastaFile <- tempFasta
            }

            if(pairedEnd == F){
                cat(paste0("###UNPAIRED READS###\n"))
                # python2 genReads.py -R 250 -o out/GCA_01  -r TEST.fa -c 1 -M 0 -p 1 --rng 0 --bam
                system(paste("python2 genReads.py -R", ReadLength, "-o", PathPrefix, "-r", InputFastaFile, "-c", ReadCoverage, "-M 0 -p 1 --rng 0 --bam"))
            } else {
                cat(paste0("###PAIRED READS###\n"))
                system(paste("python2 genReads.py -R", ReadLength, "-o", PathPrefix, "-r", InputFastaFile, "-c", ReadCoverage, "--pe", MeanFragmentSize, FragmentStdDev, "-M 0 -p 1 --rng 0 --bam"))
            }

            # remove temp fasta
            if(!Cleaned){
                file.remove(tempFasta)
            }
            cat(paste0("###FINISHED ACCESSION: ", FilePrefix, "###\n"))
            return(1)
        }
    }
    else  if (Simulator == "Mason2" | Simulator == "Mason"){
        # MASON
        # output paths
        OutputFile.Left <- file.path(TargetDirectory,paste0(FilePrefix,FileExtension))
        if(pairedEnd){
            OutputFile.Base <- OutputFile.Left
            OutputFile.Left <- file.path(TargetDirectory,paste0(FilePrefix,"_l",FileExtension))
            OutputFile.Right <- file.path(TargetDirectory,paste0(FilePrefix,"_r",FileExtension))
        }
        OutputFile.BAM <- file.path(TargetDirectory,paste0(FilePrefix,".bam"))

        if(file.exists(OutputFile.Left) & file.info(OutputFile.Left)$size > 0) {
            return(1)
        } else{
            cat(paste0("###SIMULATING ACCESSION: ", FilePrefix, "###\n"))
            cat(paste0("###READ NUMBER: ", ReadNumber, "###\n"))
            # clean after any previous errors
            if(file.exists(OutputFile.Left)){
                file.remove(OutputFile.Left)
            }

            if(!Cleaned){
                # remove contigs smaller than fragment length + create temp fasta
                tempFasta <- sub(paste0("[.]",InputFileExtension),paste0(".temp.",InputFileExtension),InputFastaFile)
                if (pairedEnd | Simulator == "Mason2"){
                    system(paste("bioawk -cfastx '{if(length($seq) > ", MeanFragmentSize + 6 * FragmentStdDev + ReadMargin," ) {print \">\"$name \" \" $comment;print $seq}}'",InputFastaFile,">",tempFasta ) )
                } else {
                    system(paste("bioawk -cfastx '{if(length($seq) > ", ReadLength + ReadMargin," ) {print \">\"$name \" \" $comment;print $seq}}'",InputFastaFile,">",tempFasta ) )
                }
                InputFastaFile <- tempFasta
            }

            if(Simulator == "Mason" && InputFileExtension != "fa"){
                # change file extension to .fa for Mason v0.
                tempFasta.fa <- sub(paste0("[.]",InputFileExtension),paste0(".fa"),InputFastaFile)
                file.copy(InputFastaFile, tempFasta.fa)
                InputFastaFile <- tempFasta.fa
            }

            # delete old indices
            if(file.exists(paste0(InputFastaFile, ".fai"))){
                file.remove(paste0(InputFastaFile, ".fai"))
            }

            if(pairedEnd == F){
                cat(paste0("###UNPAIRED READS###\n"))
                if (Simulator == "Mason2"){
                    system(paste("./mason2 --illumina-read-length", ReadLength, "--fragment-mean-size", MeanFragmentSize, "--fragment-size-std-dev", FragmentStdDev, "--read-name-prefix", FilePrefix, "-oa", OutputFile.BAM, "-ir", InputFastaFile, "-n", ReadNumber, "-o", OutputFile.Left))
                } else if (Simulator == "Mason") {
                    system(paste("./mason illumina --read-length", ReadLength, "--num-reads", ReadNumber, aNg.option,"-sq -o", OutputFile.Left, InputFastaFile))
                }
            } else {
                cat(paste0("###PAIRED READS###\n"))
                if (Simulator == "Mason2"){
                    system(paste("./mason2 --illumina-read-length", ReadLength, "--fragment-mean-size", MeanFragmentSize, "--fragment-size-std-dev", FragmentStdDev, "--read-name-prefix", FilePrefix, "-oa", OutputFile.BAM, "-ir", InputFastaFile, "-n", ReadNumber, "-o", OutputFile.Left, "-or", OutputFile.Right))
                } else if (Simulator == "Mason") {
                    system(paste("./mason illumina --mate-pairs --read-length", ReadLength, "--num-reads", ceiling(ReadNumber/2), aNg.option, "-ll", MeanFragmentSize, "-le", FragmentStdDev, "-sq -o", OutputFile.Base, InputFastaFile))
                }
            }
            # remove temp fasta
            if(!Cleaned){
                file.remove(tempFasta)
            }
            if(Simulator == "Mason" && InputFileExtension != "fa"){
                file.remove(tempFasta.fa)
            }
            cat(paste0("###FINISHED ACCESSION: ", FilePrefix, "###\n"))
            return(1)
        }
    }
    else  if (Simulator == "DeepSimulator"){
        # paired end simulation not implemented yet
        if(pairedEnd){
        }
        cat(paste0("###SIMULATING ACCESSION: ", FilePrefix, "###\n"))
        cat(paste0("###READ NUMBER: ", ReadNumber, "###\n"))
        if(!Cleaned){
          # remove contigs smaller than fragment length + create temp fasta (copied from mason)
          tempFasta <- sub(paste0("[.]",InputFileExtension),paste0(".temp.",InputFileExtension),InputFastaFile)
          system(paste("bioawk -cfastx '{if(length($seq) > ", ReadLength + ReadMargin," ) {print \">\"$name \" \" $comment;print $seq}}'",InputFastaFile,">",tempFasta ))
          InputFastaFile <- tempFasta
        }
        # split fastas with multiple contigs (1 contig per file)
        # otherwise its not possible to specify the correct read number to simulate
	    temp_folder_split <- file.path(TargetDirectory,"temp")
	    if(!dir.exists(temp_folder_split)){
		    dir.create(temp_folder_split)
        }
	    fasta_folder_split <- file.path(temp_folder_split, basename(InputFastaFile),"")
	    dir.create(fasta_folder_split)
        system(paste0("bioawk -cfastx '{print \">\"$name \" \" $comment \"\\n\" $seq > (","\"",fasta_folder_split,"\"","$name\".fasta\")}' ",InputFastaFile))
	    # simulate reads (output from the contigs is combined into one fasta again)
	    output_fasta <- file.path(TargetDirectory,basename(InputFastaFile))
	    file.create(output_fasta)
	    total_seq_length <- as.numeric(system(paste("bioawk -cfastx 'BEGIN{ seq_length = 0} {seq_length += length($seq)} END{print seq_length}'", InputFastaFile  ), intern = T))
	    for(file in list.files(fasta_folder_split, full.names = TRUE)){
		    seq_length <- as.numeric(system(paste("bioawk -cfastx '{print length($seq)}'",file), intern = T))
		    read_numb <- seq_length/total_seq_length * ReadNumber
		    output <- file.path(TargetDirectory,paste0(basename(InputFastaFile),"_", basename(file) ))
	        home_dir_ds <- "/mnt/biolibs/ubuntu/DeepSimulator"
		    system(paste("./deep_simulator -i", file, "-n",round(read_numb),"-l", ReadLength, "-o", output, "-c 14", "-H", home_dir_ds, "-S 1" ))
		    system(paste("cat", file.path(output,"pass.fastq"), ">>", output_fasta))
		    # if uncommented, only the sequence is kept, other info like current signal is deleted
            #system(paste("rm -r ", output))
        }
	system(paste0("rm -r ",fasta_folder_split))
	}
}

Simulate.Reads.fromMultipleGenomes <- function(Members = NULL, TotalReadNumber = NULL, Proportional2GenomeSize = T, Fix.Coverage = F,
                                               ReadLength = 250, pairedEnd = F, FastaFileLocation = NULL, IMGdata = NULL,
                                               TargetDirectory = NULL, FastaExtension = ".fna",  MeanFragmentSize,
                                               FragmentStdDev, Workers, Simulator = c("Neat", "Mason", "Mason2"), Cleaned = T,
                                               FilenamePostfixPattern="_", ReadMargin = 10, AllowNsFromGenome = F, RelativeGenomeSizes=F) {

    if(any(c(is.null(Members), is.null(TotalReadNumber),is.null(Proportional2GenomeSize ),is.null(FastaFileLocation  ),is.null(IMGdata ),is.null(TargetDirectory ) ))) stop("Please submit valid variables to function Simulate.Reads.fromMultipleGenomes")

    dir.create(file.path(TargetDirectory), showWarnings = FALSE)

    if(Simulator == "Neat"){
        if( Proportional2GenomeSize == T){
            # number of reads per genome length = coverage
            ReadCoveragePerGenome <- rep((TotalReadNumber*ReadLength)/sum(IMGdata$Genome.Size[Members]), length(Members))
        } else {
             # number of reads per genome length = coverage
            ReadCoveragePerGenome <- ((TotalReadNumber/length(Members))*ReadLength)/ IMGdata$Genome.Size[Members]
        }

        if(Fix.Coverage == T) {
            ReadCoveragePerGenome <- rep(TotalReadNumber, length(Members))
        }
    } else  if (Simulator == "Mason2" | Simulator == "Mason" | Simulator == "DeepSimulator"){
        if( Proportional2GenomeSize == T){
            ReadNumberPerGenome <- ceiling(IMGdata$Genome.Size[Members]/sum(IMGdata$Genome.Size[Members]) * TotalReadNumber )
            if (pairedEnd){
                ReadNumberPerGenome <- sapply(ReadNumberPerGenome, function(x){max(2,x)})
            } else {
                ReadNumberPerGenome <- sapply(ReadNumberPerGenome, function(x){max(1,x)})
            }
        } else {
            ReadNumberPerGenome <- rep(round(TotalReadNumber/length(Members)), length(Members))
        }

        if(Fix.Coverage == T) {
            if(TotalReadNumber > 10) stop("When computing ReadNumber from Coverage, please limit value of TotalReadNumber to 10.")
            ReadNumberPerGenome <- ceiling(TotalReadNumber/ReadLength*IMGdata$Genome.Size[Members])
        }
    }

    # Get members' fasta file locations
    FastaFiles <- system(paste0("find ", file.path(FastaFileLocation), " -type f -name '*", FastaExtension, "'"), intern=T)
    # ignore old temp files
    FastaFiles <- FastaFiles[!grepl("\\.temp\\.", FastaFiles)]

    library(foreach)
    library(doParallel)
    registerDoParallel(Workers)
    print(paste("###Simulating using", Workers, "workers###"))

    MinFragmentSize <- MeanFragmentSize + 6 * FragmentStdDev + ReadMargin
    # Set stddev for genomes shorter than minimum
    StdDev.too.short <- 1

    if(pairedEnd){
        # If genome smaller than mean fragment size, accept smaller fragment size
        MeanFragmentSizes <- sapply(IMGdata$Genome.Size[Members], function(x){min(x - 6 * StdDev.too.short, MeanFragmentSize)})
        # If genome smaller than minimum, set stddev to zero so it gets
        FragmentStdDevs <- sapply(IMGdata$Genome.Size[Members], function(x){if(MinFragmentSize > x) StdDev.too.short else FragmentStdDev})
    } else {
        MeanFragmentSizes <- rep(MeanFragmentSize, length(Members))
        FragmentStdDevs <- rep(FragmentStdDev, length(Members))
    }

    if (RelativeGenomeSizes){
        ReadLengths <- rep(ReadLength, length(Members))
    } else {
        ReadLengths <- sapply(IMGdata$Genome.Size[Members], function(x){min(x, ReadLength)})
    }

    Check <- foreach(i = 1:length(Members) ) %dopar% {

        # Find corresponding fasta file
        CurrentFasta <- grep(paste("\\/",IMGdata$assembly_accession[Members[i]],FilenamePostfixPattern,sep=""),FastaFiles,value=T)

        if(MeanFragmentSize > MeanFragmentSizes[i] || FragmentStdDev > FragmentStdDevs[i] || ReadLength > ReadLengths[i]){
            cat(paste("###WARNING: Genome smaller than fragment/read size (", basename(CurrentFasta), ")###\n### frag size:", MeanFragmentSizes[i], "stddev:", FragmentStdDevs[i], "read len:", ReadLengths[i], "###\n"))
        }


        if(length(CurrentFasta) > 1) {
            stop(paste0("More than one match for given Accession: ", IMGdata$assembly_accession[Members[i]],": ", paste0(CurrentFasta, collapse=" "), " in ", paste(file.path(FastaFileLocation))))
        } else if(length(CurrentFasta) == 0) {
            stop(paste0("No match for given Accession: ", IMGdata$assembly_accession[Members[i]],": ", paste0(CurrentFasta, collapse=" "), " in ", paste(file.path(FastaFileLocation))))
        } else {
            if(Simulator == "Neat"){
                Simulate.Reads(CurrentFasta, ReadCoverage = ReadCoveragePerGenome[i] ,ReadLength = ReadLengths[i], pairedEnd = pairedEnd, TargetDirectory = TargetDirectory, MeanFragmentSize = MeanFragmentSizes[i], FragmentStdDev = FragmentStdDevs[i], Simulator = Simulator, Cleaned = Cleaned, AllowNsFromGenome = AllowNsFromGenome)
            } else {
                Simulate.Reads(CurrentFasta, ReadLength = ReadLengths[i], pairedEnd = pairedEnd, TargetDirectory = TargetDirectory, MeanFragmentSize = MeanFragmentSizes[i], FragmentStdDev = FragmentStdDevs[i], ReadMargin = ReadMargin, Simulator = Simulator, ReadNumber =  ReadNumberPerGenome[i], Cleaned = Cleaned, AllowNsFromGenome = AllowNsFromGenome)
            }         
        }
    }  
    
    return(unlist(Check))  
}
