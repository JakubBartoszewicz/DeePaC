# Blast comparison

# Script by Carlus Deneke, modified by Jakub Bartoszewicz

# special care for the choice of the correct parameters, the default, megablast, aims at aligning nearly identical sequences. Better choices for mismatch/gap penaalties are "task -dc-megablast" or "task -blastn"
# see http://www.ncbi.nlm.nih.gov/books/NBK279675/ and http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3848038/

require(stringr, quietly = T)
options(warn=1)
MatchBlastResults2IMG <- function(Blast,IMGdata, groundTruth=F, use.suppTable=F, suppTable=NULL, byContig=FALSE,
                                  header.pattern="\\|.*", id.as.accession=F, gb.accession.from.filename=F){

  if(groundTruth == T){
    if (byContig == F){
      query_acc <- sapply(strsplit(as.character(Blast$Query),"[/]"), function(x) tail(x,1))
      query_acc <- str_replace(string=query_acc, pattern="\\.fq.*", replacement="")
      if (gb.accession.from.filename) {
        query_acc <- sapply(strsplit(query_acc,"_"), function(x) paste(head(x,2), collapse="_"))
      }
      # map to IMG and extract species and label
      Query2IMG <- match(query_acc,as.character(IMGdata$assembly_accession))
      Query_Species <- as.character(IMGdata$Species)[Query2IMG]
      Query_Label <- IMGdata$Pathogenic[Query2IMG]
    } else {
      query_acc <- str_replace(string=as.character(Blast$Query), pattern=header.pattern, replacement="")

      if(use.suppTable){
        reptable <- read.table(suppTable)
        ids_right <- as.character(reptable$V2)
        names(ids_right) <- as.character(reptable$V1)
        query_new <- ids_right[query_acc]
        names(query_new) <- NULL
        if (!isTRUE(all.equal(query_new, query_acc))){
          query_acc <- query_new
        }
      }
      # map to IMG and extract species and label
      if (id.as.accession){
        queries <- sapply(query_acc, function(q){grepl(pattern = q, x=IMGdata$refseq.id, fixed=TRUE)})
      } else {
        queries <- sapply(query_acc, function(q){grepl(pattern = q, x=IMGdata$assembly_accession, fixed=TRUE)})
      }
      if(ncol(queries) > 1) {
        queries <- rowSums(queries) > 0
      }
      Query2IMG <- IMGdata[queries,]
      Query_Species <- Query2IMG$Species
      Query_Label <- Query2IMG$Pathogenic
    }
  }

  myReferences <- str_replace(string=as.character(Blast$Target), pattern=header.pattern, replacement="")
  if(use.suppTable){
    reptable <- read.table(suppTable)
    ids_right <- as.character(reptable$V2)
    names(ids_right) <- as.character(reptable$V1)
    myReferences <- ids_right[myReferences]
  }
  if (id.as.accession){
    myReferences_acc <- sapply(1:length(myReferences), function(x){IMGdata[grepl(pattern = myReferences[x], x = IMGdata$refseq.id, fixed = T),"assembly_accession"]})
  } else {
    myReferences_acc <- myReferences
  }

  # map to IMG and extract species and label
  Match2IMG <- match(myReferences_acc,as.character(IMGdata$assembly_accession))
  Matched_Species <- as.character(IMGdata$Species)[Match2IMG]
  Matched_Label <- IMGdata$Pathogenic[Match2IMG]

  if(groundTruth == T){
    myData <- data.frame(Reference = Blast$Target, assembly_accession = myReferences_acc, MatchedSpecies = Matched_Species, MatchedLabel = Matched_Label, QuerySpecies = Query_Species, QueryLabel = Query_Label)
  } else {
    myData <- data.frame(Reference = Blast$Target, assembly_accession = myReferences_acc, MatchedSpecies = Matched_Species, MatchedLabel = Matched_Label)
  }

  rownames(myData) <- sapply(strsplit(as.character(Blast$Query),"[/]"), function(x) tail(x,1))

  return(myData)
}

HomeFolder <- "~/SCRATCH_NOBAK/Benchmark_fungi"
ProjectFolder <-  "PathogenReads"
WorkingDirectory <- "HP_NHP"
# Choose test folder
ReadType <- "Left_250bp"
Fold <- 1
by.contig <- FALSE
# for the viral dataset
# data.header.pattern <- "^._"
# for the fungal and bacterial datasets
data.header.pattern <- "\\|.*"
# true for the viral dataset
set.id.as.accession <- FALSE
# false for the viral dataset
accession.from.filename <- TRUE

IMGdata <- readRDS(file.path(HomeFolder,ProjectFolder,"IMG_1_folds_fungi.rds") )
suppTable <- "~/SCRATCH_NOBAK/blastrepair/ids_tab"
use.suppTable <- F

Do.BuildDB <- F
Do.RunBlast <- F
Do.ProcessBlast <- T
# Switch for All Strain DB
Do.AllStrains <- F

# multi core
Cores <- 1


# load libraries functions and databases

require(foreach, quietly = T)

# Set paths

TestFolders_all <- list.dirs(file.path(HomeFolder,ProjectFolder,"OtherData",WorkingDirectory,"testData","Reads"),recursive = F, full.names = F)
TestFolders_all <- grep(ReadType,TestFolders_all,value = T)
TestReadFolder <- grep(paste("fold",Fold,sep=""),TestFolders_all, value = T)

Path2TestFiles <- file.path(HomeFolder,ProjectFolder,"OtherData",WorkingDirectory,"testData","Reads",TestReadFolder)


if(Cores > 1) {
  library(doParallel)
  registerDoParallel(Cores)
}


# --------------------------------------------------------------------------
# create blast database

# load all training genomes:

DBdir <- file.path(HomeFolder,ProjectFolder,"OtherData",WorkingDirectory,"Benchmark","Blast")
dir.create(DBdir, showWarnings=F)

if(Do.BuildDB == T){

GenomeFastas <- list.files(file.path(HomeFolder,ProjectFolder,"OtherData",WorkingDirectory,"Benchmark"), full.names = T, pattern = "fasta" )
if(Do.AllStrains == T) {
  GenomeFasta_selected <- grep("AllStrains",GenomeFastas,value = T)
  DBTitle <- "AllStrains"
  DBOutput <- file.path(DBdir,paste("AllStrains_fold",Fold, sep=""))
} else {
  GenomeFasta_selected <- grep("AllTrainingGenomes",GenomeFastas,value = T)
  DBTitle <- "AllTrainingGenomes"
  DBOutput <- file.path(DBdir,paste("AllTrainingGenomes_fold",Fold, sep=""))
}
GenomeFasta_selected <- grep(paste("fold",Fold,sep=""),GenomeFasta_selected, value=T)

if(length(GenomeFasta_selected) != 1) stop(paste("No Genomes fasta file found in", file.path(HomeFolder,ProjectFolder,"OtherData",WorkingDirectory,"Benchmark"), "for library building"))

Time <- system.time( system(paste("makeblastdb -in",GenomeFasta_selected,"-input_type fasta -dbtype nucl -title",DBTitle,"-out",DBOutput) ) )

print(paste("Blast library building took",paste(round(summary(Time),1),collapse=";"),"s"))
}



# --------------------------------------------------------------------------
# run blast


if(Do.AllStrains == T) {
  DBOutput <- file.path(DBdir,paste("AllStrains_fold",Fold, sep=""))
  MappingFolder <- "AllStrains"
} else {
  DBOutput <- file.path(DBdir,paste("AllTrainingGenomes_fold",Fold,sep=""))
  MappingFolder <- "AllTrainingGenomes"
}

dir.create(file.path(Path2TestFiles,"Blast"))

# abort if exists already
dir.create(file.path(Path2TestFiles,"Blast",MappingFolder))

# write log
LogFile <- file.path(Path2TestFiles,"Blast",MappingFolder,"Log.txt")
sink(file = file.path(Path2TestFiles,"Blast",MappingFolder,"ScreenOutput.txt"), append = T, type = "output", split = T)


if(Do.RunBlast ==T) {

  # find all read files
  ReadFiles <- list.files(Path2TestFiles,pattern="fa$|fna$",full.names = T)

  write(paste("Starting blast alignment on",Sys.time()),file = LogFile, append = F)
  print(paste("New run on",date()))

  # Options
  Options <- "-task dc-megablast" # for inter-species comparisons
  # Options <- "-task blastn" # the traditional program used for inter-species comparisons

  # loop over all read files
  StartTime <- proc.time()

  Check <-  foreach(i = 1:length(ReadFiles)) %do% {
    print(paste("Processing item",i,":",ReadFiles[i]))

    InFile <- ReadFiles[i]
    OutFile <- file.path(Path2TestFiles,"Blast",MappingFolder,sub("fa$|fna$","blast",tail(strsplit(ReadFiles[i],"[/]")[[1]],1)) )

    Time <- system.time( system(paste("blastn -outfmt 6 -max_target_seqs 1 -num_threads ",Cores, " ", Options,"-db",DBOutput,"-query",InFile,"-out",OutFile) ) )
    write(paste("Blast alignment of file",InFile,"took",paste(round(summary(Time),1),collapse=";"),"s"),file = LogFile, append = T)

    return(file.exists(OutFile))
  }

  EndTime <- proc.time()

  print(paste("Blast alignment took",(EndTime[3]-StartTime[3])/60,"min") )
  print("---Finito")

}

# ==========================================
# Blast Processing

if(Do.ProcessBlast == T) {
  # load functions
  BlastFiles <- list.files(file.path(Path2TestFiles,"Blast",MappingFolder),pattern = "blast$", full.names = T)

  StartTime <- proc.time()

  Check <- foreach(i = 1:length(BlastFiles)) %do% {

    print(paste("Processing file",i))

    Time1 <- proc.time()

    Blast <- read.table(BlastFiles[i])
    colnames(Blast) <- c("Query","Target","PercentIdentity","Alignment_length","mismatches","gap_opens","query_Start","query_End","target_Start","target_End","Evalue","BitScore")

    # remove secondary hits
    Dups <- which(duplicated(Blast$Query))
    if(length(Dups>0) )Blast <- Blast[-Dups,]

    # Match to IMG
    Blast_matched <- MatchBlastResults2IMG (Blast= Blast,IMGdata = IMGdata, T, use.suppTable, suppTable, by.contig,
                                            data.header.pattern, set.id.as.accession, accession.from.filename)

    Time <- proc.time() - Time1
    write(paste("Blast analysis of file",BlastFiles[i],"took",paste(round(summary(Time),1),collapse=";"),"s"),file = LogFile, append = T)

    saveRDS(Blast_matched,sub("[.]blast","_matched.rds",BlastFiles[i]))

    return(file.exists(sub("[.]blast","_matched.rds",BlastFiles[i]))  )

  }

  EndTime <- proc.time()

  print(paste("Blast Analysis took",(EndTime[3]-StartTime[3])/60,"min") )

}
warnings()
sink()







