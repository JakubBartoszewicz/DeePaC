# ---
# Functions for simulating reads with Mason by Carlus Deneke (slightly modified: parallel mason)
Simulate.Reads <- function(InputFastaFile = NULL,ReadNumber = NULL,ReadLength = 250, pairedEnd = F,TargetDirectory = NULL, Error.handling = T) {
  # run Mason read simulator
  
  # make sure you are on a linux machine
  # needs mason + bioawk
  
  if(!file.exists(InputFastaFile)) stop("Please submit a valid input Fasta file")
  # if(pairedEnd == T) stop("Option pairedEnd not implemented yet")
  
  OutputFile <- file.path(TargetDirectory,tail(strsplit(InputFastaFile,"/")[[1]],1) )
  logFile <- file.path(TargetDirectory,"log.txt")           
  
  sink(file= logFile,append = T)
  
  if(pairedEnd == F){
    system(paste("./mason illumina --read-length", ReadLength ,"--num-reads",max(2,ReadNumber)," -o", OutputFile,InputFastaFile,">>",logFile) )    
    } else {
    system(paste("./mason illumina --mate-pairs --read-length", ReadLength ,"--num-reads",max(2,ceiling(ReadNumber/2))," -o", OutputFile,InputFastaFile,">>",logFile) )  
    OutputFile <- sub("[.]fasta","_1.fasta",OutputFile)
    }
  
  # check if file.exists
  if(file.exists(OutputFile) ) {sink(); return(1)} else if(Error.handling == T ){
    # else error routine: 
    # remove contigs smaller than read length + create temp fasta
    tempFasta <- sub("[.]fasta","_temp.fasta",InputFastaFile)
    system(paste("bioawk -cfastx '{if(length($seq) > ",ReadLength + 5," ) {print \">\"$name $comment;print $seq}}'",InputFastaFile,">",tempFasta ) )

    # simulate reads from temp fasta
    if(pairedEnd == F){
    system(paste("./mason illumina --read-length", ReadLength ,"--num-reads",max(2,ReadNumber)," -o", OutputFile,tempFasta,">>",logFile) )
    } else {
      system(paste("./mason illumina --mate-pairs --read-length", ReadLength ,"--num-reads",max(2,ReadNumber)," -o", OutputFile,tempFasta,">>",logFile) )  
      OutputFile <- sub("[.]fasta","_1.fasta",OutputFile)
    }
    # remove temp fasta
    file.remove(tempFasta)
    
    # keep info in log
    print(paste("Reads for File",InputFastaFile,"could not be simulated.","\n","Removing short contigs could",ifelse(file.exists(OutputFile),"fix","not fix"),"the problem."))  
  
    sink()
      
    if(file.exists(OutputFile) ) return(2) else return(0)
  } else {sink();return(0)}

}    

Simulate.Reads.fromMultipleGenomes <- function(Members = NULL, TotalReadNumber = NULL, Proportional2GenomeSize = T, Fix.Coverage = F, ReadLength = 250, pairedEnd = F, FastaFileLocation = NULL, IMGdata = NULL, TargetDirectory = NULL,Error.handling = T) {
  
  if(any(c(is.null(Members),  is.null(TotalReadNumber),is.null(Proportional2GenomeSize ),is.null(FastaFileLocation  ),is.null(IMGdata ),is.null(TargetDirectory ) ))) stop("Please submit valid variables to function Simulate.Reads.fromMultipleGenomes")
  
  if( Proportional2GenomeSize == T){
    ReadNumberPerGenome <- ceiling(IMGdata$Genome.Size[Members]/sum(IMGdata$Genome.Size[Members]) * TotalReadNumber )  
  } else {
    ReadNumberPerGenome <- round(TotalReadNumber/length(Members) )
  }
    
  if(Fix.Coverage == T) {
    if(TotalReadNumber > 10) stop("When computing ReadNumber from Coverage, please limit value of TotalReadNumber to 10.")
    ReadNumberPerGenome <- ceiling(TotalReadNumber/ReadLength*IMGdata$Genome.Size[Members])
  }
  
  # Get members' fasta file locations  
  FastaFiles <- list.files(FastaFileLocation,full.names = T,pattern="fasta$")
  
  library(foreach)  
  Check <- foreach(i = 1:length(Members) ) %dopar% {
    
    # Find corresponding fasta file
    CurrentFasta <- grep(paste(IMGdata$Bioproject.Accession[Members[i] ],".",sep=""),FastaFiles,value=T,fixed=T)
    if(length(CurrentFasta) > 1) {
      stop("More than one match for given Bioproject Accession")
    } else if(length(CurrentFasta) == 0) {
      return(0)
    } else {
      Simulate.Reads(CurrentFasta,ReadNumber = ReadNumberPerGenome[i] ,ReadLength = ReadLength, pairedEnd = pairedEnd, TargetDirectory = TargetDirectory, Error.handling = Error.handling)      
    }
    
   # print(paste("Simulating with parameters",FastaFiles[i],ReadNumberPerGenome[i] ,ReadLength,pairedEnd,TargetDirectory))
  }  
  
  return(unlist(Check))
  
}
