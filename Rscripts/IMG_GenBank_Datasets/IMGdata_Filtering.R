# read in full data
date.postfix.img <- "_160119"
IMGdata_df <- readRDS(file.path(paste0("IMG", date.postfix.img, ".rds")))
include.drafts <- TRUE

if(include.drafts){
    Sequencing.Status.pattern <- "Finished|Permanent Draft|Draft"
} else {
    Sequencing.Status.pattern <- "Finished|Permanent Draft"
}

# Initial filtering
FilterRules1 <- intersect(grep("Bacteria",IMGdata_df$Domain,ignore.case = T),
                          intersect(grep(Sequencing.Status.pattern,IMGdata_df$Sequencing.Status,ignore.case = T),
                                    grep("Genome Analysis",IMGdata_df$JGI.Analysis.Project.Type,ignore.case = T) ))
IMGdata_filtered <- IMGdata_df[FilterRules1,]

# remove data without ncbu project accession
IMGdata_filtered <- IMGdata_filtered[-which(IMGdata_filtered$NCBI.Bioproject.Accession == ""),]
# remove unclassified species
IMGdata_filtered <- IMGdata_filtered[-which(IMGdata_filtered$Species == "unclassified"),]

Duplicated.Accessions <- unique(IMGdata_filtered$NCBI.Bioproject.Accession[which(duplicated(IMGdata_filtered$NCBI.Bioproject.Accession))])

require(foreach)
Rows.2remove  <- foreach(i = 1:length(Duplicated.Accessions)) %do% {
  
  # find all entries
  Current.Dups <- which(IMGdata_filtered$NCBI.Bioproject.Accession == Duplicated.Accessions[i])
  
  # find newest entry
  # -- a 'factor vs. character' bug was here
  Add.Date <- sapply(IMGdata_filtered$Add.Date[Current.Dups], as.character)
  Current.Dups.2remove <- Current.Dups[-tail(order(sapply(strsplit(Add.Date,"[.]"), function(x) paste(x[3],x[2],x[1],sep = ""))),1)]
  
  # return row number to remove
  return(Current.Dups.2remove)

}

# remove duplicates
IMGdata_filtered <- IMGdata_filtered[-unlist(Rows.2remove),]


# ---
# Set of rules for pathogenicity

# 1a)Check for occurence of "Non-Pathogen" in Phenotype
NPs <- grep("Non-Pathogen",IMGdata_filtered$Phenotype,value=F,ignore.case = T)

# 1b)Check for occurence of "Pathogen" in Phenotype
P_temp <- grep("Pathogen",IMGdata_filtered$Phenotype,value=F,ignore.case = T)
Ps1 <- setdiff(P_temp,NPs)

# 2)Check for occurence of "Pathogen" in Relevance
Ps2 <- grep("Pathogen",IMGdata_filtered$Relevance,value=F,ignore.case = T)

# 3)Check for non-empty entries in Disease
Ps3 <- which(IMGdata_filtered$Diseases != "")
Ps <- union(union(Ps1,Ps2),Ps3)

# -----------------
# Host

# 1) Occurrence of Human in Host.name
H1 <- grep("^Human|Homo sapiens",IMGdata_filtered$Host.Name,value=F,ignore.case = T)

# 2) Occurrence of Human in Eco.system Category
H2 <- grep("^Human|Homo sapiens",IMGdata_filtered$Ecosystem.Category,value=F,ignore.case = T)

# 3) Occurrence of Human in Habitat
H3 <- grep("Human|Homo sapiens",IMGdata_filtered$Habitat,value=F,ignore.case = T)

# 4) Part of the Human Microbiome Project
H4 <- grep("HMP",IMGdata_filtered$Study.Name,value=F,ignore.case = T)

Hs <- union(union(union(H1,H2),H3),H4)

# ----
# Add new metadata

# New column Conflicts
IMGdata_filtered$Conflicts <- rep(NA,nrow(IMGdata_filtered) )

# New column Pathogenic
IMGdata_filtered$Pathogenic <- rep(NA,nrow(IMGdata_filtered) )

IMGdata_filtered$Pathogenic[NPs] <- F
IMGdata_filtered$Pathogenic[Ps] <- T
# remove conflicts
IMGdata_filtered$Conflicts[intersect(Ps,NPs)] <- "Ambiguous label"
IMGdata_filtered$Pathogenic[intersect(Ps,NPs)] <- NA

# New column HumanHost
IMGdata_filtered$HumanHost <- rep(F,nrow(IMGdata_filtered) )
IMGdata_filtered$HumanHost[Hs] <- T

# New column Rules
IMGdata_filtered$Rules <- rep(NA,nrow(IMGdata_filtered) )

IMGdata_filtered$Rules[H1] <- ifelse(is.na(IMGdata_filtered$Rules[H1]),"H1",paste(IMGdata_filtered$Rules[H1],"H1",sep=","))
IMGdata_filtered$Rules[H2] <- ifelse(is.na(IMGdata_filtered$Rules[H2]),"H2",paste(IMGdata_filtered$Rules[H2],"H2",sep=","))
IMGdata_filtered$Rules[H3] <- ifelse(is.na(IMGdata_filtered$Rules[H3]),"H3",paste(IMGdata_filtered$Rules[H3],"H3",sep=","))
IMGdata_filtered$Rules[H4] <- ifelse(is.na(IMGdata_filtered$Rules[H4]),"H4",paste(IMGdata_filtered$Rules[H4],"H4",sep=","))

IMGdata_filtered$Rules[NPs] <- ifelse(is.na(IMGdata_filtered$Rules[NPs]),"NPs",paste(IMGdata_filtered$Rules[NPs],"NPs",sep=","))  
IMGdata_filtered$Rules[Ps1] <- ifelse(is.na(IMGdata_filtered$Rules[Ps1]),"Ps1",paste(IMGdata_filtered$Rules[Ps1],"Ps1",sep=","))
IMGdata_filtered$Rules[Ps2] <- ifelse(is.na(IMGdata_filtered$Rules[Ps2]),"Ps2",paste(IMGdata_filtered$Rules[Ps2],"Ps2",sep=","))
IMGdata_filtered$Rules[Ps3] <- ifelse(is.na(IMGdata_filtered$Rules[Ps3]),"Ps3",paste(IMGdata_filtered$Rules[Ps3],"Ps3",sep=","))

# save results

saveRDS(IMGdata_filtered ,file.path(paste0("IMG", date.postfix.img, "_filtered.rds")))

IMGdata_filtered_HPNHP  <- IMGdata_filtered[!is.na(IMGdata_filtered$Pathogenic) & IMGdata_filtered$HumanHost == T,]

# Annotate species with label conflict and HUMAN HOST

Species_HP <- unique(IMGdata_filtered_HPNHP$Species[which(IMGdata_filtered_HPNHP$Pathogenic == T)])
Species_NHP <- unique(IMGdata_filtered_HPNHP$Species[which(IMGdata_filtered_HPNHP$Pathogenic == F)])
Species_HNA <- unique(IMGdata_filtered_HPNHP$Species[which(is.na(IMGdata_filtered_HPNHP$Pathogenic))])

Species_HPNHP_conflict <- intersect(Species_HP, Species_NHP)

Check <- foreach(Current.Species = Species_HPNHP_conflict) %do% {

  Current.Rows <- which(IMGdata_filtered_HPNHP$Species == Current.Species)
  
  IMGdata_filtered_HPNHP$Conflicts[Current.Rows] <- ifelse(is.na(IMGdata_filtered_HPNHP$Conflicts[Current.Rows]),
         "HLabelConflict",
         paste(IMGdata_filtered_HPNHP$Conflicts[Current.Rows],"HLabelConflict",sep=","))
  
  return(1) 
}

saveRDS(IMGdata_filtered_HPNHP,file.path(paste0("IMG", date.postfix.img, "_filtered_HPNHP.rds")))

table(Pathogenic = IMGdata_filtered_HPNHP$Pathogenic,HumanHost = IMGdata_filtered_HPNHP$HumanHost)
print(c(HP.all = length(Species_HP), HP.pure = length(setdiff(Species_HP, Species_HPNHP_conflict)), HNP = length(setdiff(Species_NHP, Species_HPNHP_conflict)), HMix = length(Species_HPNHP_conflict), HNA = length(Species_HNA), Sum = length(unique(IMGdata_filtered_HPNHP$Species))))


######################## Manual Inspection of Conflicting Labels #############################

IMGdata_filtered_HPNHP_conflicts <- IMGdata_filtered_HPNHP[grep("HLabelConflict",IMGdata_filtered_HPNHP$Conflicts),]
saveRDS(IMGdata_filtered_HPNHP_conflicts,file.path(paste0("IMG", date.postfix.img, "_filtered_HPNHP_conflicts.rds")))
table(Species = droplevels(IMGdata_filtered_HPNHP_conflicts$Species), Pathogenic = IMGdata_filtered_HPNHP_conflicts$Pathogenic)
IMGdata_filtered_HPNHP_resolved <- IMGdata_filtered_HPNHP[-which(IMGdata_filtered_HPNHP$Species %in% Species_HPNHP_conflict & IMGdata_filtered_HPNHP$Pathogenic == F),]
table(Pathogenic = IMGdata_filtered_HPNHP_resolved$Pathogenic,HumanHost = IMGdata_filtered_HPNHP_resolved$HumanHost)
saveRDS(IMGdata_filtered_HPNHP_resolved,file.path(paste0("IMG", date.postfix.img, "_filtered_HPNHP_resolved.rds")))

######################## Compare with old data                  ##############################
#IMGdata_old <- readRDS("IMG_all_genomes_170418_taxontable128751_filtered_ruled_HPNHP_resolved_inc.rds")
#old_species <- sort(setdiff(gsub('\\[|\\]','',unique(IMGdata_old$Species)), gsub('\\[|\\]','',unique(IMGdata_filtered_HPNHP_resolved$Species))))
#new_species <- sort(setdiff(gsub('\\[|\\]','',unique(IMGdata_filtered_HPNHP_resolved$Species)), gsub('\\[|\\]','',unique(IMGdata_old$Species))))

######################## Check genomes not flagged as Complete ###############################
incomplete <- IMGdata_filtered_HPNHP_resolved[! grepl("Complete",IMGdata_filtered_HPNHP_resolved$Seq.Status,ignore.case = T),]