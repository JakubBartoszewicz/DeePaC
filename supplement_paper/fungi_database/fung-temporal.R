library(plyr)
source("FUNGI_DATA_CUR/src/fungi-fctions.R")

all.dat <- readRDS("FUNGI_DATA_CUR/release_selected/all_data.rds")

genbank <- read.csv("FUNGI_DATA_CUR/src/genbank_211009/assembly_summary_ed_211009.txt", sep="\t", stringsAsFactors = F)
genbank$seq_rel_date <- as.Date(genbank$seq_rel_date)
genbank_cl <- genbank[genbank$refseq_category!="na",]
genbank_cl <- genbank_cl[,c("assembly_accession", "refseq_category", "taxid", "species_taxid", "organism_name", "infraspecific_name", "assembly_level", "seq_rel_date", "asm_name", "ftp_path")]

genbank_n <- read.csv("FUNGI_DATA_CUR/TEMPORAL/src/genbank_220102/assembly_summary_ed_220102.txt", sep="\t", stringsAsFactors = F)
genbank_n$seq_rel_date <- as.Date(genbank_n$seq_rel_date)
genbank_ncl <- genbank_n[genbank_n$refseq_category!="na",]
genbank_ncl <- genbank_ncl[,c("assembly_accession", "refseq_category", "taxid", "species_taxid", "organism_name", "infraspecific_name", "assembly_level", "seq_rel_date", "asm_name", "ftp_path")]

new_acc <- genbank_ncl[!(genbank_ncl$assembly_accession %in% genbank_cl$assembly_accession),]
new_sp <- new_acc[!(new_acc$species_taxid %in% genbank_cl$species_taxid),]
new_sp_filtered <- filter_species_names(new_sp, species_column = "organism_name")

new_sp_metadata <- merge(new_sp_filtered, all.dat, by = "species_taxid", all.x = T, suffixes = c("",".OLD"))
new_sp_metadata[,11:19] <- NULL
new_sp_metadata$Species <- new_sp_metadata$organism_name
new_sp_metadata$source.name <- new_sp_metadata$organism_name
new_sp_metadata$source.taxid <- new_sp_metadata$taxid
# for easier search in grin, remove brackets and annotations regarding adherence to the specific Code of Nomenclature
new_sp_metadata$Species <- gsub("\\[|\\]|", "", x = new_sp_metadata$Species)
new_sp_metadata$Species <- gsub(" \\(nom\\. inval\\.\\)", "", x = new_sp_metadata$Species)
new_sp_metadata$Species <- gsub(" \\(nom\\. nud\\.\\)", "", x = new_sp_metadata$Species)

write.csv(x=new_sp_metadata[,c("organism_name", "species_taxid", "Species")], file = "FUNGI_DATA_CUR/TEMPORAL/src/genbank_new.csv", row.names = FALSE)
write.csv(x=new_sp_metadata[!is.na(new_sp_metadata$putative.human.host),c("organism_name", "species_taxid", "Species")], file = "FUNGI_DATA_CUR/TEMPORAL/src/genbank_new_put_human.csv", row.names = FALSE)

# none of the putative human pathogens confirmed in the Atlas. A. montevidensis clearly annotated as such in GRIN, and confirmed in the Atlas.

pureatlas.citation <- 'de Hoog GS, Guarro J, Gené J, Ahmed S, Al-Hatmi AMS, Figueras MJ & Vitale RG (2020) Atlas of Clinical Fungi, 4th edition. Hilversum.'
puregrin_new.citation <- 'Farr, D.F., & Rossman, A.Y. Fungal Databases, U.S. National Fungus Collections, ARS, USDA. Retrieved January 2, 2022, from https://nt.ars-grin.gov/fungaldatabases/'
n_grin_manual <- read.csv("FUNGI_DATA_CUR/TEMPORAL/src/GRIN_manual_220102/grin_temporal.csv", stringsAsFactors = F)
colnames(n_grin_manual)[c(1,6,7,8)] <- c("source.name", "plant.host.grin", "animal.host.grin", "human.host.grin")
n_grin_manual$source.taxid <- ""
n_grin_manual$human.pathogen <- NA
n_grin_manual$animal.pathogen <- NA
n_grin_manual$plant.pathogen <- NA
n_grin_manual$plant.host <- NA
n_grin_manual$putative.human.host <- NA
n_grin_manual$putative.animal.host <- NA
n_grin_manual$putative.plant.host <- NA
n_grin_manual$human.pathogen.source <- "" 
n_grin_manual$animal.pathogen.source <- ""
n_grin_manual$plant.pathogen.source <- ""
n_grin_manual$plant.host.source <- ""
n_grin_manual$putative.human.host.source <- ""
n_grin_manual$putative.animal.host.source <- ""
n_grin_manual$putative.plant.host.source <- ""


# extract the appropriate columns for species with a specified disease and a plant host
n_grin_manual_pl <- n_grin_manual[n_grin_manual$disease!="" & !is.na(n_grin_manual$plant.host.grin), c(1:3,10:24)]
n_grin_manual_pl$source.taxid <- n_grin_manual_pl$species_taxid
n_grin_manual_pl$plant.pathogen <- T
n_grin_manual_pl$plant.host <- T
n_grin_manual_pl$plant.pathogen.source <- puregrin_new.citation 
n_grin_manual_pl$plant.host.source <- puregrin_new.citation
n_grin_manual_pl[setdiff(names(all.dat), names(n_grin_manual_pl))] <- NA

# extract the appropriate columns for species with a specified disease and an animal host
n_grin_manual_hp <- n_grin_manual[!is.na(n_grin_manual$human.host.grin), c(1:3,10:24)]
n_grin_manual_hp$source.taxid <- n_grin_manual_hp$species_taxid
n_grin_manual_hp$human.pathogen <- T
n_grin_manual_hp$human.pathogen.source <- paste0(puregrin_new.citation,";", pureatlas.citation)
n_grin_manual_hp[setdiff(names(all.dat), names(n_grin_manual_hp))] <- NA

# extract the appropriate columns for species without a specified disease and a plant host
n_grin_manual_ple <- n_grin_manual[n_grin_manual$disease=="" & !is.na(n_grin_manual$plant.host.grin), c(1:3,10:24)]
n_grin_manual_ple$source.taxid <- n_grin_manual_ple$species_taxid
n_grin_manual_ple$plant.host <- T
n_grin_manual_ple$plant.host.source <- puregrin_new.citation
n_grin_manual_ple[setdiff(names(all.dat), names(n_grin_manual_ple))] <- NA

all.dat.new <- all.dat

all.dat.new$plant.pathogen[all.dat.new$species_taxid %in% n_grin_manual_pl$species_taxid] <- T
all.dat.new$plant.host[all.dat.new$species_taxid %in% n_grin_manual_pl$species_taxid] <- T 
all.dat.new$plant.host[all.dat.new$species_taxid %in% n_grin_manual_ple$species_taxid] <- T 
all.dat.new$human.pathogen[all.dat.new$species_taxid %in% n_grin_manual_hp$species_taxid] <- T 

all.dat.new$Species[is.na(all.dat.new$Species)] <- all.dat.new$organism_name[is.na(all.dat.new$Species)]

all.dat.new <- add_single_label_resource(all.dat.new, n_grin_manual_pl, puregrin_new.citation, "plant.pathogen")
all.dat.new <- add_single_label_resource(all.dat.new, n_grin_manual_pl, puregrin_new.citation, "plant.host")
all.dat.new <- add_single_label_resource(all.dat.new, n_grin_manual_hp, puregrin_new.citation, "human.pathogen")
all.dat.new <- add_single_label_resource(all.dat.new, n_grin_manual_hp, pureatlas.citation, "human.pathogen")
all.dat.new <- add_single_label_resource(all.dat.new, n_grin_manual_ple, puregrin_new.citation, "plant.host")

all.dat.new[c(2,3,5,6,7,9,10)] <- sapply(all.dat.new[c(2,3,5,6,7,9,10)],as.character)

updated_taxids <- new_sp$species_taxid

# add updated labels
updates <- genbank_ncl[genbank_ncl$species_taxid %in% updated_taxids,]
all.dat.new <- merge(all.dat.new, updates, by = "species_taxid", all = T, suffixes = c("",".NEW"))
all.dat.new[all.dat.new$species_taxid %in% updated_taxids,2:10] <- all.dat.new[all.dat.new$species_taxid %in% updated_taxids,30:38]
all.dat.new[,30:58] <- NULL
all.dat.new$label.date <- as.Date("2021-10-09")
all.dat.new$label.date[all.dat.new$species_taxid %in% updated_taxids] <- as.Date("2022-01-02")

# get temporal test set
temporal.test <- all.dat.new[all.dat.new$species_taxid %in% c(n_grin_manual_pl$species_taxid, n_grin_manual_hp$species_taxid),]
temporal.test$Ambiguous <- FALSE
temporal.test$fold1 <- "test"
temporal.test$subset <- "selected"
colnames(temporal.test)[14] <- "Pathogenic"
temporal.test$Pathogenic[is.na(temporal.test$Pathogenic)] <- F

saveRDS(all.dat.new, "FUNGI_DATA_CUR/TEMPORAL/all_data_2022-01-02.rds")
saveRDS(temporal.test, "FUNGI_DATA_CUR/TEMPORAL/temporal-test.rds")
write.csv(all.dat.new, "FUNGI_DATA_CUR/TEMPORAL/all_data_2022-01-02.csv")
write.csv(temporal.test, "FUNGI_DATA_CUR/TEMPORAL/temporal-test.csv")

selected <- temporal.test
download_format <- ".fna"
urls.test.HP <- sapply(as.character(selected$ftp_path[selected$fold1=="test" & selected$Pathogenic]), function(f){name <- unlist(strsplit(as.character(f), split = "/")); name <- name[length(name)]; return(paste0(f, "/", name, "_genomic", download_format, ".gz"))})
writeLines(urls.test.HP, con = paste0("FUNGI_DATA_CUR/TEMPORAL/urls.test.HP", download_format, ".txt"))

urls.test.NP <- sapply(as.character(selected$ftp_path[selected$fold1=="test" & !selected$Pathogenic]), function(f){name <- unlist(strsplit(as.character(f), split = "/")); name <- name[length(name)]; return(paste0(f, "/", name, "_genomic", download_format, ".gz"))})
writeLines(urls.test.NP, con = paste0("FUNGI_DATA_CUR/TEMPORAL/urls.test.NP", download_format, ".txt"))

tvt <- readRDS("FUNGI_DATA_CUR/TVT_manual/IMG_1_folds_fungi.rds")
tvt.new <- rbind(tvt, selected[,-30])
saveRDS(tvt.new, "FUNGI_DATA_CUR/TEMPORAL/IMG_1_folds_fungi_temporal.rds")
