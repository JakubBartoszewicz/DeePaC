library(plyr)
source("FUNGI_DATA_CUR/src/fungi-taxonomy.R")
source("FUNGI_DATA_CUR/src/fungi-fctions.R")
genbank.pathogens <- readRDS("FUNGI_DATA_CUR/release_selected/fungal_pathogens.rds")
labs <- read.csv2("fungi-multi-plots/fungi-all.csv", header = F, stringsAsFactors = F)
colnames(labs) <- c("orig_class", "fasta_file", "no_reads")
labs$assembly_accession <- gsub(pattern = "(GCA_[0-9]+\\.[0-9]).+", replacement = "\\1", x = labs$fasta_file)
labs.merged <- join(labs, genbank.pathogens[,c("assembly_accession", "species_taxid", "Species")])
tax.families <- do.call(rbind, lapply(labs.merged$species_taxid, get_ancestor_at_rank, target_rank="family", graph=t))
tax.orders <- do.call(rbind, lapply(labs.merged$species_taxid, get_ancestor_at_rank, target_rank="order", graph=t))
tax.classes <- do.call(rbind, lapply(labs.merged$species_taxid, get_ancestor_at_rank, target_rank="class", graph=t))
tax.phyla <- do.call(rbind, lapply(labs.merged$species_taxid, get_ancestor_at_rank, target_rank="phylum", graph=t))

tax.families.clean <- tax.families
tax.families.clean$taxid[is.na(tax.families.clean$rank)] <- NA
tax.orders.clean <- tax.orders
tax.orders.clean$taxid[is.na(tax.orders.clean$rank)] <- NA
tax.classes.clean <- tax.classes
tax.classes.clean$taxid[is.na(tax.classes.clean$rank)] <- NA
tax.phyla.clean <- tax.phyla
tax.phyla.clean$taxid[is.na(tax.phyla.clean$rank)] <- NA

tax.families.names <- join(tax.families.clean, n[,c("taxid", "name")])
tax.orders.names <- join(tax.orders.clean, n[,c("taxid", "name")])
tax.classes.names <- join(tax.classes.clean, n[,c("taxid", "name")])
tax.phyla.names <- join(tax.phyla.clean, n[,c("taxid", "name")])

labs.tax <- labs.merged
labs.tax$tax.family_taxid <- tax.families.names$taxid
labs.tax$tax.family_name <-  tax.families.names$name
labs.tax$tax.order_taxid <- tax.orders.names$taxid
labs.tax$tax.order_name <- tax.orders.names$name
labs.tax$tax.class_taxid <- tax.classes.names$taxid
labs.tax$tax.class_name <- tax.classes.names$name
labs.tax$tax.phylum_taxid <- tax.phyla.names$taxid
labs.tax$tax.phylum_name <- tax.phyla.names$name

labs.phyl <- labs.tax
labs.phyl$label <- as.numeric(as.factor(labs.phyl$tax.phylum_name))-1

labs.out <- labs
labs.out$orig_class <- labs.phyl$label

write.csv2(labs.out[,1:3], "fungi-multi-plots/fungi-phyla.csv", row.names = F, quote = F)

labs.class <- labs.tax
labs.class$label <- as.numeric(as.factor(labs.class$tax.class_name))
labs.class$label[is.na(labs.class$label)] <- 0

labs.out <- labs
labs.out$orig_class <- labs.class$label

write.csv2(labs.out[,1:3], "fungi-multi-plots/fungi-class.csv", row.names = F, quote = F)

labs.order <- labs.tax
labs.order$label <- as.numeric(as.factor(labs.order$tax.order_name))
labs.order$label[is.na(labs.order$label)] <- 0

labs.out <- labs
labs.out$orig_class <- labs.order$label

write.csv2(labs.out[,1:3], "fungi-multi-plots/fungi-order.csv", row.names = F, quote = F)

labs.family <- labs.tax
labs.family$label <- as.numeric(as.factor(labs.family$tax.family_name))
labs.family$label[is.na(labs.family$label)] <- 0

labs.out <- labs
labs.out$orig_class <- labs.family$label

write.csv2(labs.out[,1:3], "fungi-multi-plots/fungi-family.csv", row.names = F, quote = F)

speciesP <- readRDS("fungi-multi-plots/speciesPreds.rds")
speciesP$Species <- gsub(x = speciesP$QuerySpecies, pattern = "^([a-zA-Z]+)\\.(.*)", replacement = "\\1 \\2")
speciesP$Species <- gsub(x = speciesP$Species, pattern = "^(.+)\\.(.*)", replacement = "\\1-\\2")
speciesV <- readRDS("fungi-multi-plots/speciesVal.rds")
speciesV$Species <- gsub(x = rownames(speciesV), pattern = "^([a-zA-Z]+)\\.(.*)", replacement = "\\1 \\2")
speciesV$Species <- gsub(x = speciesV$Species, pattern = "^(.+)\\.(.*)", replacement = "\\1-\\2")
colnames(speciesV)[1] <- "Prediction"
speciesTrain<- readRDS("fungi-multi-plots/speciesTrain.rds")
speciesTrain$Species <- gsub(x = rownames(speciesTrain), pattern = "^([a-zA-Z]+)\\.(.*)", replacement = "\\1 \\2")
speciesTrain$Species <- gsub(x = speciesTrain$Species, pattern = "^(.+)\\.(.*)", replacement = "\\1-\\2")
speciesTrain$Species <- gsub(x = speciesTrain$Species, pattern = "gattii-", replacement = "gattii ")
speciesTrain$Species <- gsub(x = speciesTrain$Species, pattern = "\\.\\.nom\\.\\.inval\\.-", replacement = " (nom. inval.)")
colnames(speciesTrain)[1] <- "Prediction"

labs.blast.db <- join(labs.merged, rbind(speciesV[,c("Species", "Prediction")], speciesP[,c("Species", "Prediction")]))
# copy labels of training genomes
labs.blast.db$Prediction[is.na(labs.blast.db$Prediction)] <- labs.blast.db$orig_class[is.na(labs.blast.db$Prediction)]

labs.blast <- join(labs.merged, rbind(speciesTrain[,c("Species", "Prediction")], speciesV[,c("Species", "Prediction")], speciesP[,c("Species", "Prediction")]))

labs.blast.order <- join(labs.blast, labs.order)
labs.blast.order.db <- join(labs.blast.db, labs.order)
labs.blast.errors <- labs.blast.order[labs.blast.order$orig_class!=labs.blast.order$Prediction,]
labs.blast.errors.db <- labs.blast.order.db[labs.blast.order.db$orig_class!=labs.blast.order.db$Prediction,]

labs.out <- labs
labs.out$orig_class <- as.numeric(labs.blast$Prediction)
write.csv2(labs.out[,1:3], "fungi-multi-plots/fungi-blast.csv", row.names = F, quote = F)


genbank.tempo <- readRDS("FUNGI_DATA_CUR/TEMPORAL/temporal-test.rds")
labs.temp <- read.csv2("fungi-multi-plots/fungi-tempo.csv", header = F, stringsAsFactors = F)
colnames(labs.temp) <- c("orig_class", "fasta_file", "no_reads")
labs.temp$assembly_accession <- gsub(pattern = "(GCA_[0-9]+\\.[0-9]).+", replacement = "\\1", x = labs.temp$fasta_file)
labs.temp.merged <- join(labs.temp, genbank.tempo[,c("assembly_accession", "species_taxid", "Species")])

speciesTempR <- readRDS("fungi-multi-plots/speciesTempReads.rds")
speciesTempR$Species <- gsub(x = speciesTempR$QuerySpecies, pattern = "^([a-zA-Z]+)\\.(.*)", replacement = "\\1 \\2")
speciesTempC <- readRDS("fungi-multi-plots/speciesTempConts.rds")
speciesTempC$Species <- gsub(x = speciesTempC$QuerySpecies, pattern = "^([a-zA-Z]+)\\.(.*)", replacement = "\\1 \\2")
colnames(speciesTempC)[1] <- "Prediction"

labs.tempR <- join(labs.temp.merged, speciesTempR[,c("Species", "Prediction")])
labs.tempC <- join(labs.temp.merged, speciesTempC[,c("Species", "Prediction")])

labs.temp.out <- labs.temp
labs.temp.out$orig_class <- as.numeric(labs.tempR$Prediction)
write.csv2(labs.temp.out[,1:3], "fungi-multi-plots/fungi-temp-reads.csv", row.names = F, quote = F)
labs.temp.out$orig_class <- labs.temp.out$orig_class + 2
labs.temp.out <- rbind(labs, labs.temp.out)
write.csv2(labs.temp.out[,1:3], "fungi-multi-plots/fungi-temp-reads-all.csv", row.names = F, quote = F)

labs.temp.out <- labs.temp
labs.temp.out$orig_class <- as.numeric(labs.tempC$Prediction)
write.csv2(labs.temp.out[,1:3], "fungi-multi-plots/fungi-temp-conts.csv", row.names = F, quote = F)
labs.temp.out$orig_class <- labs.temp.out$orig_class + 2
labs.temp.out <- rbind(labs, labs.temp.out)
write.csv2(labs.temp.out[,1:3], "fungi-multi-plots/fungi-temp-conts-all.csv", row.names = F, quote = F)
