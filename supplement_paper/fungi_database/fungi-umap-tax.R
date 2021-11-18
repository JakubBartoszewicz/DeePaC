library(plyr)
shuff <- read.csv2("fungi-multi-plots/fungi-all.csv", header = F, stringsAsFactors = F)
colnames(shuff) <- c("orig_class", "fasta_file", "no_reads")
shuff$assembly_accession <- gsub(pattern = "(GCA_[0-9]+\\.[0-9]).+", replacement = "\\1", x = shuff$fasta_file)
shuff.merged <- join(shuff, genbank.pathogens[,c("assembly_accession", "species_taxid", "Species")])
tax.families <- do.call(rbind, lapply(shuff.merged$species_taxid, get_ancestor_at_rank, target_rank="family", graph=t))
tax.orders <- do.call(rbind, lapply(shuff.merged$species_taxid, get_ancestor_at_rank, target_rank="order", graph=t))
tax.classes <- do.call(rbind, lapply(shuff.merged$species_taxid, get_ancestor_at_rank, target_rank="class", graph=t))
tax.phyla <- do.call(rbind, lapply(shuff.merged$species_taxid, get_ancestor_at_rank, target_rank="phylum", graph=t))

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

shuff.tax <- shuff.merged
shuff.tax$tax.family_taxid <- tax.families.names$taxid
shuff.tax$tax.family_name <-  tax.families.names$name
shuff.tax$tax.order_taxid <- tax.orders.names$taxid
shuff.tax$tax.order_name <- tax.orders.names$name
shuff.tax$tax.class_taxid <- tax.classes.names$taxid
shuff.tax$tax.class_name <- tax.classes.names$name
shuff.tax$tax.phylum_taxid <- tax.phyla.names$taxid
shuff.tax$tax.phylum_name <- tax.phyla.names$name

shuff.phyl <- shuff.tax
shuff.phyl$label <- as.numeric(as.factor(shuff.phyl$tax.phylum_name))-1

shuff.out <- shuff
shuff.out$orig_class <- shuff.phyl$label

write.csv2(shuff.out[,1:3], "fungi-multi-plots/fungi-phyla.csv", row.names = F, quote = F)

shuff.class <- shuff.tax
shuff.class$label <- as.numeric(as.factor(shuff.class$tax.class_name))
shuff.class$label[is.na(shuff.class$label)] <- 0

shuff.out <- shuff
shuff.out$orig_class <- shuff.class$label

write.csv2(shuff.out[,1:3], "fungi-multi-plots/fungi-class.csv", row.names = F, quote = F)

shuff.order <- shuff.tax
shuff.order$label <- as.numeric(as.factor(shuff.order$tax.order_name))
shuff.order$label[is.na(shuff.order$label)] <- 0

shuff.out <- shuff
shuff.out$orig_class <- shuff.order$label

write.csv2(shuff.out[,1:3], "fungi-multi-plots/fungi-order.csv", row.names = F, quote = F)

shuff.family <- shuff.tax
shuff.family$label <- as.numeric(as.factor(shuff.family$tax.family_name))
shuff.family$label[is.na(shuff.family$label)] <- 0

shuff.out <- shuff
shuff.out$orig_class <- shuff.family$label

write.csv2(shuff.out[,1:3], "fungi-multi-plots/fungi-family.csv", row.names = F, quote = F)

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

shuff.blast <- join(shuff.merged, rbind(speciesTrain[,c("Species", "Prediction")], speciesV[,c("Species", "Prediction")], speciesP[,c("Species", "Prediction")]))

shuff.out <- shuff
shuff.out$orig_class <- as.numeric(shuff.blast$Prediction)
write.csv2(shuff.out[,1:3], "fungi-multi-plots/fungi-blast.csv", row.names = F, quote = F)
