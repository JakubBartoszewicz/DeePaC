## DFVF

dfvf_hp$human.pathogen <- TRUE
dfvf_an$animal.pathogen <- TRUE
dfvf_pl$plant.pathogen <- TRUE
dfvf_pl$plant.host<- TRUE
dfvf.citation <- 'Tao Lu, Bo Yao, Chi Zhang, DFVF: database of fungal virulence factors, Database, Volume 2012, 2012, bas032, https://doi.org/10.1093/database/bas032'
dfvf_all <- dfvf_hp
dfvf_all[dfvf_all$human.pathogen, "human.pathogen.source"] <- dfvf.citation 
dfvf_all <- add_single_label_resource(dfvf_all, dfvf_an, dfvf.citation , "animal.pathogen")
dfvf_all <- add_single_label_resource(dfvf_all, dfvf_pl, dfvf.citation , "plant.pathogen")
dfvf_all <- add_single_label_resource(dfvf_all, dfvf_pl, dfvf.citation , "plant.host")

write.csv(x=dfvf_hp, file = "FUNGI_DATA_CUR/labels/dfvf_hp.csv", row.names= FALSE)
write.csv(x=dfvf_an, file = "FUNGI_DATA_CUR/labels/dfvf_an.csv", row.names= FALSE)
write.csv(x=dfvf_pl, file = "FUNGI_DATA_CUR/labels/dfvf_pl.csv", row.names= FALSE)
write.csv(x=dfvf_all, file = "FUNGI_DATA_CUR/labels/dfvf_all.csv", row.names= FALSE)
all.w.tax <- dfvf_all

## TAXONOMY pathogen SEARCH

tax_hp <- tax_hp[tax_hp$source.name != "Pneumocystis carinii",] #after the nomenclature change, this is reserved for the rat-infecting Pneumocystis
tax_hp$putative.human.host <- TRUE
puretax.citation  <- 'Conrad L Schoch, Stacy Ciufo, Mikhail Domrachev, Carol L Hotton, Sivakumar Kannan, Rogneda Khovanskaya, Detlef Leipe, Richard Mcveigh, Kathleen O’Neill, Barbara Robbertse, Shobha Sharma, Vladimir Soussov, John P Sullivan, Lu Sun, Seán Turner, Ilene Karsch-Mizrachi, NCBI Taxonomy: a comprehensive update on curation, resources and tools, Database, Volume 2020, 2020, baaa062, https://doi.org/10.1093/database/baaa062'
write.csv(x=tax_hp, file = "FUNGI_DATA_CUR/labels/tax_search.csv", row.names = FALSE)
# add suspected human host
all.w.tax <- add_single_label_resource(all.w.tax, tax_hp, puretax.citation, "putative.human.host")

## TAYLOR ET AL

taylor.citation <- 'Taylor, L. H., Latham, S. M., & Woolhouse, M. E. (2001). Risk factors for human disease emergence. Philosophical transactions of the Royal Society of London. Series B, Biological sciences, 356(1411), 983–989. https://doi.org/10.1098/rstb.2001.0888'
taylor$human.pathogen <- TRUE
write.csv(x=taylor, file = "FUNGI_DATA_CUR/labels/taylor.csv", row.names = FALSE)
taylor.w.tax <- taylor[taylor$species_taxid!="",]
taylor.wo.tax <- taylor[taylor$species_taxid=="",]
all.w.tax <- add_single_label_resource(all.w.tax, taylor.w.tax, taylor.citation, "human.pathogen")
all.wo.tax <- taylor.wo.tax

## VACHER ET AL

vacher.citation <- 'Vacher C, Piou D, Desprez-Loustau ML (2008) Architecture of an Antagonistic Tree/Fungus Network: The Asymmetric Influence of Past Evolutionary History. PLOS ONE 3(3): e1740. https://doi.org/10.1371/journal.pone.0001740'
vacher$plant.pathogen <- TRUE
vacher$plant.host <- TRUE
write.csv(x=vacher, file = "FUNGI_DATA_CUR/labels/vacher.csv", row.names = FALSE)
vacher.w.tax <- vacher[vacher$species_taxid!="",]
vacher.wo.tax <- vacher[vacher$species_taxid=="",]
all.w.tax <- add_single_label_resource(all.w.tax, vacher.w.tax, vacher.citation, "plant.pathogen")
all.wo.tax <- add_single_label_resource(all.wo.tax, vacher.wo.tax, vacher.citation, "plant.pathogen")
all.w.tax <- add_single_label_resource(all.w.tax, vacher.w.tax, vacher.citation, "plant.host")
all.wo.tax <- add_single_label_resource(all.wo.tax, vacher.wo.tax, vacher.citation, "plant.host")

# Doehlemann

doehlemann$plant.pathogen <- TRUE
doehlemann$plant.host <- TRUE
write.csv(x=doehlemann, file = "FUNGI_DATA_CUR/labels/doehlemann.csv", row.names = FALSE)
doehlemann.w.tax <- doehlemann[doehlemann$species_taxid!="",]
doehlemann.wo.tax <- doehlemann[doehlemann$species_taxid=="",]
doehlemann.citation <- "Doehlemann, G., Ökmen, B., Zhu, W., & Sharon, A. (2017). Plant Pathogenic Fungi. Microbiology Spectrum, 5(1). doi: 10.1128/microbiolspec.funk-0023-2016"
all.w.tax <- add_single_label_resource(all.w.tax, doehlemann.w.tax, doehlemann.citation, "plant.pathogen")
all.wo.tax <- add_single_label_resource(all.wo.tax, doehlemann.wo.tax, doehlemann.citation, "plant.pathogen")
all.w.tax <- add_single_label_resource(all.w.tax, doehlemann.w.tax, doehlemann.citation, "plant.host")
all.wo.tax <- add_single_label_resource(all.wo.tax, doehlemann.wo.tax, doehlemann.citation, "plant.host")

# Smith

smith$animal.pathogen <- TRUE
write.csv(x=smith, file = "FUNGI_DATA_CUR/labels/smith.csv", row.names = FALSE)
smith.w.tax <- smith[smith$species_taxid!="",]
smith.wo.tax <- smith[smith$species_taxid=="",]
smith.citation <- "Smith, J.M. (2006). Fungal Pathogens of Nonhuman Animals. In eLS, (Ed.). https://doi.org/10.1038/npg.els.0004235"
all.w.tax <- add_single_label_resource(all.w.tax, smith.w.tax, smith.citation, "animal.pathogen")
all.wo.tax <- add_single_label_resource(all.wo.tax, smith.wo.tax, smith.citation, "animal.pathogen")

# Seyedmousavi

seyedmousavi_an$animal.pathogen <- TRUE
write.csv(x=seyedmousavi_an, file = "FUNGI_DATA_CUR/labels/seyedmousavi_animal.csv", row.names = FALSE)
seyedmousavi_an.w.tax <- seyedmousavi_an[seyedmousavi_an$species_taxid!="",]
seyedmousavi_an.wo.tax <- seyedmousavi_an[seyedmousavi_an$species_taxid=="",]
seyedmousavi.citation <- "Seyedmousavi, S., Bosco, S., de Hoog, S., Ebel, F., Elad, D., Gomes, R. R., Jacobsen, I. D., Jensen, H. E., Martel, A., Mignon, B., Pasmans, F., Piecková, E., Rodrigues, A. M., Singh, K., Vicente, V. A., Wibbelt, G., Wiederhold, N. P., & Guillot, J. (2018). Fungal infections in animals: a patchwork of different situations. Medical mycology, 56(suppl_1), 165–187. https://doi.org/10.1093/mmy/myx104"
all.w.tax <- add_single_label_resource(all.w.tax, seyedmousavi_an.w.tax, seyedmousavi.citation, "animal.pathogen")
all.wo.tax <- add_single_label_resource(all.wo.tax, seyedmousavi_an.wo.tax, seyedmousavi.citation, "animal.pathogen")

seyedmousavi_hp$human.pathogen <- TRUE
write.csv(x=seyedmousavi_hp, file = "FUNGI_DATA_CUR/labels/seyedmousavi_human.csv", row.names = FALSE)
seyedmousavi_hp.w.tax <- seyedmousavi_hp[seyedmousavi_hp$species_taxid!="",]
seyedmousavi_hp.wo.tax <- seyedmousavi_hp[seyedmousavi_hp$species_taxid=="",]
all.w.tax <- add_single_label_resource(all.w.tax, seyedmousavi_hp.w.tax, seyedmousavi.citation, "human.pathogen")
all.wo.tax <- add_single_label_resource(all.wo.tax, seyedmousavi_hp.wo.tax, seyedmousavi.citation, "human.pathogen")

# Reviews: HP

reviews_hp$human.pathogen <- TRUE
write.csv(x=reviews_hp, file = "FUNGI_DATA_CUR/labels/reviews_human.csv", row.names = FALSE)
reviews_hp.w.tax <- reviews_hp[reviews_hp$species_taxid!="",]
reviews_hp.wo.tax <- reviews_hp[reviews_hp$species_taxid=="",]
revhp.citations <- read.csv("FUNGI_DATA_CUR/src/manual/reviews_hp_list.csv", stringsAsFactors = F)
colnames(revhp.citations) <- c("source.name", "human.pathogen.review")
# remove redundant spaces
revhp.citations$source.name <- gsub("\\s+", " ", x = revhp.citations$source.name)
revhp.citations$source.name <- gsub("\\s+$", "", x = revhp.citations$source.name)
# match citations to names
revhp.citations <- merge(revhp.citations, reviews_hp, by="source.name")
revhp.citations.bup <- revhp.citations[, c("source.name", "Species", "human.pathogen.review")]
revhp.citations <- revhp.citations[, c("Species", "human.pathogen.review")]
all.w.tax <- add_single_label_resource(all.w.tax, reviews_hp.w.tax, "", "human.pathogen")
all.wo.tax <- add_single_label_resource(all.wo.tax, reviews_hp.wo.tax, "", "human.pathogen")
all.w.tax <- add_custom_citations(all.w.tax, revhp.citations, "human.pathogen.source", "human.pathogen.review")
all.wo.tax <- add_custom_citations(all.wo.tax, revhp.citations, "human.pathogen.source", "human.pathogen.review")

# Reviews: AN

reviews_an$animal.pathogen <- TRUE
write.csv(x=reviews_an, file = "FUNGI_DATA_CUR/labels/reviews_animal.csv", row.names = FALSE)
reviews_an.w.tax <- reviews_an[reviews_an$species_taxid!="",]
reviews_an.wo.tax <- reviews_an[reviews_an$species_taxid=="",]
revan.citations <- read.csv("FUNGI_DATA_CUR/src/manual/reviews_an_list.csv", stringsAsFactors = F)
colnames(revan.citations) <- c("source.name", "animal.pathogen.review")
# remove redundant spaces
revan.citations$source.name <- gsub("\\s+", " ", x = revan.citations$source.name)
revan.citations$source.name <- gsub("\\s+$", "", x = revan.citations$source.name)
# match citations to names
revan.citations <- merge(revan.citations, reviews_an, by="source.name")
revan.citations.bup <- revan.citations[, c("source.name", "Species", "animal.pathogen.review")]
revan.citations <- revan.citations[, c("Species", "animal.pathogen.review")]
all.w.tax <- add_single_label_resource(all.w.tax, reviews_an.w.tax, "", "animal.pathogen")
all.wo.tax <- add_single_label_resource(all.wo.tax, reviews_an.wo.tax, "", "animal.pathogen")
all.w.tax <- add_custom_citations(all.w.tax, revan.citations, "animal.pathogen.source", "animal.pathogen.review")
all.wo.tax <- add_custom_citations(all.wo.tax, revan.citations, "animal.pathogen.source", "animal.pathogen.review")

# Reviews: PL

reviews_pl$plant.pathogen <- TRUE
reviews_pl$plant.host<- TRUE
write.csv(x=reviews_pl, file = "FUNGI_DATA_CUR/labels/reviews_plant.csv", row.names = FALSE)
reviews_pl.w.tax <- reviews_pl[reviews_pl$species_taxid!="",]
reviews_pl.wo.tax <- reviews_pl[reviews_pl$species_taxid=="",]
revpl.citations <- read.csv("FUNGI_DATA_CUR/src/manual/reviews_pl_list.csv", stringsAsFactors = F)
colnames(revpl.citations) <- c("source.name", "plant.pathogen.review")
# remove redundant spaces
revpl.citations$source.name <- gsub("\\s+", " ", x = revpl.citations$source.name)
revpl.citations$source.name <- gsub("\\s+$", "", x = revpl.citations$source.name)
# match citations to names
revpl.citations <- merge(revpl.citations, reviews_pl, by="source.name")
revpl.citations.bup <- revpl.citations[, c("source.name", "Species", "plant.pathogen.review")]
revpl.citations <- revpl.citations[, c("Species", "plant.pathogen.review")]
all.w.tax <- add_single_label_resource(all.w.tax, reviews_pl.w.tax, "", "plant.pathogen")
all.wo.tax <- add_single_label_resource(all.wo.tax, reviews_pl.wo.tax, "", "plant.pathogen")
all.w.tax <- add_custom_citations(all.w.tax, revpl.citations, "plant.pathogen.source", "plant.pathogen.review")
all.wo.tax <- add_custom_citations(all.wo.tax, revpl.citations, "plant.pathogen.source", "plant.pathogen.review")
all.w.tax <- add_custom_citations(all.w.tax, revpl.citations, "plant.host.source", "plant.pathogen.review")
all.wo.tax <- add_custom_citations(all.wo.tax, revpl.citations, "plant.host.source", "plant.pathogen.review")

# GRIN human pathogen search

puregrin_hp.citation <- 'Farr, D.F., & Rossman, A.Y. Fungal Databases, U.S. National Fungus Collections, ARS, USDA. Retrieved October 9, 2021, from https://nt.ars-grin.gov/fungaldatabases/'
grin_hp$putative.human.host <- TRUE
write.csv(x=grin_hp, file = "FUNGI_DATA_CUR/labels/grin_human.csv", row.names = FALSE)
grin_hp.w.tax <- grin_hp[grin_hp$species_taxid!="",]
grin_hp.wo.tax <- grin_hp[grin_hp$species_taxid=="",]
# add suspected human host
all.w.tax <- add_single_label_resource(all.w.tax, grin_hp.w.tax, puregrin_hp.citation, "putative.human.host")
all.wo.tax <- add_single_label_resource(all.wo.tax, grin_hp.wo.tax, puregrin_hp.citation, "putative.human.host")

## GRIN diagnostic

diag.citation <- 'Systematic Mycology and Microbiology Laboratory, ARS, USDA. Invasive Fungi, 2010. Retrieved October 9, 2021, from https://nt.ars-grin.gov/sbmlweb/fungi/diagnosticfactsheets.cfm'
grin_pl1$plant.pathogen <- TRUE
grin_pl1$plant.host <- TRUE
write.csv(x=grin_pl1, file = "FUNGI_DATA_CUR/labels/grin_diag.csv", row.names = FALSE)
grin_pl1.w.tax <- grin_pl1[grin_pl1$species_taxid!="",]
grin_pl1.wo.tax <- grin_pl1[grin_pl1$species_taxid=="",]
all.w.tax <- add_single_label_resource(all.w.tax, grin_pl1.w.tax, diag.citation, "plant.pathogen")
all.wo.tax <- add_single_label_resource(all.wo.tax, grin_pl1.wo.tax, diag.citation, "plant.pathogen")
all.w.tax <- add_single_label_resource(all.w.tax, grin_pl1.w.tax, diag.citation, "plant.host")
all.wo.tax <- add_single_label_resource(all.wo.tax, grin_pl1.wo.tax, diag.citation, "plant.host")

# GRIN nomenclature

puregrin_pl2.citation <- 'Farr, D.F., & Rossman, A.Y. Fungal Databases, U.S. National Fungus Collections, ARS, USDA. Retrieved October 9, 2021, from https://nt.ars-grin.gov/fungaldatabases/'
grin_pl2$plant.pathogen <- TRUE
grin_pl2$plant.host <- TRUE
write.csv(x=grin_pl2, file = "FUNGI_DATA_CUR/labels/grin_plant2.csv", row.names = FALSE)
grin_pl2.w.tax <- grin_pl2[grin_pl2$species_taxid!="",]
grin_pl2.wo.tax <- grin_pl2[grin_pl2$species_taxid=="",]
all.w.tax <- add_single_label_resource(all.w.tax, grin_pl2.w.tax, puregrin_hp.citation, "plant.pathogen")
all.wo.tax <- add_single_label_resource(all.wo.tax, grin_pl2.wo.tax, puregrin_hp.citation, "plant.pathogen")
all.w.tax <- add_single_label_resource(all.w.tax, grin_pl2.w.tax, puregrin_hp.citation, "plant.host")
all.wo.tax <- add_single_label_resource(all.wo.tax, grin_pl2.wo.tax, puregrin_hp.citation, "plant.host")


## Matching

genbank.matched <- merge(genbank_cl, all.w.tax, by = "species_taxid")
genbank.not.matched <- genbank_cl[!(genbank_cl$species_taxid %in% genbank.matched$species_taxid),]
genbank.not.matched[,"Species"] <- n[match(genbank.not.matched$species_taxid, n$taxid), "name"]
# in case a new taxid is in genbank but not yet in the Taxonomy dump
genbank.not.matched[is.na(genbank.not.matched$Species),"Species"] <- as.character(genbank.not.matched[is.na(genbank.not.matched$Species),"organism_name"])
genbank.not.matched <- filter_species_names(genbank.not.matched, "Species")
# for easier search in grin, remove brackets and annotations regarding adherence to the specific Code of Nomenclature
genbank.not.matched$Species <- gsub("\\[|\\]|", "", x = genbank.not.matched$Species)
genbank.not.matched$Species <- gsub(" \\(nom\\. inval\\.\\)", "", x = genbank.not.matched$Species)
genbank.not.matched$Species <- gsub(" \\(nom\\. nud\\.\\)", "", x = genbank.not.matched$Species)
write.csv(x=genbank.not.matched[,c("organism_name", "species_taxid", "Species")], file = "FUNGI_DATA_CUR/src/genbank_not_matched.csv", row.names = FALSE)

# GRIN manual

grin_manual <- read.csv("FUNGI_DATA_CUR/src/grin_manual_211009/grin_matched_final.csv", stringsAsFactors = F)
colnames(grin_manual)[c(1,6,7)] <- c("source.name", "plant.host.grin", "animal.host.grin")
grin_manual$source.taxid <- ""
grin_manual$human.pathogen <- F
grin_manual$animal.pathogen <- F
grin_manual$plant.pathogen <- F
grin_manual$plant.host <- F
grin_manual$putative.human.host <- F
grin_manual$putative.animal.host <- F
grin_manual$putative.plant.host <- F
grin_manual$human.pathogen.source <- "" 
grin_manual$animal.pathogen.source <- ""
grin_manual$plant.pathogen.source <- ""
grin_manual$plant.host.source <- ""
grin_manual$putative.human.host.source <- ""
grin_manual$putative.animal.host.source <- ""
grin_manual$putative.plant.host.source <- ""

# extract the appropriate columns for species with a specified disease and a plant host
grin_manual_pl <- grin_manual[grin_manual$disease!="" & !is.na(grin_manual$plant.host.grin), c(1:3,9:23)]
grin_manual_pl.old  <- grin_manual_pl[grin_manual_pl$species_taxid %in% all.w.tax$species_taxid,]
grin_manual_pl <- grin_manual_pl[!(grin_manual_pl$species_taxid %in% all.w.tax$species_taxid),]
grin_manual_pl$source.taxid <- grin_manual_pl$species_taxid
grin_manual_pl$plant.pathogen <- T
grin_manual_pl$plant.host <- T
grin_manual_pl$plant.pathogen.source <- puregrin_pl2.citation 
grin_manual_pl$plant.host.source <- puregrin_pl2.citation

# extract the appropriate columns for species with a specified disease and an animal host
grin_manual_an <- grin_manual[grin_manual$disease!="" & grin_manual$animal.host.grin!="", c(1:3,9:23)]
grin_manual_an.old  <- grin_manual_an[grin_manual_an$species_taxid %in% all.w.tax$species_taxid,]
grin_manual_an <- grin_manual_an[!(grin_manual_an$species_taxid %in% all.w.tax$species_taxid),]
grin_manual_an$source.taxid <- grin_manual_an$species_taxid
grin_manual_an$animal.pathogen <- T
grin_manual_an$animal.pathogen.source <- puregrin_pl2.citation 

# extract the appropriate columns for species without a specified disease and a plant host
grin_manual_ple <- grin_manual[grin_manual$disease=="" & !is.na(grin_manual$plant.host.grin), c(1:3,9:23)]
grin_manual_ple.old <- grin_manual_ple[grin_manual_ple$species_taxid %in% all.w.tax$species_taxid,]
grin_manual_ple <- grin_manual_ple[!(grin_manual_ple$species_taxid %in% all.w.tax$species_taxid),]
grin_manual_ple$source.taxid <- grin_manual_ple$species_taxid
grin_manual_ple$plant.host <- T
grin_manual_ple$plant.host.source <- puregrin_pl2.citation 

# bind papers and grin together
all.w.tax.man <- rbind(all.w.tax, grin_manual_pl, grin_manual_an, grin_manual_ple)

# our matched genomes
genbank.matched.man <- merge(genbank_cl, all.w.tax.man, by = "species_taxid")


# WARDEH

wardeh.citation <- "Wardeh, M., Risley, C., McIntyre, M. et al. Database of pathogen-pathogen and related species interactions, and their global distribution. Sci Data 2, 150049 (2015). https://doi.org/10.1038/sdata.2015.49"
wardeh_hp <- wardeh_hp[wardeh_hp$source.name != "pneumocystis carinii",] #after the nomenclature change, this is reserved for the rat-infecting Pneumocystis
wardeh_hp$putative.human.host <- TRUE
wardeh_an$putative.animal.host <- TRUE
wardeh_pl$putative.plant.host <- TRUE
all.wardeh <- wardeh_hp
# add suspected, auto-extracted hosts
all.wardeh[all.wardeh$human.pathogen, "putative.human.host.source"] <- wardeh.citation 
all.wardeh <- add_single_label_resource(all.wardeh, wardeh_an, wardeh.citation, "putative.animal.host")
all.wardeh <- add_single_label_resource(all.wardeh, wardeh_pl, wardeh.citation, "putative.plant.host")

# genomes from filtered EID2
genbank.wardeh <- merge(genbank_cl, all.wardeh, by = "species_taxid")

# combine our DB with filtered EID2
all.w.tax.combined.unconfirmed <- all.w.tax.man
all.w.tax.combined.unconfirmed <- add_single_label_resource(all.w.tax.combined.unconfirmed, wardeh_hp, wardeh.citation, "putative.human.host")
all.w.tax.combined.unconfirmed <- add_single_label_resource(all.w.tax.combined.unconfirmed, wardeh_an, wardeh.citation, "putative.animal.host")
all.w.tax.combined.unconfirmed <- add_single_label_resource(all.w.tax.combined.unconfirmed, wardeh_pl, wardeh.citation, "putative.plant.host")

# combined genomes, but not fully curated yet
genbank.combined.unconfirmed <- merge(genbank_cl, all.w.tax.combined.unconfirmed, by = "species_taxid")

# "clear" human pathogens
ours.hp.species <- genbank.matched.man$species_taxid[genbank.matched.man$human.pathogen]
# only those negatives that are pathogens
ours.np.species <- genbank.matched.man$species_taxid[!genbank.matched.man$human.pathogen & (genbank.matched.man$animal.pathogen | genbank.matched.man$plant.pathogen)]
# all non-positives from papers and grin, including non-pathos with plant hosts
ours.npe.species <- genbank.matched.man$species_taxid[!genbank.matched.man$human.pathogen]
# suspected positives from EID2
wardeh.hp.species <- genbank.wardeh$species_taxid[genbank.wardeh$putative.human.host]
# suspected non-positives form EID2
wardeh.np.species <- genbank.wardeh$species_taxid[!genbank.wardeh$putative.human.host]

# Plot Venn diagrams
grid.newpage()
venn.plot <- venn.diagram(list(ours.hp=ours.hp.species, ours.np=ours.np.species,
                               wardeh.hp=wardeh.hp.species, wardeh.np=wardeh.np.species), filename = NULL)
grid.draw(venn.plot)
grid.newpage()
venn.plot <- venn.diagram(list(ours.hp=ours.hp.species, ours.npe=ours.npe.species,
                               wardeh.hp=wardeh.hp.species, wardeh.np=wardeh.np.species), filename = NULL)
grid.draw(venn.plot)


# get our annotations supported only by the Taxonomy search for human host
puretax <- all.w.tax.man[all.w.tax.man$putative.human.host.source == puretax.citation & all.w.tax.man$human.pathogen.source == "",]
# get our annotations supported only by the human host in GRIN
puregrin_hp <- all.w.tax.man[all.w.tax.man$putative.human.host.source == puregrin_hp.citation & all.w.tax.man$human.pathogen.source == "",]
# get our annotations supported only by those two
taxgrin.citation <- paste0(puretax.citation, "; ", puregrin_hp.citation)
taxgrin <- all.w.tax.man[all.w.tax.man$putative.human.host.source == taxgrin.citation & all.w.tax.man$human.pathogen.source == "",]
# get suspected positive annotations supported only by Wardeh et al
purewardeh <- genbank.combined.unconfirmed[!genbank.combined.unconfirmed$human.pathogen & genbank.combined.unconfirmed$putative.human.host,]
# save them for manual curation with the Atlas
write.csv(x=puretax[,c("source.name", "species_taxid", "Species")], file = "FUNGI_DATA_CUR/src/puretax.csv", row.names = FALSE)
write.csv(x=puregrin_hp[,c("source.name", "species_taxid", "Species")], file = "FUNGI_DATA_CUR/src/puregrin_hp.csv", row.names = FALSE)
write.csv(x=taxgrin[,c("source.name", "species_taxid", "Species")], file = "FUNGI_DATA_CUR/src/taxgrin.csv", row.names = FALSE)
write.csv(x=purewardeh[,c("source.name", "species_taxid", "Species")], file = "FUNGI_DATA_CUR/src/purewardeh.csv", row.names = FALSE)

## Atlas manual search 

all.w.tax.confirmed <- all.w.tax.combined.unconfirmed
pureatlas.citation <- 'de Hoog GS, Guarro J, Gené J, Ahmed S, Al-Hatmi AMS, Figueras MJ & Vitale RG (2020) Atlas of Clinical Fungi, 4th edition. Hilversum.'
# here, the results of the manual curation with the Atlas have already been loaded and resolved by fungi-resolve.R
atlas_ours_hp$human.pathogen <- TRUE
atlas_ours_an$animal.pathogen <- TRUE
atlas_ours_pl$plant.pathogen <- TRUE
atlas_ours_pl$plant.host <- TRUE
atlas_wardeh_hp$human.pathogen <- TRUE
atlas_wardeh_an$animal.pathogen <- TRUE
atlas_wardeh_pl$plant.pathogen <- TRUE
atlas_wardeh_pl$plant.host <- TRUE

# check if anything in the manually curated files is actually already annotated as a confirmed pathogen (if too many species got curated, ignore them)
# get lists of confirmed pathogens befor the curation with the Atlas
confirmed_hp.preatlas <- all.w.tax.combined.unconfirmed$species_taxid[all.w.tax.combined.unconfirmed$human.pathogen]
confirmed_an.preatlas <- all.w.tax.combined.unconfirmed$species_taxid[all.w.tax.combined.unconfirmed$animal.pathogen]
confirmed_pl.preatlas <- all.w.tax.combined.unconfirmed$species_taxid[all.w.tax.combined.unconfirmed$plant.pathogen]

# if redundant species got curated, save them for inspection and remove them
atlas_ours_hp.old  <- atlas_ours_hp[atlas_ours_hp$species_taxid %in% confirmed_hp.preatlas,]
atlas_ours_hp <- atlas_ours_hp[!(atlas_ours_hp$species_taxid %in% confirmed_hp.preatlas),]
atlas_ours_an.old  <- atlas_ours_an[atlas_ours_an$species_taxid %in% confirmed_an.preatlas,]
atlas_ours_an <- atlas_ours_an[!(atlas_ours_an$species_taxid %in% confirmed_an.preatlas),]
atlas_ours_pl.old  <- atlas_ours_pl[atlas_ours_pl$species_taxid %in% confirmed_pl.preatlas,]
atlas_ours_pl <- atlas_ours_pl[!(atlas_ours_pl$species_taxid %in% confirmed_pl.preatlas),]

atlas_wardeh_hp.old  <- atlas_wardeh_hp[atlas_wardeh_hp$species_taxid %in% confirmed_hp.preatlas,]
atlas_wardeh_hp <- atlas_wardeh_hp[!(atlas_wardeh_hp$species_taxid %in% confirmed_hp.preatlas),]
atlas_wardeh_an.old  <- atlas_wardeh_an[atlas_wardeh_an$species_taxid %in% confirmed_an.preatlas,]
atlas_wardeh_an <- atlas_wardeh_an[!(atlas_wardeh_an$species_taxid %in% confirmed_an.preatlas),]
atlas_wardeh_pl.old  <- atlas_wardeh_pl[atlas_wardeh_pl$species_taxid %in% confirmed_pl.preatlas,]
atlas_wardeh_pl <- atlas_wardeh_pl[!(atlas_wardeh_pl$species_taxid %in% confirmed_pl.preatlas),]

# add curated human pathogens
all.w.tax.confirmed <- add_single_label_resource(all.w.tax.confirmed, atlas_ours_hp, pureatlas.citation, "human.pathogen")
# update citations for records, for which a human host was suggested based on Taxonomy or GRIN
all.w.tax.confirmed <- add_single_label_resource(all.w.tax.confirmed, atlas_ours_hp[atlas_ours_hp$species_taxid %in% tax_hp$species_taxid,], 
                                                 puretax.citation, "human.pathogen")
all.w.tax.confirmed <- add_single_label_resource(all.w.tax.confirmed, atlas_ours_hp[atlas_ours_hp$species_taxid %in% grin_hp.w.tax$species_taxid,], 
                                                 puregrin_hp.citation, "human.pathogen")

# add found negatives
all.w.tax.confirmed <- add_single_label_resource(all.w.tax.confirmed, atlas_ours_an, pureatlas.citation, "animal.pathogen")
all.w.tax.confirmed <- add_single_label_resource(all.w.tax.confirmed, atlas_ours_pl, pureatlas.citation, "plant.pathogen")
all.w.tax.confirmed <- add_single_label_resource(all.w.tax.confirmed, atlas_ours_pl, pureatlas.citation, "plant.host")
# avoid duplicates
atlas_wardeh_hp <- atlas_wardeh_hp[!(atlas_wardeh_hp$species_taxid %in% atlas_ours_hp$species_taxid),]
atlas_wardeh_an <- atlas_wardeh_an[!(atlas_wardeh_an$species_taxid %in% atlas_ours_an$species_taxid),]
atlas_wardeh_pl <- atlas_wardeh_pl[!(atlas_wardeh_pl$species_taxid %in% atlas_ours_pl$species_taxid),]

# add the species from EID2 + Atlas
all.w.tax.confirmed <- add_single_label_resource(all.w.tax.confirmed, atlas_wardeh_hp, wardeh.citation, "human.pathogen")
all.w.tax.confirmed <- add_single_label_resource(all.w.tax.confirmed, atlas_wardeh_an, wardeh.citation, "animal.pathogen")
all.w.tax.confirmed <- add_single_label_resource(all.w.tax.confirmed, atlas_wardeh_pl, wardeh.citation, "plant.pathogen")
all.w.tax.confirmed <- add_single_label_resource(all.w.tax.confirmed, atlas_wardeh_pl, wardeh.citation, "plant.host")
all.w.tax.confirmed <- add_single_label_resource(all.w.tax.confirmed, atlas_wardeh_hp, pureatlas.citation, "human.pathogen")
all.w.tax.confirmed <- add_single_label_resource(all.w.tax.confirmed, atlas_wardeh_an, pureatlas.citation, "animal.pathogen")
all.w.tax.confirmed <- add_single_label_resource(all.w.tax.confirmed, atlas_wardeh_pl, pureatlas.citation, "plant.pathogen")
all.w.tax.confirmed <- add_single_label_resource(all.w.tax.confirmed, atlas_wardeh_pl, pureatlas.citation, "plant.host")

# ATLAS SYNONYMS

# load a table of synonyms according to the Atlas, which are not represented as synonyms in NCBI Taxonomy
atlas.synonyms <- read.csv("FUNGI_DATA_CUR/src/atlas/atlas-synonyms.csv", stringsAsFactors = F)
colnames(atlas.synonyms) <- c("Species", "species_taxid", "atlas.synonym.name", "atlas.synonym.taxid")
# match synonyms with species already in the DB, without annotating them so they don't leak into training/val/test sets. Just add synonym info
all.w.tax.confirmed.syn <- merge(all.w.tax.confirmed, atlas.synonyms[2:4], by = "species_taxid", all.x=T)

# bind all data together
all.dat <- rbind.fill(all.w.tax.confirmed.syn, all.wo.tax)

# all genomes with or without annotation
genbank.all.final <- merge(genbank_cl, all.w.tax.confirmed.syn, by = "species_taxid", all.x = T)
# exclude atlas synonyms
genbank.all.final.wo.syn <- genbank.all.final[is.na(genbank.all.final$atlas.synonym.taxid),]
# get matched without synonyms
genbank.matched.final <- merge(genbank_cl, all.w.tax.confirmed.syn[is.na(all.w.tax.confirmed.syn$atlas.synonym.taxid),], by = "species_taxid")
# get "clear" pathogens
genbank.pathogens <- genbank.matched.final[genbank.matched.final$human.pathogen | genbank.matched.final$animal.pathogen | genbank.matched.final$plant.pathogen,]
# get confirmed plant hosts
genbank.supplementary.strict <- genbank.matched.final[!(genbank.matched.final$species_taxid %in% genbank.pathogens$species_taxid) &
                                                        genbank.matched.final$plant.host,]
# get suspected negatives without any suggestions of having a human host
genbank.supplementary.wide <- genbank.matched.final[!(genbank.matched.final$species_taxid %in% genbank.pathogens$species_taxid) &
                                                      !(genbank.matched.final$species_taxid %in% genbank.supplementary.strict$species_taxid) &
                                                      !genbank.matched.final$putative.human.host,]
# get everything else that has a match
genbank.supplementary.rest <- genbank.matched.final[!(genbank.matched.final$species_taxid %in% genbank.pathogens$species_taxid) &
                                                     !(genbank.matched.final$species_taxid %in% genbank.supplementary.strict$species_taxid) &
                                                     !(genbank.matched.final$species_taxid %in% genbank.supplementary.wide$species_taxid),]
# get full supplementary dataset
genbank.supplementary.final <- rbind(genbank.supplementary.strict, genbank.supplementary.wide)
# get unmatched, including removed synonyms
genbank.not.matched.final <- genbank_cl[!(genbank_cl$species_taxid %in% genbank.matched.final$species_taxid),]
genbank.not.matched.final <- merge(genbank.not.matched.final, all.w.tax.confirmed.syn, by = "species_taxid", all.x = T)

# get final positive species with genomes
final.hp.species <- genbank.pathogens$species_taxid[genbank.pathogens$human.pathogen]
# get final negative species with genomes
final.np.species <- genbank.pathogens$species_taxid[genbank.pathogens$plant.pathogen | genbank.pathogens$animal.pathogen]
# get final negative species with genomes + species with plant hosts and genomes
final.npe.species  <- c(genbank.pathogens$species_taxid[genbank.pathogens$plant.pathogen | genbank.pathogens$animal.pathogen | genbank.pathogens$plant.host],
                        genbank.supplementary.strict$species_taxid)
# get final negative species with genomes + species with plant hosts and genomes + suspected negatives from EID2 with genomes
final.npe.wide.species <- c(final.npe.species, genbank.supplementary.wide$species_taxid)

# update Wardeh
# suspected positives from EID2
wardeh.hp.species <- genbank.wardeh$species_taxid[genbank.wardeh$putative.human.host]
# suspected non-positives form EID2
wardeh.np.species <- genbank.wardeh$species_taxid[genbank.wardeh$putative.plant.host | genbank.wardeh$putative.animal.host]

# Plot Venn diagrams
grid.newpage()
venn.plot <- venn.diagram(list("HP"=final.hp.species,
                               "NHP"=final.np.species,
                               "HH (Wardeh et al.)"=wardeh.hp.species,
                               "NHH (Wardeh et al.)"=wardeh.np.species),
                                fill = c("orange", "blue", "white", "white"),
                                filename = "FUNGI_DATA_CUR/venn_inclusive/core.png",
                                resolution=300,
                                imagetype="png",
                                width=3130,
                                height=2060,
                                cex = 2,
                                cat.cex = 2,)
#grid.draw(venn.plot)
grid.newpage()
venn.plot <- venn.diagram(list("HP"=final.hp.species,
                               "NHH"=final.npe.species,
                               "HH (Wardeh et al.)"=wardeh.hp.species,
                               "NHH (Wardeh et al.)"=wardeh.np.species),
                               fill = c("orange", "darkgreen", "white", "white"),
                               filename = "FUNGI_DATA_CUR/venn_inclusive/assoc.png",
                               resolution=300,
                               imagetype="png",
                               width=3130,
                               height=2060,
                               cex = 2,
                               cat.cex = 2,)
#grid.draw(venn.plot)
grid.newpage()
venn.plot <- venn.diagram(list("HP"=final.hp.species,
                               "pNHH"=final.npe.wide.species,
                               "HH (Wardeh et al.)"=wardeh.hp.species,
                               "NHH (Wardeh et al.)"=wardeh.np.species), filename = NULL)
grid.draw(venn.plot)

# bring them all, and in the darkness bind them
everything <- rbind.fill(genbank.all.final, all.dat[!(all.dat$Species %in% genbank.all.final$Species),])
everything.nas <- everything
everything.nas$human.pathogen[!everything.nas$human.pathogen] <- NA
everything.nas$animal.pathogen[!everything.nas$animal.pathogen] <- NA
everything.nas$plant.pathogen[!everything.nas$plant.pathogen] <- NA
everything.nas$plant.host[!everything.nas$plant.host] <- NA
everything.nas$putative.human.host[!everything.nas$putative.human.host] <- NA
everything.nas$putative.animal.host[!everything.nas$putative.animal.host] <- NA
everything.nas$putative.plant.host[!everything.nas$putative.plant.host] <- NA

genbank.pathogens.nas <- genbank.pathogens
genbank.pathogens.nas$human.pathogen[!genbank.pathogens.nas$human.pathogen] <- NA
genbank.pathogens.nas$animal.pathogen[!genbank.pathogens.nas$animal.pathogen] <- NA
genbank.pathogens.nas$plant.pathogen[!genbank.pathogens.nas$plant.pathogen] <- NA
genbank.pathogens.nas$plant.host[!genbank.pathogens.nas$plant.host] <- NA
genbank.pathogens.nas$putative.human.host[!genbank.pathogens.nas$putative.human.host] <- NA
genbank.pathogens.nas$putative.animal.host[!genbank.pathogens.nas$putative.animal.host] <- NA
genbank.pathogens.nas$putative.plant.host[!genbank.pathogens.nas$putative.plant.host] <- NA

saveRDS(object = everything.nas, file = "FUNGI_DATA_CUR/release/all_data.rds")
write.csv(x = everything.nas, file = "FUNGI_DATA_CUR/release/all_data.csv")
saveRDS(object = genbank.pathogens.nas, file = "FUNGI_DATA_CUR/release/core_fungal_pathogens.rds")
write.csv(x = genbank.pathogens.nas, file = "FUNGI_DATA_CUR/release/core_fungal_pathogens.csv")
saveRDS(object = genbank.supplementary.final, file = "FUNGI_DATA_CUR/release/fungal_supplementary.rds")
write.csv(x = genbank.supplementary.final, file = "FUNGI_DATA_CUR/release/fungal_supplementary.csv")
colnames(genbank.pathogens)[14] <- "Pathogenic"
colnames(genbank.supplementary.final)[14] <- "Pathogenic"
saveRDS(object = genbank.pathogens, file = "FUNGI_DATA_CUR/release/IMG_assemblies_fungi.rds")
saveRDS(object = genbank.supplementary.final, file = "FUNGI_DATA_CUR/release/IMG_assemblies_fungi_supp.rds")
