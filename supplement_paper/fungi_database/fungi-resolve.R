## Resolve

dfvf_hp <- resolve_species_taxid_list("FUNGI_DATA_CUR/src/DFVF_211009/DFVF_HUMAN.txt")
dfvf_hp <- dfvf_hp[dfvf_hp$source.name != "Pneumocystis carinii",] #after the nomenclature change, this is reserved for the rat-infecting Pneumocystis
# Stringer, J. R., Beard, C. B., Miller, R. F., & Wakefield, A. E. (2002). A new name (Pneumocystis jiroveci) for Pneumocystis from humans. Emerging infectious diseases, 8(9), 891–896. https://doi.org/10.3201/eid0809.020096
dfvf_an <- resolve_species_taxid_list("FUNGI_DATA_CUR/src/DFVF_211009/DFVF_ANIMAL.txt")
dfvf_pl <- resolve_species_taxid_list("FUNGI_DATA_CUR/src/DFVF_211009/DFVF_PLANT.txt")
taylor <- resolve_species_list("FUNGI_DATA_CUR/src/taylor-et-al/Taylor_list.txt")
taylor <- taylor[taylor$source.name != "Pneumocystis carinii",] #after the nomenclature change, this is reserved for the rat-infecting Pneumocystis
vacher <- resolve_species_list("FUNGI_DATA_CUR/src/vacher-et-al/Vacher_list.txt")
doehlemann <- resolve_species_list("FUNGI_DATA_CUR/src/doehlemann-et-al/doehlemann_list.txt")
smith <- resolve_species_list("FUNGI_DATA_CUR/src/smith/smith_list.txt")
seyedmousavi_an <- resolve_species_list("FUNGI_DATA_CUR/src/seyedmousavi-et-al/seyedmousavi_list.txt")
seyedmousavi_hp <- resolve_species_list("FUNGI_DATA_CUR/src/seyedmousavi-et-al/seyedmousavi_human_list.txt")
reviews_hp <- resolve_species_list("FUNGI_DATA_CUR/src/manual/reviews_hp_list.txt")
reviews_an <- resolve_species_list("FUNGI_DATA_CUR/src/manual/reviews_an_list.txt")
reviews_pl <- resolve_species_list("FUNGI_DATA_CUR/src/manual/reviews_pl_list.txt")

grin_hp <- resolve_species_list("FUNGI_DATA_CUR/src/grin_human_211009/grin_human_list_synonym_edit.txt")
grin_pl1 <- resolve_species_list("FUNGI_DATA_CUR/src/diagnosticfactsheets_211009/diagnosticfactsheets_211009_list.txt")
grin_pl2 <- resolve_species_list("FUNGI_DATA_CUR/src/nomenclaturefactsheets_211009/nomenclaturefactsheets-list.txt")

tax_hp <- resolve_species_taxid_list("FUNGI_DATA_CUR/src/taxonomy_search_211009/taxonomy_result_list.txt")

wardeh <- read.csv("FUNGI_DATA_CUR/src/WARDEH/fungi_data_full/fungi_data.txt", sep = "\t", header = T, stringsAsFactors = F)
# positives are those with human as a carrier
wardeh_hp.raw <- wardeh[wardeh$carrier == "homo sapiens",]
# plants mention plant or plants
wardeh_pl.raw <- wardeh[wardeh$carrierClass == "plant" | wardeh$carrierClass == "plants",]
# animal groups
wardeh_an.raw <- wardeh[wardeh$carrierClass == "arthropods" | wardeh$carrierClass == "worms" | wardeh$carrierClass == "sponges" | 
                          wardeh$carrierClass == "fish" | wardeh$carrierClass == "mammals" | 
                          wardeh$carrierClass == "amphibian" | wardeh$carrierClass == "reptilies" |
                          wardeh$carrierClass == "cnidarians" | wardeh$carrierClass == "aves" |
                          wardeh$carrierClass == "bryozoans" | wardeh$carrierClass == "mollusks" |
                          wardeh$carrierClass == "helminth" | wardeh$carrierClass == "animals" &
                          wardeh$carrier != "homo sapiens",]
# select cargo name and NCBI taxid
wardeh_hp.dat <- wardeh_hp.raw[,c(1,3)]
wardeh_pl.dat <- wardeh_pl.raw[,c(1,3)]
wardeh_an.dat <- wardeh_an.raw[,c(1,3)]
colnames(wardeh_hp.dat) <- c("source.name", "source.taxid")
colnames(wardeh_pl.dat) <- c("source.name", "source.taxid")
colnames(wardeh_an.dat) <- c("source.name", "source.taxid")
wardeh_hp.dat <- unique(wardeh_hp.dat)
wardeh_pl.dat <- unique(wardeh_pl.dat)
wardeh_an.dat <- unique(wardeh_an.dat)

wardeh_hp <- resolve_species_taxid_df(wardeh_hp.dat, filter_long=T)
wardeh_pl <- resolve_species_taxid_df(wardeh_pl.dat, filter_long=T)
wardeh_an <- resolve_species_taxid_df(wardeh_an.dat, filter_long=T)

write.csv(x=wardeh_hp, file = "FUNGI_DATA_CUR/labels/wardeh_hp.csv", row.names = FALSE)
write.csv(x=wardeh_pl, file = "FUNGI_DATA_CUR/labels/wardeh_pl.csv", row.names = FALSE)
write.csv(x=wardeh_an, file = "FUNGI_DATA_CUR/labels/wardeh_an.csv", row.names = FALSE)

atlas_ours_manual <- read.csv("FUNGI_DATA_CUR/src/atlas/ours-atlas-clean.csv")
atlas_ours_manual$source.taxid <- atlas_ours_manual$species_taxid
# select human, animal, and plant classes where the appropriate field is not NA
atlas_ours_hp <- resolve_species_taxid_df(atlas_ours_manual[!is.na(atlas_ours_manual$human.host), c("source.name", "source.taxid", "Species", "species_taxid")])
atlas_ours_an <- resolve_species_taxid_df(atlas_ours_manual[!is.na(atlas_ours_manual$animal.host), c("source.name", "source.taxid", "Species", "species_taxid")])
atlas_ours_pl <- resolve_species_taxid_df(atlas_ours_manual[!is.na(atlas_ours_manual$plant.host), c("source.name", "source.taxid", "Species", "species_taxid")])

write.csv(x=atlas_ours_hp, file = "FUNGI_DATA_CUR/labels/atlas_ours_hp.csv", row.names = FALSE)
write.csv(x=atlas_ours_an, file = "FUNGI_DATA_CUR/labels/atlas_ours_an.csv", row.names = FALSE)
write.csv(x=atlas_ours_pl, file = "FUNGI_DATA_CUR/labels/atlas_ours_pl.csv", row.names = FALSE)

atlas_wardeh_manual <- read.csv("FUNGI_DATA_CUR/src/atlas/wardeh-atlas-clean.csv")
atlas_wardeh_manual$source.taxid <- atlas_wardeh_manual$species_taxid
# select human, animal, and plant classes where the appropriate field is not NA
atlas_wardeh_hp <- resolve_species_taxid_df(atlas_wardeh_manual[!is.na(atlas_wardeh_manual$human.host), c("source.name", "source.taxid", "Species", "species_taxid")])
atlas_wardeh_hp <- atlas_wardeh_hp[atlas_wardeh_hp$source.name != "Pneumocystis carinii",] #after the nomenclature change, this is reserved for the rat-infecting Pneumocystis
atlas_wardeh_an <- resolve_species_taxid_df(atlas_wardeh_manual[!is.na(atlas_wardeh_manual$animal.host), c("source.name", "source.taxid", "Species", "species_taxid")])
atlas_wardeh_pl <- resolve_species_taxid_df(atlas_wardeh_manual[!is.na(atlas_wardeh_manual$plant.host), c("source.name", "source.taxid", "Species", "species_taxid")])

write.csv(x=atlas_wardeh_hp, file = "FUNGI_DATA_CUR/labels/atlas_wardeh_hp.csv", row.names = FALSE)
write.csv(x=atlas_wardeh_an, file = "FUNGI_DATA_CUR/labels/atlas_wardeh_an.csv", row.names = FALSE)
write.csv(x=atlas_wardeh_pl, file = "FUNGI_DATA_CUR/labels/atlas_wardeh_pl.csv", row.names = FALSE)
