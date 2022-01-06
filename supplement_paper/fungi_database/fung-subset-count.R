library(VennDiagram)
all.dat.new <- readRDS("FUNGI_DATA_CUR/TEMPORAL/all_data_2022-01-02.rds")
all.dat.old <- readRDS("FUNGI_DATA_CUR/release_selected/all_data.rds")


print("Updated labels:")
print(sum(all.dat.new$label.date==as.Date("2022-01-02")))
print("Updated proper species:")
filtered <- filter_species_names(all.dat.new, species_column = "organism_name")
print(sum(
  !is.na(filtered$ftp_path)
  & filtered$label.date==as.Date("2022-01-02")))
print("New proper species:")
print(sum(
  !(filtered$species_taxid %in% all.dat.old$species_taxid)
  & !is.na(filtered$ftp_path)
  & filtered$label.date==as.Date("2022-01-02")))
print("Previously absent, now labelled species:")
print(sum(
  (!is.na(filtered$human.pathogen) | !is.na(filtered$animal.pathogen) | !is.na(filtered$plant.pathogen) | !is.na(filtered$plant.host)) 
  & !(filtered$species_taxid %in% all.dat.old$species_taxid)
  & !is.na(filtered$ftp_path)
  & filtered$label.date==as.Date("2022-01-02")))


print("Labelled (updated):")
print(sum(
  (!is.na(all.dat.new$human.pathogen) | !is.na(all.dat.new$animal.pathogen) | !is.na(all.dat.new$plant.pathogen) | !is.na(all.dat.new$plant.host)) 
  & !is.na(all.dat.new$ftp_path)
  & is.na(all.dat.new$atlas.synonym.name)))
print("Labelled (old):")
print(sum(
  (!is.na(all.dat.old$human.pathogen) | !is.na(all.dat.old$animal.pathogen) | !is.na(all.dat.old$plant.pathogen) | !is.na(all.dat.old$plant.host)) 
  & !is.na(all.dat.old$ftp_path) 
  & is.na(all.dat.old$atlas.synonym.name)))
print("Labelled (new):")
print(sum(
  (!is.na(all.dat.new$human.pathogen) | !is.na(all.dat.new$animal.pathogen) | !is.na(all.dat.new$plant.pathogen) | !is.na(all.dat.new$plant.host))
  & !is.na(all.dat.new$ftp_path)
  & is.na(all.dat.new$atlas.synonym.name)
  & all.dat.new$label.date==as.Date("2022-01-02")))

print("Core (updated):")
print(sum(
  (!is.na(all.dat.new$human.pathogen) | !is.na(all.dat.new$animal.pathogen) | !is.na(all.dat.new$plant.pathogen)) 
  & !is.na(all.dat.new$ftp_path)
  & is.na(all.dat.new$atlas.synonym.name)))
print("Core (old):")
print(sum(
  (!is.na(all.dat.old$human.pathogen) | !is.na(all.dat.old$animal.pathogen) | !is.na(all.dat.old$plant.pathogen)) 
  & !is.na(all.dat.old$ftp_path) 
  & is.na(all.dat.old$atlas.synonym.name)))
print("Core (new):")
print(sum(
  (!is.na(all.dat.new$human.pathogen) | !is.na(all.dat.new$animal.pathogen) | !is.na(all.dat.new$plant.pathogen))
  & !is.na(all.dat.new$ftp_path)
  & is.na(all.dat.new$atlas.synonym.name)
  & all.dat.new$label.date==as.Date("2022-01-02")))

print("HP (updated):")
print(sum(
  !is.na(all.dat.new$human.pathogen) 
  & !is.na(all.dat.new$ftp_path)
  & is.na(all.dat.new$atlas.synonym.name)))
print("HP (old):")
print(sum(
  !is.na(all.dat.old$human.pathogen) 
  & !is.na(all.dat.old$ftp_path) 
  & is.na(all.dat.old$atlas.synonym.name)))
print("HP (new):")
print(sum(
  !is.na(all.dat.new$human.pathogen) 
  & !is.na(all.dat.new$ftp_path)
  & is.na(all.dat.new$atlas.synonym.name)
  & all.dat.new$label.date==as.Date("2022-01-02")))

print("NHP (updated):")
print(sum(
  is.na(all.dat.new$human.pathogen) 
  & (!is.na(all.dat.new$animal.pathogen) | !is.na(all.dat.new$plant.pathogen)) 
  & !is.na(all.dat.new$ftp_path)
  & is.na(all.dat.new$atlas.synonym.name)))
print("NHP (old):")
print(sum(
  is.na(all.dat.old$human.pathogen) 
  & (!is.na(all.dat.old$animal.pathogen) | !is.na(all.dat.old$plant.pathogen)) 
  & !is.na(all.dat.old$ftp_path) 
  & is.na(all.dat.old$atlas.synonym.name)))
print("NHP (new):")
print(sum(
  is.na(all.dat.new$human.pathogen) 
  & (!is.na(all.dat.new$animal.pathogen) | !is.na(all.dat.new$plant.pathogen)) 
  & !is.na(all.dat.new$ftp_path)
  & is.na(all.dat.new$atlas.synonym.name)
  & all.dat.new$label.date==as.Date("2022-01-02")))

print("Plant assoc (updated):")
print(sum(
  !is.na(all.dat.new$plant.host) & is.na(all.dat.new$human.pathogen) & is.na(all.dat.new$animal.pathogen) & is.na(all.dat.new$plant.pathogen)
  & !is.na(all.dat.new$ftp_path)
  & is.na(all.dat.new$atlas.synonym.name)))
print("Plant assoc (old):")
print(sum(
  !is.na(all.dat.old$plant.host) & is.na(all.dat.old$human.pathogen) & is.na(all.dat.old$animal.pathogen) & is.na(all.dat.old$plant.pathogen)
  & !is.na(all.dat.old$ftp_path)
  & is.na(all.dat.old$atlas.synonym.name)))
print("Plant assoc (new):")
print(sum(
  !is.na(all.dat.new$plant.host) & is.na(all.dat.new$human.pathogen) & is.na(all.dat.new$animal.pathogen) & is.na(all.dat.new$plant.pathogen)
  & !is.na(all.dat.new$ftp_path)
  & is.na(all.dat.new$atlas.synonym.name)
  & all.dat.new$label.date==as.Date("2022-01-02")))

print("Putatively labelled genomes (updated):")
print(sum(
  (is.na(all.dat.new$plant.host) & is.na(all.dat.new$human.pathogen) & is.na(all.dat.new$animal.pathogen) & is.na(all.dat.new$plant.pathogen))
  & (!is.na(all.dat.new$putative.human.host) | !is.na(all.dat.new$putative.animal.host) | !is.na(all.dat.new$putative.plant.host))
  & !is.na(all.dat.new$ftp_path)
  & is.na(all.dat.new$atlas.synonym.name)))
print("Putatively labelled genomes (old):")
print(sum(
  (is.na(all.dat.old$plant.host) & is.na(all.dat.old$human.pathogen) & is.na(all.dat.old$animal.pathogen) & is.na(all.dat.old$plant.pathogen))
  & (!is.na(all.dat.old$putative.human.host) | !is.na(all.dat.old$putative.animal.host) | !is.na(all.dat.old$putative.plant.host))
  & !is.na(all.dat.old$ftp_path)
  & is.na(all.dat.old$atlas.synonym.name)))

print("Unlabelled genomes (updated):")
print(sum(
  (is.na(all.dat.new$plant.host) & is.na(all.dat.new$human.pathogen) & is.na(all.dat.new$animal.pathogen) & is.na(all.dat.new$plant.pathogen))
  & (is.na(all.dat.new$putative.human.host) & is.na(all.dat.new$putative.animal.host) & is.na(all.dat.new$putative.plant.host))
  & !is.na(all.dat.new$ftp_path)
  & is.na(all.dat.new$atlas.synonym.name)))
print("Unlabelled genomes (old):")
print(sum(
  (is.na(all.dat.old$plant.host) & is.na(all.dat.old$human.pathogen) & is.na(all.dat.old$animal.pathogen) & is.na(all.dat.old$plant.pathogen))
  & (is.na(all.dat.old$putative.human.host) & is.na(all.dat.old$putative.animal.host) & is.na(all.dat.old$putative.plant.host))
  & !is.na(all.dat.old$ftp_path)
  & is.na(all.dat.old$atlas.synonym.name)))

print("Atlas synonyms (updated):")
print(sum(
  !is.na(all.dat.new$atlas.synonym.name)))
print("Atlas synonyms (old):")
print(sum(
  !is.na(all.dat.old$atlas.synonym.name)))

print("Atlas synonym genomes (updated):")
print(sum(
  !is.na(all.dat.new$ftp_path)
  & !is.na(all.dat.new$atlas.synonym.name)))
print("Atlas synonym genomes (old):")
print(sum(
  !is.na(all.dat.old$ftp_path)
  & !is.na(all.dat.old$atlas.synonym.name)))

print("Labelled without genomes (updated):")
print(sum(
  (!is.na(all.dat.new$human.pathogen) | !is.na(all.dat.new$animal.pathogen) | !is.na(all.dat.new$plant.pathogen)) 
  & is.na(all.dat.new$ftp_path)
  & is.na(all.dat.new$atlas.synonym.name)))
print("Labelled without genomes (old):")
print(sum(
  (!is.na(all.dat.old$human.pathogen) | !is.na(all.dat.old$animal.pathogen) | !is.na(all.dat.old$plant.pathogen))
  & is.na(all.dat.old$ftp_path)
  & is.na(all.dat.old$atlas.synonym.name)))

print("Labelled without taxids (updated):")
print(sum(
  (!is.na(all.dat.new$human.pathogen) | !is.na(all.dat.new$animal.pathogen) | !is.na(all.dat.new$plant.pathogen)) 
  & all.dat.new$species_taxid==""
  & is.na(all.dat.new$atlas.synonym.name)))
print("Labelled without taxids (old):")
print(sum(
  (!is.na(all.dat.old$human.pathogen) | !is.na(all.dat.old$animal.pathogen) | !is.na(all.dat.old$plant.pathogen))
  & all.dat.old$species_taxid==""
  & is.na(all.dat.old$atlas.synonym.name)))

print("Put. labelled without genomes (updated):")
print(sum(
  (is.na(all.dat.new$plant.host) & is.na(all.dat.new$human.pathogen) & is.na(all.dat.new$animal.pathogen) & is.na(all.dat.new$plant.pathogen))
  & (!is.na(all.dat.new$putative.human.host) | !is.na(all.dat.new$putative.animal.host) | !is.na(all.dat.new$putative.plant.host))
  & is.na(all.dat.new$ftp_path)
  & is.na(all.dat.new$atlas.synonym.name)))
print("Put. labelled without genomes (old):")
print(sum(
  (is.na(all.dat.old$plant.host) & is.na(all.dat.old$human.pathogen) & is.na(all.dat.old$animal.pathogen) & is.na(all.dat.old$plant.pathogen))
  & (!is.na(all.dat.old$putative.human.host) | !is.na(all.dat.old$putative.animal.host) | !is.na(all.dat.old$putative.plant.host))
  & is.na(all.dat.old$ftp_path)
  & is.na(all.dat.old$atlas.synonym.name)))

print("Put. labelled without taxids (updated):")
print(sum(
  (is.na(all.dat.new$plant.host) & is.na(all.dat.new$human.pathogen) & is.na(all.dat.new$animal.pathogen) & is.na(all.dat.new$plant.pathogen))
  & (!is.na(all.dat.new$putative.human.host) | !is.na(all.dat.new$putative.animal.host) | !is.na(all.dat.new$putative.plant.host))
  & all.dat.new$species_taxid==""
  & is.na(all.dat.new$atlas.synonym.name)))
print("Put. labelled without taxids (old):")
print(sum(
  (is.na(all.dat.old$plant.host) & is.na(all.dat.old$human.pathogen) & is.na(all.dat.old$animal.pathogen) & is.na(all.dat.old$plant.pathogen))
  & (!is.na(all.dat.old$putative.human.host) | !is.na(all.dat.old$putative.animal.host) | !is.na(all.dat.old$putative.plant.host))
  & all.dat.old$species_taxid==""
  & is.na(all.dat.old$atlas.synonym.name)))



# Remove synonyms
genomes.new <- all.dat.new[!is.na(all.dat.new$ftp_path) & is.na(all.dat.new$atlas.synonym.name),]
genomes.old <- all.dat.old[!is.na(all.dat.old$ftp_path) & is.na(all.dat.old$atlas.synonym.name),]
wardeh.search <- "Wardeh"


# get final positive species with genomes
final.hp.species.new <- genomes.new$species_taxid[!is.na(genomes.new$human.pathogen)]
# get final negative species with genomes
final.np.species.new <- genomes.new$species_taxid[!is.na(genomes.new$plant.pathogen) | !is.na(genomes.new$animal.pathogen)]
# get final negative species with genomes + species with plant hosts and genomes
final.npe.species.new  <- genomes.new$species_taxid[!is.na(genomes.new$plant.pathogen) | !is.na(genomes.new$animal.pathogen) | !is.na(genomes.new$plant.host)]

# update Wardeh
# suspected positives from EID2
wardeh.hp.species.new <- genomes.new$species_taxid[!is.na(genomes.new$putative.human.host) & grepl(pattern = wardeh.search, x = genomes.new$putative.human.host.source)]
# suspected non-positives form EID2
wardeh.np.species.new <- genomes.new$species_taxid[(!is.na(genomes.new$putative.plant.host) & grepl(pattern = wardeh.search, x = genomes.new$putative.plant.host.source))
                                               | !is.na(genomes.new$putative.animal.host) & grepl(pattern = wardeh.search, x = genomes.new$putative.animal.host.source)]

# Plot Venn diagrams (NOT COUNTING SYNONYMS AS SEPARATE RECORDS)
grid.newpage()
venn.plot <- venn.diagram(list("HP"=final.hp.species.new, 
                               "NHP"=final.np.species.new, 
                               "HH (Wardeh et al.)"=wardeh.hp.species.new, 
                               "NHH (Wardeh et al.)"=wardeh.np.species.new),
                          fill = c("orange", "blue", "white", "white"),
                          filename = "FUNGI_DATA_CUR/TEMPORAL/venn_inclusive/core_new.png",
                          resolution=300,
                          imagetype="png",
                          width=3130,
                          height=2060,
                          cex = 2,
                          cat.cex = 2,)
#grid.draw(venn.plot)
grid.newpage()
venn.plot <- venn.diagram(list("HP"=final.hp.species.new, 
                               "NHH"=final.npe.species.new, 
                               "HH (Wardeh et al.)"=wardeh.hp.species.new, 
                               "NHH (Wardeh et al.)"=wardeh.np.species.new),
                          fill = c("orange", "darkgreen", "white", "white"),
                          filename = "FUNGI_DATA_CUR/TEMPORAL/venn_inclusive/assoc_new.png",
                          resolution=300,
                          imagetype="png",
                          width=3130,
                          height=2060,
                          cex = 2,
                          cat.cex = 2,)

# get final positive species with genomes
final.hp.species.old <- genomes.old$species_taxid[!is.na(genomes.old$human.pathogen)]
# get final negative species with genomes
final.np.species.old <- genomes.old$species_taxid[!is.na(genomes.old$plant.pathogen) | !is.na(genomes.old$animal.pathogen)]
# get final negative species with genomes + species with plant hosts and genomes
final.npe.species.old  <- genomes.old$species_taxid[!is.na(genomes.old$plant.pathogen) | !is.na(genomes.old$animal.pathogen) | !is.na(genomes.old$plant.host)]

# update Wardeh
# suspected positives from EID2
wardeh.hp.species.old <- genomes.old$species_taxid[!is.na(genomes.old$putative.human.host) & grepl(pattern = wardeh.search, x = genomes.old$putative.human.host.source)]
# suspected non-positives form EID2
wardeh.np.species.old <- genomes.old$species_taxid[(!is.na(genomes.old$putative.plant.host) & grepl(pattern = wardeh.search, x = genomes.old$putative.plant.host.source))
                                               | !is.na(genomes.old$putative.animal.host) & grepl(pattern = wardeh.search, x = genomes.old$putative.animal.host.source)]

# Plot Venn diagrams
grid.newpage()
venn.plot <- venn.diagram(list("HP"=final.hp.species.old, 
                               "NHP"=final.np.species.old, 
                               "HH (Wardeh et al.)"=wardeh.hp.species.old, 
                               "NHH (Wardeh et al.)"=wardeh.np.species.old),
                          fill = c("orange", "blue", "white", "white"),
                          filename = "FUNGI_DATA_CUR/TEMPORAL/venn_inclusive/core_old.png",
                          resolution=300,
                          imagetype="png",
                          width=3130,
                          height=2060,
                          cex = 2,
                          cat.cex = 2,)
#grid.draw(venn.plot)
grid.newpage()
venn.plot <- venn.diagram(list("HP"=final.hp.species.old, 
                               "NHH"=final.npe.species.old, 
                               "HH (Wardeh et al.)"=wardeh.hp.species.old, 
                               "NHH (Wardeh et al.)"=wardeh.np.species.old),
                          fill = c("orange", "darkgreen", "white", "white"),
                          filename = "FUNGI_DATA_CUR/TEMPORAL/venn_inclusive/assoc_old.png",
                          resolution=300,
                          imagetype="png",
                          width=3130,
                          height=2060,
                          cex = 2,
                          cat.cex = 2,)

