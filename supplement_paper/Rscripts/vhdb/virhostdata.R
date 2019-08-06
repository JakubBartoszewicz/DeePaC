# Read in data
all.vhdbd <- read.delim("virushostdb.daily.tsv")
# Delete record(s) with Variola as host name (Variola is both the virus and a fish), with "root" and empty host info
all.vhdbd <- all.vhdbd[!(all.vhdbd[,"host.name"] %in% c("Variola", "root", "", NA)), ]
# Delete viroids
all.vhdbd <- all.vhdbd[!grepl(pattern = "viroid", x = all.vhdbd[,"virus.name"], ignore.case = TRUE),]
# Delete satellites
all.vhdbd <- all.vhdbd[!grepl(pattern = "satellite", x = all.vhdbd$virus.name, ignore.case = TRUE) & !grepl(pattern = "satellite", x = all.vhdbd$virus.lineage, ignore.case = TRUE),]
# For compatibility with old IMG scripts
all.vhdbd$ftp_path <- ""
all.vhdbd$Pathogenic <- FALSE
all.vhdbd$assembly_level <- ""
all.vhdbd$assembly_accession <- all.vhdbd$virus.tax.id
all.vhdbd$Species <- all.vhdbd$virus.name

# Get human viruses
human <- all.vhdbd[all.vhdbd[,"host.name"] %in% c("Homo sapiens"), ]
# For compatibility with old IMG scripts
human$Pathogenic <- TRUE
# Get all non-human viruses
n.human.all <- all.vhdbd[!(all.vhdbd[,"virus.tax.id"] %in% human[,"virus.tax.id"]), ]

# Get all eukaryotic viruses
all.eukarya <- all.vhdbd[grepl("Eukaryota", all.vhdbd[,"host.lineage"]), ]
# Get all non-eukaryotic viruses (prokaryotes and occasionally other viruses) (so: no human viruses here)
n.eukarya.all <- all.vhdbd[!(all.vhdbd[,"virus.tax.id"] %in% all.eukarya[,"virus.tax.id"]), ]

# Get all metazoan viruses
all.metazoa <- all.eukarya[grepl("Metazoa", all.eukarya[,"host.lineage"]), ]
# Get all eukaryotic, but non-metazoan viruses (so: no human viruses here)
n.metazoa.euk <- all.eukarya[!(all.eukarya[,"virus.tax.id"] %in% all.metazoa[,"virus.tax.id"]), ]

# Get all Chordata viruses
all.chordata <- all.metazoa[grepl("Chordata", all.metazoa[,"host.lineage"]), ]
# Get all metazoan, but not Chordata viruses (so: no human viruses here)
n.chordata.met <- all.metazoa[!(all.metazoa[,"virus.tax.id"] %in% all.chordata[,"virus.tax.id"]), ]

# Get all Chordata, but non-human viruses
n.human.cho <- all.chordata[!(all.chordata[,"virus.tax.id"] %in% human[,"virus.tax.id"]), ]

# Remove duplicates
n.human.cho <- n.human.cho[!duplicated(n.human.cho[,"virus.tax.id"]),]
n.chordata.met <- n.chordata.met[!duplicated(n.chordata.met[,"virus.tax.id"]),]
n.metazoa.euk <- n.metazoa.euk[!duplicated(n.metazoa.euk[,"virus.tax.id"]),]
n.eukarya.all <- n.eukarya.all[!duplicated(n.eukarya.all[,"virus.tax.id"]),]

saveRDS(object = human, file = "IMG_assemblies_human.rds")
saveRDS(object = n.human.cho, file = "IMG_assemblies_chordata.rds")
saveRDS(object = n.chordata.met, file = "IMG_assemblies_metazoa.rds")
saveRDS(object = n.metazoa.euk, file = "IMG_assemblies_eukarya.rds")
saveRDS(object = n.eukarya.all, file = "IMG_assemblies_neukarya.rds")