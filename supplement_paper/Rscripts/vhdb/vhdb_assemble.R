human <- readRDS("VHDB_1_folds_human.rds")
human$refseq.id <- as.character(human$refseq.id)

nhuman <- readRDS("VHDB_1_folds_all_nhuman.rds")
nhuman$refseq.id <- as.character(nhuman$refseq.id)

lapply(1:nrow(human), function(i){system(paste0("cat ", gsub(",", ".fa", human[i,"refseq.id"]), ".fa >> vhdb_assembled/", human[i,"virus.tax.id"], ".fa"))})
lapply(1:nrow(nhuman), function(i){system(paste0("cat ", gsub(",", ".fa", nhuman[i,"refseq.id"]), ".fa >> vhdb_assembled/", nhuman[i,"virus.tax.id"], ".fa"))})
