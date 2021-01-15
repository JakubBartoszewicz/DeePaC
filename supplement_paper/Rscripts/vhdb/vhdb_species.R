library(plyr)
id.updates <- read.csv("taxid_updates.csv")
n <- read.table("new_taxdump/names.dmp", sep = "\t", fill = TRUE, quote = NULL, comment.char = "")[,c(1,3,7)]
colnames(n) <- c("taxid", "name", "name_type")
n$name <- as.character(n$name)
n$name_type <- relevel(n$name_type, "scientific name")
n <- n[order(n$taxid, n$name_type),]
n <- n[!duplicated(n$taxid),]
n$name_type <- as.character(n$name_type)
t <- read.table("new_taxdump/nodes.dmp", sep = "\t", fill = TRUE, stringsAsFactors=F, quote = NULL, comment.char = "")[,c(1,3,5)]
colnames(t) <- c("taxid", "parent_id", "rank")

get_ancestor_at_rank <- function(descendant_taxid, target_rank, graph){
  taxid <- descendant_taxid
  while (taxid!=0){
    if (taxid==1){
      return (NULL)
    } else {
      rank <- graph[graph$taxid==taxid,"rank"]
      if (length(rank) > 0) {
        if (graph[graph$taxid==taxid,"rank"]==target_rank){
          return (graph[graph$taxid==taxid,])
        } else {
          taxid <- graph[graph$taxid==taxid,"parent_id"]
        }
      } else {
          cat(paste0("NO MATCH FOR ", taxid, " (searched for: ", descendant_taxid, ")\n"))
          orig <- graph[graph$taxid==descendant_taxid,]
          if (nrow(orig) > 0) {
            return (orig)
          } else {
            return(cbind(taxid=descendant_taxid, parent_id=NA, rank=NA))
          }
      }
    }
  }
}


datfile <- "vhdb-rds/IMG_assemblies_human.rds"
vdat <- readRDS(datfile)
vdat <- join(vdat, id.updates, by="virus.tax.id")
vdat$virus.tax.id[!is.na(vdat$new_taxid)] <- vdat$new_taxid[!is.na(vdat$new_taxid)]
vdat$new_taxid <- NULL
vdat.spec <- do.call(rbind, lapply(vdat$virus.tax.id, get_ancestor_at_rank, target_rank="species", graph=t))
vdat.spec[,"Species"] <- n[match(vdat.spec$taxid, n$taxid), "name"]
vdat[,"Species"] <- vdat.spec$Species
vdat[,"Species.taxid"] <- vdat.spec$taxid
vdat[,"Ambiguous"] <- FALSE
done_species <- unique(vdat[,"Species"])
saveRDS(vdat, paste0(gsub(".rds$", "", datfile), "_species.rds"))


datfile <- "vhdb-rds/IMG_assemblies_chordata.rds"
vdat <- readRDS(datfile)
vdat <- join(vdat, id.updates, by="virus.tax.id")
vdat$virus.tax.id[!is.na(vdat$new_taxid)] <- vdat$new_taxid[!is.na(vdat$new_taxid)]
vdat$new_taxid <- NULL
vdat.spec <- do.call(rbind, lapply(vdat$virus.tax.id, get_ancestor_at_rank, target_rank="species", graph=t))
vdat.spec[,"Species"] <- n[match(vdat.spec$taxid, n$taxid), "name"]
vdat[,"Species"] <- vdat.spec$Species
vdat[,"Species.taxid"] <- vdat.spec$taxid
vdat[,"Ambiguous"] <- FALSE
vdat[vdat$Species %in% done_species,"Ambiguous"] <- TRUE
done_species <- unique(c(done_species, vdat$Species))
saveRDS(vdat, paste0(gsub(".rds$", "", datfile), "_species.rds"))

datfile <- "vhdb-rds/IMG_assemblies_metazoa.rds"
vdat <- readRDS(datfile)
vdat <- join(vdat, id.updates, by="virus.tax.id")
vdat$virus.tax.id[!is.na(vdat$new_taxid)] <- vdat$new_taxid[!is.na(vdat$new_taxid)]
vdat$new_taxid <- NULL
vdat.spec <- do.call(rbind, lapply(vdat$virus.tax.id, get_ancestor_at_rank, target_rank="species", graph=t))
vdat.spec[,"Species"] <- n[match(vdat.spec$taxid, n$taxid), "name"]
vdat[,"Species"] <- vdat.spec$Species
vdat[,"Species.taxid"] <- vdat.spec$taxid
vdat[,"Ambiguous"] <- FALSE
vdat[vdat$Species %in% done_species,"Ambiguous"] <- TRUE
done_species <- unique(c(done_species, vdat$Species))
saveRDS(vdat, paste0(gsub(".rds$", "", datfile), "_species.rds"))

datfile <- "vhdb-rds/IMG_assemblies_eukarya.rds"
vdat <- readRDS(datfile)
vdat <- join(vdat, id.updates, by="virus.tax.id")
vdat$virus.tax.id[!is.na(vdat$new_taxid)] <- vdat$new_taxid[!is.na(vdat$new_taxid)]
vdat$new_taxid <- NULL
vdat.spec <- do.call(rbind, lapply(vdat$virus.tax.id, get_ancestor_at_rank, target_rank="species", graph=t))
vdat.spec[,"Species"] <- n[match(vdat.spec$taxid, n$taxid), "name"]
vdat[,"Species"] <- vdat.spec$Species
vdat[,"Species.taxid"] <- vdat.spec$taxid
vdat[,"Ambiguous"] <- FALSE
vdat[vdat$Species %in% done_species,"Ambiguous"] <- TRUE
done_species <- unique(c(done_species, vdat$Species))
saveRDS(vdat, paste0(gsub(".rds$", "", datfile), "_species.rds"))

datfile <- "vhdb-rds/IMG_assemblies_neukarya.rds"
vdat <- readRDS(datfile)
vdat <- join(vdat, id.updates, by="virus.tax.id")
vdat$virus.tax.id[!is.na(vdat$new_taxid)] <- vdat$new_taxid[!is.na(vdat$new_taxid)]
vdat$new_taxid <- NULL
vdat.spec <- do.call(rbind, lapply(vdat$virus.tax.id, get_ancestor_at_rank, target_rank="species", graph=t))
vdat.spec[,"Species"] <- n[match(vdat.spec$taxid, n$taxid), "name"]
vdat[,"Species"] <- vdat.spec$Species
vdat[,"Species.taxid"] <- vdat.spec$taxid
vdat[,"Ambiguous"] <- FALSE
vdat[vdat$Species %in% done_species,"Ambiguous"] <- TRUE
done_species <- unique(c(done_species, vdat$Species))
saveRDS(vdat, paste0(gsub(".rds$", "", datfile), "_species.rds"))