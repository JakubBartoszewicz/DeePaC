#Searching functions
get_ancestor_at_rank <- function(descendant_taxid, target_rank, graph){
  taxid <- descendant_taxid
  while (taxid!=0){
    if (taxid==1){
      cat(paste0("REACHED THE TOP OF THE TREE. (searched for: ", descendant_taxid, ")\n"))
      return(cbind(taxid=descendant_taxid, parent_id=NA, rank=NA))
    } else {
      rank <- graph[graph$taxid==taxid,"rank"]
      if (length(rank) > 0) {
        if (graph[graph$taxid==taxid,"rank"]==target_rank){
          return (graph[graph$taxid==taxid,])
        } else {
          taxid <- graph[graph$taxid==taxid,"parent_id"]
        }
      } else {
        # if no match found, check the merged.dmp data
        merged.id <- m[m$old_id==taxid,2]
        if (length(merged.id)>0) {
          cat(paste0(taxid, " merged with ", merged.id, " (searched for: ", descendant_taxid, ")\n"))
          taxid <- merged.id
        } else {
          # if no merged match found, there's no match
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
}

get_species_matches <- function(source.name, graph){
  # grep the source.name in the set of names
  matched <- graph[grepl(x=graph$name, pattern=source.name, fixed=TRUE),]
  if (nrow(matched) > 0){
    # return all matches
    matched <- cbind.data.frame(source.name, matched, stringsAsFactors = F)
  } else{
    # if no match, return a row with NAs for this name
    matched <- cbind.data.frame(source.name, taxid=NA, name=source.name, name_type=NA, stringsAsFactors = F)
  }
  return(matched)
}

# Processing functions

filter_species_names <- function(vdat, species_column="Species"){
  # remove names suggesting it's not an established species name, or only a forma specialis (f. sp.) of a species
  vdat_clean <- vdat[!grepl(x=vdat[,species_column], pattern="uncultured", fixed = TRUE) &
                       !grepl(x=vdat[,species_column], pattern="vouchered ", fixed = TRUE) &
                       !grepl(x=vdat[,species_column], pattern=" f. sp. ", fixed = TRUE) &
                       !grepl(x=vdat[,species_column], pattern=" sp. ", fixed = TRUE) &
                       !grepl(x=vdat[,species_column], pattern=" sp.$",) &
                       !grepl(x=vdat[,species_column], pattern=" cf. ", fixed = TRUE) &
                       !grepl(x=vdat[,species_column], pattern=" aff. ", fixed = TRUE) &
                       !grepl(x=vdat[,species_column], pattern=" x ", fixed = TRUE),]
  vdat_clean <- filter_common_oomycota(vdat_clean, species_column)
  vdat_clean <- filter_nonspecies(vdat_clean, species_column)
  return(vdat_clean)
}

filter_common_oomycota <- function(vdat, species_column="Species"){
  # remove some genera of Oomycota, which are no longer recognized as fungi and will not be in the fungal part of genbank anyway
  vdat_clean <- vdat[!grepl(x=vdat[,species_column], pattern="Pythium", fixed = TRUE) &
                       !grepl(x=vdat[,species_column], pattern="Phytopythium", fixed = TRUE) &
                       !grepl(x=vdat[,species_column], pattern="Phytophthora", fixed = TRUE),]
  return(vdat_clean)
}

filter_nonspecies <- function(vdat, species_column="Species"){
  # remove names suggesting it's not actually a fungal species (but a virus, a species group/complex, a hybrid)
  vdat_clean <- vdat[!grepl(x=vdat[,species_column], pattern="virus", fixed = TRUE) &
                       !grepl(x=vdat[,species_column], pattern=" group", fixed = TRUE) &
                       !grepl(x=vdat[,species_column], pattern=" complex", fixed = TRUE) &
                       !grepl(x=vdat[,species_column], pattern=" sensu lato", fixed = TRUE) &
                       !grepl(x=vdat[,species_column], pattern=" hybrid", fixed = TRUE) &
                       !grepl(x=vdat[,species_column], pattern=" x ", fixed = TRUE),]
  return(vdat_clean)
}

filter_long_names <- function(vdat, species_column="Species"){
  # remove if number of words > 2 (many unnamed isolates), except for species in C. gatti complex and ignoring annotations regarding adherence to the specific Code of Nomenclature
  vdat_clean <- vdat[!(lengths(strsplit(vdat[,species_column], " "))>2) |
                       grepl(x=vdat[,species_column], pattern="Cryptococcus gatti", fixed = TRUE) |
                       grepl(x=vdat[,species_column], pattern="(nom.", fixed = TRUE),]
  
  return(vdat_clean)
}

resolve_multispecies <- function(vdat, multi.names, multi.column, ref.column){
  vdat_clean <- vdat
  # for each name in a list of names with ambiguous hits
  for (i in 1:length(multi.names)){
    # get that name
    multi.name <- multi.names[i]
    # get the hits
    multi.name.species <- unique(vdat_clean[vdat_clean[,multi.column] == multi.name, ref.column])
    # check if there's an exact match
    exact <- tolower(multi.name) == tolower(multi.name.species)
    if (sum(exact) > 0) {
      # if there are exact matches, prefer those
      vdat_clean <- vdat_clean[vdat_clean[,multi.column] != multi.name | (tolower(vdat_clean[,ref.column]) == tolower(vdat_clean[,multi.column])),]
    } else {
      # find candidate names in taxonomy
      possible.names <- n_extras[n_extras$taxid %in% vdat_clean$species_taxid[vdat_clean[,multi.column]==multi.name],]
      exact.ext <- tolower(multi.name) == tolower(possible.names$name)
      if (sum(exact.ext) > 0) {
        # if there are exact matches in synonyms, prefer those
        vdat_clean <- vdat_clean[vdat_clean[,multi.column] != multi.name | (vdat_clean$species_taxid %in% possible.names$taxid[exact.ext]),]
      } else {
        # get varieties
        possible.names.var <- possible.names[grepl(x = possible.names$name, pattern = " var."),]
        possible.names.edit <- possible.names[!grepl(x = possible.names$name, pattern = " var."),]
        # take first two words of non-vars
        possible.names.edit$name <- gsub(pattern = "([A-Za-z]+) ([A-Za-z]+) .+", replacement = "\\1 \\2", x = possible.names.edit$name)
        # sum edited names and unedited var names
        possible.names.edit <- rbind(possible.names.edit, possible.names.var)
        exact.ext.edit <- tolower(multi.name) == tolower(possible.names.edit$name)
        if (sum(exact.ext.edit) > 0) {
          # if there are exact matches in synonyms, prefer those
          vdat_clean <- vdat_clean[vdat_clean[,multi.column] != multi.name | (vdat_clean$species_taxid %in% possible.names.edit$taxid[exact.ext.edit]),]
        } else {
          # otherwise, take the first one
          vdat_clean <- vdat_clean[!(vdat_clean[,multi.column] == multi.name & duplicated(vdat_clean[,multi.column])),]
        }
      }
    }
  }
  return(vdat_clean)
}

resolve_species_taxid_list <- function(datfile){
  vdat_clean <- read.table(datfile, sep="\t", stringsAsFactors = F)
  colnames(vdat_clean) <- c("source.name", "source.taxid")
  vdat_uniq <- resolve_species_taxid_df(vdat_clean)
  return(vdat_uniq)
}

resolve_species_taxid_df <- function(vdat, filter_long=F){
  vdat_clean <- vdat
  # delete "[" "]"
  vdat_clean$source.name <- gsub("\\[|\\]", "", x = vdat_clean$source.name)
  # find species-level names
  vdat.spec <- do.call(rbind, lapply(vdat_clean$source.taxid, get_ancestor_at_rank, target_rank="species", graph=t))
  vdat.spec[,"Species"] <- n[match(vdat.spec$taxid, n$taxid), "name"]
  vdat_clean[,"Species"] <- vdat.spec$Species
  vdat_clean[,"species_taxid"] <- vdat.spec$taxid
  vdat_clean <- filter_species_names(vdat_clean)
  if (filter_long){
    vdat_clean <- filter_long_names(vdat_clean)
  }
  vdat_clean <- vdat_clean[!is.na(vdat_clean$Species),]
  
  # get duplicate matches from multiple source names
  multispecies.ids <- vdat_clean$Species %in% vdat_clean$Species[duplicated(vdat_clean$Species)]
  multispecies.names <- unique(vdat_clean[multispecies.ids, "Species"])
  
  if(sum(multispecies.ids)>0){
    # remove approximate matches and select the best one if possible
    vdat_clean <- resolve_multispecies(vdat_clean, multispecies.names, multi.column="Species", ref.column="source.name")
  }
  
  # remove duplicates
  vdat_uniq <- vdat_clean[!duplicated(vdat_clean[,"Species"]),]
  # reorder columns
  vdat_uniq <- vdat_uniq[,c(4,3,2,1)]
  # create additional columns to fill in later
  vdat_uniq[,c("human.pathogen", "animal.pathogen", "plant.pathogen", "plant.host",
               "putative.human.host", "putative.animal.host", "putative.plant.host")] <- FALSE
  vdat_uniq[,c("human.pathogen.source", "animal.pathogen.source", "plant.pathogen.source", "plant.host.source",
               "putative.human.host.source", "putative.animal.host.source", "putative.plant.host.source")] <- ""
  vdat_uniq$Species <- gsub("\\[|\\]", "", x = vdat_uniq$Species)
  return(vdat_uniq)
}

resolve_species_list <- function(datfile){
  speclist <- readLines(datfile)
  # delete "[" "]"
  speclist <- gsub("\\[|\\]", "", x = speclist)
  speclist <- gsub("\\s+", " ", x = speclist)
  speclist <- gsub("\\s+$", "", x = speclist)
  
  for (i in 1:nrow(typos)) {
    speclist[speclist==typos$source.name[i]] <- typos$taxonomy.name[i]
  }
  
  # find name matches in Taxonomy
  species.matched <- do.call(rbind, lapply(speclist, get_species_matches, graph=n_extras))
  
  # filter non-species names
  species_clean <- filter_nonspecies(species.matched, species_column = "name")
  species.wo.taxids <- species_clean[is.na(species_clean$taxid),]
  species_clean <- species_clean[!is.na(species_clean$taxid),]
  
  # find species-level names
  vdat.spec <- do.call(rbind, lapply(species_clean$taxid, get_ancestor_at_rank, target_rank="species", graph=t))
  vdat.spec[,"Species"] <- n_extras[match(vdat.spec$taxid, n_extras$taxid), "name"]
  
  # create the data frame
  vdat_clean <- data.frame(species_taxid=vdat.spec$taxid, Species=vdat.spec$Species, source.taxid="", source.name=species_clean$source.name,
                           human.pathogen=F, animal.pathogen=F, plant.pathogen=F, plant.host=F,
                           putative.human.host=F, putative.animal.host=F, putative.plant.host=F,
                           human.pathogen.source="", animal.pathogen.source="", plant.pathogen.source="", plant.host.source="",
                           putative.human.host.source="", putative.animal.host.source="", putative.plant.host.source="",
                           stringsAsFactors = FALSE)
  # filer out non-specific species
  vdat_clean <- filter_species_names(vdat_clean)
  vdat_clean <- vdat_clean[!duplicated(vdat_clean[, c("Species", "source.name")]),]
  
  # get ambiguous matches
  multispecies.source.ids <- vdat_clean$source.name %in% vdat_clean$source.name[duplicated(vdat_clean$source.name)]
  multispecies.source.names <- unique(vdat_clean[multispecies.source.ids, "source.name"])
  
  if(sum(multispecies.source.ids)>0){
    # remove approximate matches and select the best one if possible
    vdat_clean <- resolve_multispecies(vdat_clean, multispecies.source.names, multi.column="source.name", ref.column="Species")
    # add the names for which no appropriate match was found to the list of species without taxids
    missing.source.names <- multispecies.source.names[!(multispecies.source.names %in% vdat_clean$source.name)]
    if (length(missing.source.names)>0){
      species.wo.taxids <- rbind(species.wo.taxids, cbind.data.frame(source.name=missing.source.names, taxid=NA, name=missing.source.names, name_type=NA, stringsAsFactors = F))
    }
  }
  
  # get duplicate matches from multiple source names
  multispecies.ids <- vdat_clean$Species %in% vdat_clean$Species[duplicated(vdat_clean$Species)]
  multispecies.names <- unique(vdat_clean[multispecies.ids, "Species"])
  
  if(sum(multispecies.ids)>0){
    # remove approximate matches and select the best one if possible
    vdat_clean <- resolve_multispecies(vdat_clean, multispecies.names, multi.column="Species", ref.column="source.name")
    # add the names for which no appropriate match was found to the list of species without taxids
    missing.names <- multispecies.names[!(multispecies.names %in% vdat_clean$Species)]
    if (length(missing.names)>0){
      species.wo.taxids <- rbind(species.wo.taxids, cbind.data.frame(source.name=missing.names, taxid=NA, name=missing.names, name_type=NA, stringsAsFactors = F))
    }
  }
  
  if (nrow(species.wo.taxids)>0){
    # if some species weren't found, create a df for them
    vdat_wo_taxids <- data.frame(species_taxid="", Species=species.wo.taxids$source.name, source.taxid="", source.name=species.wo.taxids$source.name,
                                 human.pathogen=F, animal.pathogen=F, plant.pathogen=F, plant.host=F,
                                 putative.human.host=F, putative.animal.host=F, putative.plant.host=F,
                                 human.pathogen.source="", animal.pathogen.source="", plant.pathogen.source="", plant.host.source="",
                                 putative.human.host.source="", putative.animal.host.source="", putative.plant.host.source="",
                                 stringsAsFactors = FALSE)
    vdat_wo_taxids <- filter_species_names(vdat_wo_taxids)
    
    # ... deduplicate and join
    vdat_uniq <- vdat_clean[!duplicated(vdat_clean[,"Species"]),]
    vdat_uniq <- rbind(vdat_uniq, vdat_wo_taxids)
  } else {
    # or just deduplicate if all species found
    vdat_uniq <- vdat_clean[!duplicated(vdat_clean[,"Species"]),]
  }
  # remove brackets
  vdat_uniq$Species <- gsub("\\[|\\]", "", x = vdat_uniq$Species)
  return(vdat_uniq)
}

add_single_label_resource <- function(old_data, new_data, citation, label_name){
  vdat_all <- old_data
  # update the selected label for all Species names which are found in the new resource
  vdat_all[vdat_all$Species %in% new_data$Species, label_name] <- TRUE
  # remove brackets
  vdat_all$Species <- gsub("\\[|\\]", "", x = vdat_all$Species)
  # bind but remove the duplicates (new ones will be the duplicates if they occurred before, but the labels for the old ones have been updated)
  vdat_all <- rbind(vdat_all, new_data)
  vdat_all <- vdat_all[!duplicated(vdat_all$Species),]
  # get all citations
  old_citations <- vdat_all[, paste0(label_name, ".source")]
  # if no citation is present for a species present in the new resource, add the new citations
  vdat_all[vdat_all$Species %in% new_data$Species & old_citations == "", paste0(label_name, ".source")] <- citation
  # if a citation is present, add the new citations to it
  vdat_all[vdat_all$Species %in% new_data$Species & old_citations != "", paste0(label_name, ".source")] <- paste0(vdat_all[vdat_all$Species %in% new_data$Species & old_citations != "", paste0(label_name, ".source")], "; ", citation)
  return(vdat_all)
}

add_custom_citations <- function(vdat, citations, vdat.source.column, citations.source.column){
  # add a temporary column by merging the data with the citation table by Species name
  vdat.temp <- merge(vdat, citations, by="Species", all.x = TRUE)
  # change NAs to empty strings just in case, if any are present
  vdat.temp[is.na(vdat.temp[,citations.source.column]), citations.source.column] <- ""
  # add the content of the temp column to the proper citation column
  # no semicolon necessary, as it is added by add_single_label_resource called with an empty string as a citation
  vdat.temp[,vdat.source.column] <- paste0(vdat.temp[,vdat.source.column], vdat.temp[,citations.source.column])
  vdat.temp[,citations.source.column] <- NULL
  return(vdat.temp)
}
