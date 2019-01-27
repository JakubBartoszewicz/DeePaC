library(stringr)
run.comparison <- FALSE
ids.curation <- TRUE
find.better <- FALSE
remove.duplicates.synonyms <- FALSE

date.postfix.gb <- "_160119"
date.postfix.img <- "_160119_NEW"

date.limit <- as.Date("2019/01/16", format = "%Y/%m/%d")

genbank.path <- paste0("assembly_summary_genbank", date.postfix.gb, ".txt")
img.path <- paste0("IMG", date.postfix.img, "_filtered_HPNHP_resolved.rds")

### Set paths and load
genbank_backup <- read.csv(genbank.path, sep = "\t")
IMGrequest <- readRDS(img.path)
# reorder date factors for sorting: newest = best
IMGrequest$Add.Date <- as.Date(IMGrequest$Add.Date, format = "%d.%m.%Y")
# clip seq release date to the limit
IMGrequest <- IMGrequest[IMGrequest$Add.Date <= date.limit,]

### Prepare GenBank ###
# load genbank data
genbank <- genbank_backup
# backup original name
genbank$organism_name.gb <- genbank$organism_name
# cast to character for processing
genbank$organism_name <- as.character(genbank$organism_name)
genbank$infraspecific_name <- as.character(genbank$infraspecific_name)
# define fixing organism names
add.strain.name <- function(i, gb){
    if (!is.na(gb$infraspecific_name[i]) & nchar(gb$infraspecific_name[i]) > 0){    
        # split by parentheses, semicolons and substrains
        strain.names <- strsplit(gsub("strain=", "", gb$infraspecific_name[i]), split = "\\(|\\)|;|substr\\.")[[1]]
        # remove trailing spaces and =
        strain.names <- gsub("^\\s+|\\s+$|=", "", strain.names)
        strain.names <- strain.names[nchar(strain.names)>0]
        strain.names <- sapply(strain.names, function(strain){str_detect(string = gb$organism_name[i], pattern = fixed(strain, ignore_case = TRUE))})
        to.add <- paste(names(strain.names)[!strain.names], collapse =" ")
        if (nchar(to.add) > 0)
            return(paste(gsub(" str\\.", "", gb$organism_name[i]), to.add))
        else
            return(gsub(" str\\.", "", gb$organism_name[i]))            
    } else {
        # otherwise, don't change
        return(gsub(" str\\.", "", gb$organism_name[i]))   
    }
}
# fix organism names
genbank$organism_name <- sapply(1:nrow(genbank), add.strain.name, gb = genbank)
genbank <- genbank[, c("assembly_accession", "bioproject", "taxid", "species_taxid", "organism_name", "organism_name.gb", "infraspecific_name", "version_status", "assembly_level", "seq_rel_date", "gbrs_paired_asm", "paired_asm_comp", "ftp_path")]

# reorder assembly level factors for sorting: complete genome = best
genbank$assembly_level <- factor(genbank$assembly_level,levels(genbank$assembly_level)[c(2,1,4,3)])
# reorder paired_asm_comp factors for sorting: identical = best
genbank$paired_asm_comp <- factor(genbank$paired_asm_comp,levels(genbank$paired_asm_comp)[c(2,1,3)])
# reorder date factors for sorting: newest = best
genbank$seq_rel_date <- as.Date(genbank$seq_rel_date, format = "%Y/%m/%d")
# clip seq release date to the limit
genbank <- genbank[genbank$seq_rel_date <= date.limit,]
# delete "[" "]"
genbank$organism_name <- gsub("\\[|\\]", "", x = genbank$organism_name)


### data binding ###
# load IMG request data
IMGrequest <- IMGrequest[,c("IMG.Genome.ID", "NCBI.Bioproject.Accession", "Species", "Strain", "Genome.Name...Sample.Name", "Add.Date", "Pathogenic")]
IMGrequest$bioproject <- as.character(IMGrequest$NCBI.Bioproject.Accession)
IMGrequest$Strain <- as.character(IMGrequest$Strain)
IMGrequest$Species <- as.character(IMGrequest$Species)
IMGrequest$Genome.Name...Sample.Name <- as.character(IMGrequest$Genome.Name...Sample.Name)
# backup species data
IMGrequest$Species.img <- IMGrequest$Species
# delete "[" "]"
IMGrequest$Species <- gsub("\\[|\\]", "", x = IMGrequest$Species)
# order IMG request data
IMGrequest$order.id <- 1:nrow(IMGrequest)

# merge
genbank.img <- merge(genbank, IMGrequest, by = "bioproject")
# inspect failures
INSPECT_fail_orig <- IMGrequest[!(IMGrequest$bioproject %in% genbank.img$bioproject), ]

if(ids.curation){
    ### manual curation ###
    # manually update IDs
    old.ids <- c("PRJNA195867", "PRJNA245642", "PRJNA32601", "PRJNA42503", "PRJNA32179")
    new.ids <- c("PRJEB16486", "PRJEB18154", "PRJEB20240", "PRJNA20115", "PRJNA239703")
    names(new.ids) <- old.ids
    # define updating
    update.ids <- function(b, update.list){
        if (b %in% names(update.list)){
            return (as.character(update.list[b]))
        } else {
            return(b)
        }
    } 
    # update
    IMGrequest$bioproject <- sapply(IMGrequest$bioproject, update.ids, update.list = new.ids)

    # get genbank data matching updated Bioproject IDs
    genbank.img <- merge(genbank, IMGrequest, by = "bioproject")

    # inspect failures
    INSPECT_fail_corrected <- IMGrequest[!(IMGrequest$bioproject %in% genbank.img$bioproject), ]
}

### handle Bioprojects with multiple species ###

# get ambiguous bioprojects
multispecies.ids <- genbank.img$bioproject %in% genbank.img$bioproject[duplicated(genbank.img$bioproject)]
multispecies <- genbank.img[multispecies.ids,]
# get unambiguous bioprojects
singlespecies <- genbank.img[!multispecies.ids,]
# define matching of species names
match.species <- function(i, gb, matched.list = NULL){
    if (gb$organism_name[i] %in% matched.list)
        return (TRUE)
    else
        str_detect(pattern = regex(gb$Species[i], ignore_case = TRUE) , string = gb$organism_name[i])
} 
# define matching of strain names
match.strains <- function(i, gb, matched.list = NULL){
    if (gb$organism_name[i] %in% matched.list)
    # accept trusted organism names directly matched from IMG (after they were already confirmed)
        return (TRUE)
    else {
        if (!is.na(gb$infraspecific_name[i]) & nchar(gb$infraspecific_name[i]) > 0)
            gb.match.strain <- (str_detect(pattern = fixed(gsub("strain=", " ", gsub("-", "", gb$infraspecific_name[i])), ignore_case = TRUE), string = gsub("-", "", gb$Genome.Name...Sample.Name[i])) |
            str_detect(pattern = fixed(gsub("-", "", gb$organism_name[i]), ignore_case = TRUE), string = gsub("-", "", gb$Genome.Name...Sample.Name[i])))
        else
            gb.match.strain <- FALSE
        if (!is.na(gb$Strain[i]) & nchar(gb$Strain[i]) > 0)
            img.match.strain <- (str_detect(pattern = fixed(gsub("-", "", gb$Strain[i]), ignore_case = TRUE), string = gsub("-", "", gb$infraspecific_name[i])) |
            str_detect(pattern = fixed(gsub("-", "", gb$Strain[i]), ignore_case = TRUE), string = gsub("-", "", gb$organism_name[i])))
        else 
            img.match.strain <- FALSE
        return (gb.match.strain | img.match.strain)
    }
}
if(nrow(multispecies)>0){
    # get matching organisms from ambiguous bioprojects
    multispecies.resolved <- multispecies[sapply(1:nrow(multispecies), match.species, gb = multispecies) & sapply(1:nrow(multispecies), match.strains, gb = multispecies),]

    # sort by refseq, assembly level and date (all data have version_status=='latest'. otherwise it should be sorted here too)
    multispecies.sorted <- multispecies.resolved[order(multispecies.resolved$bioproject, multispecies.resolved$paired_asm_comp, multispecies.resolved$assembly_level, multispecies.resolved$seq_rel_date),]
    # save best
    multispecies.selected <- multispecies.sorted[!duplicated(multispecies.sorted$bioproject),]

    # bind unambiguous with resolved
    selected.species <- rbind(singlespecies, multispecies.selected)
} else {
    selected.species <- singlespecies
}

if(find.better){
    ### find candidates for better assemblies in GenBank by organism name
    genbank.extended.gb <- merge(genbank, selected.species, by = "organism_name", all.y=TRUE, suffixes = c("", ".orig"), sort = FALSE)
    genbank.extended.gb$LinkedWithIMG <- FALSE
    genbank.extended.gb$LinkedWithGB <- TRUE
    genbank.extended.gb$LinkedWithIMG[genbank.extended.gb$bioproject == genbank.extended.gb$bioproject.orig] <- TRUE

    ### find candidates for better assemblies in GenBank by species + strain name
    selected.species.strains <- selected.species[nchar(selected.species$Strain) > 0,]
    genbank$Species.Strain <- genbank$organism_name
    selected.species.strains$Species.Strain <- trimws(paste(selected.species.strains$Species, selected.species.strains$Strain))
    genbank.extended.img <- merge(genbank, selected.species.strains, by = "Species.Strain", suffixes = c("", ".orig"), sort = FALSE)
    # rename columns etc.
    genbank$Species.Strain <- NULL
    genbank.extended.img$organism_name <- NULL
    colnames(genbank.extended.img)[colnames(genbank.extended.img)=="Species.Strain"] <- "organism_name"
    genbank.extended.img$organism_name <- genbank.extended.img$organism_name.orig
    genbank.extended.img$organism_name.orig <- NULL

    # update flags in genbank links
    genbank.extended.gb$LinkedWithIMG[genbank.extended.gb$assembly_accession %in% genbank.extended.img$assembly_accession] <- TRUE

    # select only new rows and inspect 
    genbank.extended.img <- genbank.extended.img[!(genbank.extended.img$assembly_accession %in% genbank.extended.gb$assembly_accession),]
    genbank.extended.img$LinkedWithIMG <- TRUE
    genbank.extended.img$LinkedWithGB <- FALSE
    INSPECT_genbank.extended.img <- genbank.extended.img

    ### resolve problems like Mycobacterium bovis BCG, where IMG strain info is too general (matching multible strains in GB)
    # delete empty infraspecific names (not enough info for a link)
    genbank.extended.img.resolved <- genbank.extended.img[nchar(genbank.extended.img$infraspecific_name) > 0,]
    # find the candidate infraspecific name in the original infraspecific name (matching to IMG Strain or Genome/Sample name is *not* enough, as in the case of Mycobacterium bovis BCG.)
    # can miss some synonyms, e.g. Clostridioides difficile VPI 10463 / ATCC 43255 (if infraspecific_name in Strain and infraspecific_name.orig in Genome/Sample name), but otherwise false positives would be accepted
    # only one missed synonym known, and not a good genome (no refseq & contigs where alternative with chromosome available)
    genbank.extended.img.resolved <- genbank.extended.img.resolved[str_detect(string = gsub("strain=|-|\\s+", "", genbank.extended.img.resolved$infraspecific_name.orig), pattern = fixed(gsub("strain=|-|\\s+", "", genbank.extended.img.resolved$infraspecific_name), ignore_case = TRUE)),]

    # delete substrains unless identical to orig
    genbank.extended.img.resolved <- genbank.extended.img.resolved[!str_detect(string = gsub("strain=|-|\\s+", "", genbank.extended.img.resolved$infraspecific_name), pattern = fixed("substr")) | genbank.extended.img.resolved$infraspecific_name == genbank.extended.img.resolved$infraspecific_name.orig,]


    # inspect removed candidates
    INSPECT_genbank.extended.img.removed <- genbank.extended.img[!(genbank.extended.img$assembly_accession %in% genbank.extended.img.resolved$assembly_accession),]

    ### bind
    genbank.selected <- rbind(genbank.extended.gb, genbank.extended.img.resolved)
} else {
    genbank.selected <- selected.species
    genbank.selected$bioproject.orig <- genbank.selected$bioproject
    genbank.selected$assembly_accession.orig <- genbank.selected$assembly_accession
}

### handle synonyms and species with multiple Bioprojects ###
if(remove.duplicates.synonyms) {
    # get ambiguous bioprojects
    multiproject.ids <- genbank.selected$organism_name %in% genbank.selected$organism_name[duplicated(genbank.selected$organism_name)]
    multiproject <- genbank.selected[multiproject.ids,]
    # get unambiguous bioprojects
    singleproject <- genbank.selected[!multiproject.ids,]
    # get matching organisms from ambiguous bioprojects
    multiproject.resolved <- multiproject[sapply(1:nrow(multiproject), match.species, gb = multiproject, matched.list = selected.species$organism_name) & sapply(1:nrow(multiproject), match.strains, gb = multiproject, matched.list = selected.species$organism_name),]


    # sort
    multiproject.sorted <- multiproject.resolved[order(multiproject.resolved$organism_name, multiproject.resolved$paired_asm_comp, multiproject.resolved$assembly_level, multiproject.resolved$seq_rel_date),]
    # save best
    multiproject.selected <- multiproject.sorted[!duplicated(multiproject.sorted$organism_name),]
    # bind
    selected.projects <- rbind(singleproject, multiproject.selected)


    ### handle duplicated genome names / sample names ###

    # get ambiguous names
    multiname.ids <- selected.projects$Genome.Name...Sample.Name %in% selected.projects$Genome.Name...Sample.Name[duplicated(selected.projects$Genome.Name...Sample.Name)]
    multiname <- selected.projects[multiname.ids,]
    # get unambiguous names
    singlename <- selected.projects[!multiname.ids,]

    # get duplicates with substrains - and keep them (important design choice, watch out!)
    multiname.substr.ids <- multiname$Genome.Name...Sample.Name %in% multiname$Genome.Name...Sample.Name[str_detect(string = multiname$infraspecific_name, pattern = "substr\\.")]
    multiname.substr <- multiname[multiname.substr.ids,]
    multiname.resolved <- multiname[!multiname.substr.ids,]

    # sort
    multiname.sorted <- multiname.resolved[order(multiname.resolved$Genome.Name...Sample.Name, multiname.resolved$paired_asm_comp, multiname.resolved$assembly_level, multiname.resolved$seq_rel_date),]
    # save best
    multiname.selected <- multiname.sorted[!duplicated(multiname.sorted$Genome.Name...Sample.Name),]
    # add substrains
    multiname.selected <- rbind(multiname.substr, multiname.selected)

    # bind
    selected.names <- rbind(singlename, multiname.selected)


    ### handle duplicated species+strain names ###

    # get ambiguous ss
    selected.names$Species.Strain <- trimws(paste(selected.names$Species, selected.names$Strain))
    multiss.ids <- selected.names$Species.Strain %in% selected.names$Species.Strain[duplicated(selected.names$Species.Strain)]
    multiss.ids[nchar(selected.names$Strain) <= 0] <- FALSE
    multiss <- selected.names[multiss.ids,]
    # get unambiguous ss
    singless <- selected.names[!multiss.ids,]

    # get duplicates with substrains - and keep them (important design choice, watch out!)
    multiss.substr.ids <- multiss$Species.Strain %in% multiss$Species.Strain[str_detect(string = multiss$infraspecific_name, pattern = "substr\\.")]
    multiss.substr <- multiss[multiss.substr.ids,]
    multiss.resolved <- multiss[!multiss.substr.ids,]

    # sort
    multiss.sorted <- multiss.resolved[order(multiss.resolved$Species.Strain, multiss.resolved$paired_asm_comp, multiss.resolved$assembly_level, multiss.resolved$seq_rel_date),]
    # save best
    multiss.selected <- multiss.sorted[!duplicated(multiss.sorted$Species.Strain),]
    # add substrains
    multiss.selected <- rbind(multiss.substr, multiss.selected)

    # bind
    selected.ss <- rbind(singless, multiss.selected)
    selected.ss$Species.Strain <- NULL

    selected.final <- selected.ss
} else {
    selected.final <- genbank.selected
}
# retrieve original species names
selected.final$Species <- selected.final$Species.img
selected.final$Species.img  <- NULL

# order by request order
selected.final <- selected.final[order(selected.final$order.id),]
selected.final$order.id <- NULL

# summary of all selected organisms
selected.summary <- selected.final[,c("organism_name","bioproject","bioproject.orig", "organism_name.gb","infraspecific_name",  "Species", "Strain", "Genome.Name...Sample.Name")]

### Save
saveRDS(selected.final, paste0("IMG_assemblies",date.postfix.img,".rds"))

### inspect failures
INSPECT_removed_final <- IMGrequest[!(IMGrequest$bioproject %in% selected.final$bioproject), ]

# inspect unmatched species (corresponding to name changes/ambiguities, but still unique identifiers, checked that)
INSPECT_selected.name.changes <- selected.final[!sapply(1:nrow(selected.final), match.species, gb = selected.final),]
INSPECT_selected.name.changes.summary <- selected.summary[!sapply(1:nrow(selected.final), match.species, gb = selected.final),]

# inspect genomes found by imgLinkR
INSPECT_selected.smart <- selected.final[as.character(selected.final$bioproject) != as.character(selected.final$bioproject.orig),]
INSPECT_selected.summary.smart <- selected.summary[as.character(selected.summary$bioproject) != as.character(selected.summary$bioproject.orig),]

# inspect species that could not be downloaded. Don't do that with strains - because of "smart" matching it won't work. Have a look at INSPECT_fail_corrected instead.
INSPECT_fail_species <- IMGrequest[!(IMGrequest$Species %in% selected.final$Species), ]

Species_HP <- unique(selected.final$Species[selected.final$Pathogenic])
Species_NHP <- unique(selected.final$Species[!selected.final$Pathogenic])
Species_Mix <- Species_HP[Species_HP %in% Species_NHP]
Species_HP <- setdiff(Species_HP, Species_Mix)
Species_NHP <- setdiff(Species_NHP, Species_Mix)

print("Linked species:")
print(c(HP.all = length(Species_HP) + length(Species_Mix), HP.pure = length(Species_HP), HNP = length(Species_NHP), HMix = length(Species_Mix), Sum = length(unique(selected.final$Species))))
print("Linked strains:")
print(c(HP = sum(selected.final$Pathogenic), HNP = sum(!selected.final$Pathogenic), Sum = length(selected.final$Pathogenic)))


### Comparisons with older methods

if (run.comparison){
    # load phylab for comparison
    phylab <- read.csv2("phylab-acc.csv", header = F)
    # load nuccore downloading for comparison
    IMGdownloaded <- readRDS("IMG_all_genomes_170418_taxontable128751_filtered_ruled_HPNHP_resolved_inc_downloaded.rds")
    IMGdownloaded$bioproject <- as.character(IMGdownloaded$NCBI.Bioproject.Accession)
    # correct the typo
    levels(IMGdownloaded$NCBI.Bioproject.Accession)[levels(IMGdownloaded$NCBI.Bioproject.Accession)=="PRJNA20115 "] <- "PRJNA20115"
    # update
    if(ids.curation) {
        IMGdownloaded$bioproject <- sapply(IMGdownloaded$bioproject, update.ids, update.list = new.ids)
    }

    ## compare to nuccore downloading

    added_corrected <- setdiff(genbank.img$bioproject, IMGdownloaded$NCBI.Bioproject.Accession)
    removed_corrected <- setdiff(IMGdownloaded$NCBI.Bioproject.Accession, genbank.img$bioproject)
    #> added_corrected
    #[1] "PRJNA18279" (= "PRJEA70285")
    #> removed_corrected
    #[1] "PRJNA32179" (= "PRJNA239703") "PRJNA61377" (plasmid) "PRJEA70285" (= "PRJNA18279")

    added_final <- setdiff(selected.final$bioproject, IMGdownloaded$bioproject)
    removed_final <- setdiff(IMGdownloaded$bioproject, selected.final$bioproject)
    #> added_final
    #[1] "PRJNA18279" (= "PRJEA70285")
    #> removed_final
    #[1]
    removed_img <- IMGdownloaded[IMGdownloaded$bioproject %in% removed_final,c("bioproject", "NCBI.Bioproject.Accession", "Species", "Strain", "Genome.Name...Sample.Name", "Pathogenic")]
    removed_really <- setdiff(IMGdownloaded$bioproject, selected.final$bioproject.orig)
    added_really<- setdiff(selected.final$bioproject.orig, IMGdownloaded$bioproject)
    removed_img_really <- IMGdownloaded[IMGdownloaded$bioproject %in% removed_really,c("bioproject", "NCBI.Bioproject.Accession", "Species", "Strain", "Genome.Name...Sample.Name", "Pathogenic")]


    ### compare to phylabelle

    genbank.phylab <- genbank[genbank$assembly_accession %in% phylab$V1,]
    acc_only_imgLinkR <- selected.final[selected.final$assembly_accession %in% setdiff(selected.final$assembly_accession, phylab$V1), ]
    acc_only_phylab <- genbank[genbank$assembly_accession %in% setdiff(phylab$V1, selected.final$assembly_accession), ]
    acc_only_imgLinkR_img <- selected.final[selected.final$assembly_accession %in% setdiff(selected.final$assembly_accession, phylab$V1), ]
    acc_only_phylab_img <- genbank.selected[genbank.selected$assembly_accession %in% setdiff(phylab$V1, selected.final$assembly_accession), ]
    acc_only_phylab_img <- acc_only_phylab_img[!duplicated(acc_only_phylab_img$assembly_accession),]
    acc_only_imgLinkR <- merge(acc_only_imgLinkR, acc_only_imgLinkR_img, all.x = TRUE)
    acc_only_phylab <- merge(acc_only_phylab, acc_only_phylab_img, all.x = TRUE)
    acc_only_phylab <- acc_only_phylab[!duplicated(acc_only_phylab$assembly_accession),]
    # organism 
    phylab.diff.summary <- merge(acc_only_phylab_img[,c("organism_name", "bioproject", "organism_name.gb", "infraspecific_name", "assembly_level", "seq_rel_date", "paired_asm_comp")], acc_only_imgLinkR_img[,c("organism_name", "bioproject", "organism_name.gb", "infraspecific_name", "assembly_level", "seq_rel_date", "paired_asm_comp")], all= TRUE, by="organism_name", suffixes = c(".phylab", ".imglinkr"))
    phylab.upgrades.summary <- merge(genbank.phylab[,c("organism_name", "bioproject", "organism_name.gb", "infraspecific_name", "assembly_level", "seq_rel_date", "paired_asm_comp")], acc_only_imgLinkR_img[,c("organism_name", "bioproject", "organism_name.gb", "infraspecific_name", "assembly_level", "seq_rel_date", "paired_asm_comp")], all.y= TRUE, by="organism_name", suffixes = c(".phylab", ".imglinkr"))



    ### compare to best imgLinkR
    selected.gold <- readRDS("IMG_APRIL/selected_final-49-new.rds")
    diffgold <- setdiff(selected.gold$assembly_accession, selected.final$assembly_accession)
    diffgold_img <- genbank[genbank$assembly_accession %in% diffgold,]
    diffgold_img <- diffgold_img[!duplicated(diffgold_img$assembly_accession),]
    rdiffgold <- setdiff(selected.final$assembly_accession, selected.gold$assembly_accession)
    rdiffgold_img <- genbank[genbank$assembly_accession %in% rdiffgold,]
    rdiffgold_img <- rdiffgold_img[!duplicated(rdiffgold_img$assembly_accession),]
    diffrdiff.summary <- merge(rdiffgold_img[,c("organism_name", "bioproject", "organism_name.gb", "infraspecific_name", "assembly_level", "seq_rel_date", "paired_asm_comp")], diffgold_img[,c("organism_name", "bioproject", "organism_name.gb", "infraspecific_name", "assembly_level", "seq_rel_date", "paired_asm_comp")], by = "organism_name", suffixes=c(".rdiff", ".diff"), all=TRUE)
}

