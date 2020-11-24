library(plyr)
# no of folds to generate
k <- 9
fold_names <- paste0("fold", 1:k)
# no of folds to download
k_download <- 1
ambiguous.species <- TRUE

human <- readRDS("IMG_all_folds_human_species.rds")
n.human.cho <- readRDS("IMG_all_folds_chordata_species.rds")
n.chordata.met <- readRDS("IMG_all_folds_metazoa_species.rds")
n.metazoa.euk <- readRDS("IMG_all_folds_eukarya_species.rds")
n.eukarya.all <- readRDS("IMG_all_folds_neukarya_species.rds")

if (ambiguous.species){
    species.ref <- human[!duplicated(human$Species), c("Species", "Species.taxid", fold_names)]
    n.human.cho.amb <- n.human.cho[n.human.cho$Ambiguous, c(1:21, ncol(n.human.cho))]
    n.human.cho.amb <- join(n.human.cho.amb, species.ref[,2:ncol(species.ref)], by="Species.taxid")
    n.human.cho[n.human.cho$Ambiguous, fold_names] <- n.human.cho.amb[fold_names]

    unamb <- rbind(human, n.human.cho)
    species.ref <- unamb[!duplicated(unamb$Species), c("Species", "Species.taxid", fold_names)]
    n.chordata.met.amb <- n.chordata.met[n.chordata.met$Ambiguous, c(1:21, ncol(n.chordata.met))]
    n.chordata.met.amb <- join(n.chordata.met.amb, species.ref[,2:ncol(species.ref)], by="Species.taxid")
    n.chordata.met[n.chordata.met$Ambiguous, fold_names] <- n.chordata.met.amb[fold_names]

    unamb <- rbind(unamb, n.chordata.met)
    species.ref <- unamb[!duplicated(unamb$Species), c("Species", "Species.taxid", fold_names)]
    n.metazoa.euk.amb <- n.metazoa.euk[n.metazoa.euk$Ambiguous, c(1:21, ncol(n.metazoa.euk))]
    n.metazoa.euk.amb <- join(n.metazoa.euk.amb, species.ref[,2:ncol(species.ref)], by="Species.taxid")
    n.metazoa.euk[n.metazoa.euk$Ambiguous, fold_names] <- n.metazoa.euk.amb[fold_names]

    unamb <- rbind(unamb, n.metazoa.euk)
    species.ref <- unamb[!duplicated(unamb$Species), c("Species", "Species.taxid", fold_names)]
    n.eukarya.all.amb <- n.eukarya.all[n.eukarya.all$Ambiguous, c(1:21, ncol(n.eukarya.all))]
    n.eukarya.all.amb <- join(n.eukarya.all.amb, species.ref[,2:ncol(species.ref)], by="Species.taxid")
    n.eukarya.all[n.eukarya.all$Ambiguous, fold_names] <- n.eukarya.all.amb[fold_names]

}

n.human.met <- rbind(n.human.cho, n.chordata.met)
n.human.euk <- rbind(n.human.met, n.metazoa.euk)
n.human.all <- rbind(n.human.euk, n.eukarya.all)

save.selected <- function(dataset, postfix, fold_names, k_download){
    selected <- dataset[dataset$subset == "selected",]
    # only folds to download
    if(k>1 & k_download<k){
        for (fold_name in fold_names[(1+k_download):length(fold_names)]) {
            selected[,fold_name] <- NULL
        }
    }

    # Save data for backup
    saveRDS(dataset, paste0("VHDB_all_folds", postfix, ".rds"))
    saveRDS(selected, paste0("VHDB_", k_download, "_folds", postfix, ".rds"))
}

# Saving Humans and Chordata is unnessecary, but for completeness...
save.selected(human, "_human_species", fold_names, k_download)
save.selected(n.human.cho, "_chordata_nhuman_species", fold_names, k_download)
save.selected(n.human.met, "_metazoa_nhuman_species", fold_names, k_download)
save.selected(n.human.euk, "_eukarya_nhuman_species", fold_names, k_download)
save.selected(n.human.all, "_all_nhuman_species", fold_names, k_download)