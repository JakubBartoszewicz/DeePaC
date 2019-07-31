# no of folds to generate
k <- 9
fold_names <- paste0("fold", 1:k)
# no of folds to download
k_download <- 1

human <- readRDS("IMG_all_folds_human.rds")
n.human.cho <- readRDS("IMG_all_folds_chordata.rds")
n.chordata.met <- readRDS("IMG_all_folds_metazoa.rds")
n.metazoa.euk <- readRDS("IMG_all_folds_eukarya.rds")
n.eukarya.all <- readRDS("IMG_all_folds_neukarya.rds")

n.human.met <- rbind(n.human.cho, n.chordata.met)
n.human.euk <- rbind(n.human.met, n.metazoa.euk)
n.human.all <- rbind(n.human.euk, n.eukarya.all)

save.selected <- function(dataset, postfix, fold_names, k_download){
    selected <- dataset[dataset$subset == "selected",]
    # only folds to download
    if(k>1 & k_download<k){
        selected[,fold_names[(1+k_download):length(fold_names)]] <- NULL
    }

    # Save data for backup
    saveRDS(dataset, paste0("VHDB_all_folds", postfix, ".rds"))
    saveRDS(selected, paste0("VHDB_", k_download, "_folds", postfix, ".rds"))
}

# Saving Humans and Chordata is unnessecary, but for completeness...
save.selected(human, "_human", fold_names, k_download)
save.selected(n.human.cho, "_chordata_nhuman", fold_names, k_download)
save.selected(n.human.met, "_metazoa_nhuman", fold_names, k_download)
save.selected(n.human.euk, "_eukarya_nhuman", fold_names, k_download)
save.selected(n.human.all, "_all_nhuman", fold_names, k_download)