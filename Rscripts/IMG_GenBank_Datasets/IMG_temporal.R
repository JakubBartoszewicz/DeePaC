img_old <- readRDS("IMG_170418_filtered_HPNHP_resolved.rds")
img_new <- readRDS("IMG_160119_filtered_HPNHP_resolved.rds")
new <- img_new[!(img_new$taxon_oid %in% img_old$taxon_oid) & !(img_new$Species %in% img_old$Species) & as.character(img_new$Species)!="", ]

saveRDS(new, "IMG_160119_NEW_filtered_HPNHP_resolved.rds")