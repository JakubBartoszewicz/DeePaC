dfvf_hp <- read.csv('FUNGI_DATA_CUR/labels_final/dfvf_hp.csv', stringsAsFactors = F)
dfvf_an <- read.csv('FUNGI_DATA_CUR/labels_final/dfvf_an.csv', stringsAsFactors = F)
dfvf_pl <- read.csv('FUNGI_DATA_CUR/labels_final/dfvf_pl.csv', stringsAsFactors = F)
taylor <- read.csv('FUNGI_DATA_CUR/labels_final/taylor.csv', stringsAsFactors = F)
vacher  <- read.csv('FUNGI_DATA_CUR/labels_final/vacher.csv', stringsAsFactors = F)
doehlemann  <- read.csv('FUNGI_DATA_CUR/labels_final/doehlemann.csv', stringsAsFactors = F)
smith  <- read.csv('FUNGI_DATA_CUR/labels_final/smith.csv', stringsAsFactors = F)
seyedmousavi_an  <- read.csv('FUNGI_DATA_CUR/labels_final/seyedmousavi_animal.csv', stringsAsFactors = F)
seyedmousavi_hp  <- read.csv('FUNGI_DATA_CUR/labels_final/seyedmousavi_human.csv', stringsAsFactors = F)
reviews_hp  <- read.csv('FUNGI_DATA_CUR/labels_final/reviews_human.csv', stringsAsFactors = F)
reviews_an  <- read.csv('FUNGI_DATA_CUR/labels_final/reviews_animal.csv', stringsAsFactors = F)
reviews_pl  <- read.csv('FUNGI_DATA_CUR/labels_final/reviews_plant.csv', stringsAsFactors = F)
grin_hp  <- read.csv('FUNGI_DATA_CUR/labels_final/grin_human.csv', stringsAsFactors = F)
grin_pl1  <- read.csv('FUNGI_DATA_CUR/labels_final/grin_diag.csv', stringsAsFactors = F)
grin_pl2  <- read.csv('FUNGI_DATA_CUR/labels_final/grin_plant2.csv', stringsAsFactors = F)
tax_hp  <- read.csv('FUNGI_DATA_CUR/labels_final/tax_search.csv', stringsAsFactors = F)
wardeh_hp  <- read.csv('FUNGI_DATA_CUR/labels_final/wardeh_hp.csv', stringsAsFactors = F)
wardeh_an  <- read.csv('FUNGI_DATA_CUR/labels_final/wardeh_an.csv', stringsAsFactors = F)
wardeh_pl  <- read.csv('FUNGI_DATA_CUR/labels_final/wardeh_pl.csv', stringsAsFactors = F)

atlas_ours_hp <- read.csv('FUNGI_DATA_CUR/labels_final/atlas_ours_hp.csv', stringsAsFactors = F)
atlas_ours_an <- read.csv('FUNGI_DATA_CUR/labels_final/atlas_ours_an.csv', stringsAsFactors = F)
atlas_ours_pl <- read.csv('FUNGI_DATA_CUR/labels_final/atlas_ours_pl.csv', stringsAsFactors = F)
atlas_wardeh_hp <- read.csv('FUNGI_DATA_CUR/labels_final/atlas_wardeh_hp.csv', stringsAsFactors = F)
atlas_wardeh_an <- read.csv('FUNGI_DATA_CUR/labels_final/atlas_wardeh_an.csv', stringsAsFactors = F)
atlas_wardeh_pl <- read.csv('FUNGI_DATA_CUR/labels_final/atlas_wardeh_pl.csv', stringsAsFactors = F)

dfvf_hp[is.na(dfvf_hp)] <- ""
dfvf_an[is.na(dfvf_an)] <- "" 
dfvf_pl[is.na(dfvf_pl)] <- "" 
taylor[is.na(taylor)] <- "" 
vacher[is.na(vacher)] <- "" 
doehlemann[is.na(doehlemann)] <- "" 
smith[is.na(smith)] <- "" 
seyedmousavi_an[is.na(seyedmousavi_an)] <- "" 
seyedmousavi_hp[is.na(seyedmousavi_hp)] <- "" 
reviews_hp[is.na(reviews_hp)] <- "" 
reviews_an[is.na(reviews_an)] <- ""
reviews_pl[is.na(reviews_pl)] <- "" 
grin_hp[is.na(grin_hp)] <- "" 
grin_pl1[is.na(grin_pl1)] <- "" 
grin_pl2[is.na(grin_pl2)] <- "" 
tax_hp[is.na(tax_hp)] <- "" 
wardeh_hp[is.na(wardeh_hp)] <- "" 
wardeh_an[is.na(wardeh_an)] <- "" 
wardeh_pl[is.na(wardeh_pl)] <- "" 

atlas_ours_hp[is.na(atlas_ours_hp)] <- "" 
atlas_ours_an[is.na(atlas_ours_an)] <- "" 
atlas_ours_pl[is.na(atlas_ours_pl)] <- "" 
atlas_wardeh_hp[is.na(atlas_wardeh_hp)] <- "" 
atlas_wardeh_an[is.na(atlas_wardeh_an)] <- "" 
atlas_wardeh_pl[is.na(atlas_wardeh_pl)] <- "" 

wardeh_hp$Species <- gsub("\\[|\\]|", "", x = wardeh_hp$Species)
wardeh_pl$Species <- gsub("\\[|\\]|", "", x = wardeh_pl$Species)
wardeh_an$Species <- gsub("\\[|\\]|", "", x = wardeh_an$Species)
wardeh_hp <- filter_nonspecies(wardeh_hp)
wardeh_pl <- filter_nonspecies(wardeh_pl)
wardeh_an <- filter_nonspecies(wardeh_an)