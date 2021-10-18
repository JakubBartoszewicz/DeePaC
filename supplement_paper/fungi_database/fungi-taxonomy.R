#Load Taxonomy
n <- read.table("FUNGI_DATA_CUR/src/taxdump_211009/names.dmp", sep = "\t", fill = TRUE, quote = NULL, comment.char = "")[,c(1,3,7)]
colnames(n) <- c("taxid", "name", "name_type")
n$name <- as.character(n$name)
n$name_type <- relevel(n$name_type, "scientific name")
n <- n[order(n$taxid, n$name_type),]
n$name_type <- as.character(n$name_type)
n_extras <- n
n <- n[!duplicated(n$taxid),]
t <- read.table("FUNGI_DATA_CUR/src/taxdump_211009/nodes.dmp", sep = "\t", fill = TRUE, stringsAsFactors=F, quote = NULL, comment.char = "")[,c(1,3,5)]
colnames(t) <- c("taxid", "parent_id", "rank")
m <- read.table("FUNGI_DATA_CUR/src/taxdump_211009/merged.dmp", sep = "\t", fill = TRUE, stringsAsFactors=F, quote = NULL, comment.char = "")[,c(1,3)]
colnames(m) <- c("old_id", "new_id")
# Correct typos
typos <- read.csv("FUNGI_DATA_CUR/src/known-typos.csv", stringsAsFactors = F)

# Genbank

genbank <- read.csv("FUNGI_DATA_CUR/src/genbank_211009/assembly_summary_ed_211009.txt", sep="\t")
genbank$seq_rel_date <- as.Date(genbank$seq_rel_date)
genbank_cl <- genbank[genbank$refseq_category!="na",]
genbank_cl <- genbank_cl[,c("assembly_accession", "refseq_category", "taxid", "species_taxid", "organism_name", "infraspecific_name", "assembly_level", "seq_rel_date", "asm_name", "ftp_path")]
# don't filter organism names in genbank - we filter species names in the 'resolve' functions