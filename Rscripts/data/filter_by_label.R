Labels <- readRDS("FilterLabels.rds")
ReadFiles <- list.files("./", pattern = "fasta$", full.names = F)

pathogenic <- ReadFiles[Labels[tools::file_path_sans_ext(ReadFiles)]]
nonpathogenic <- ReadFiles[!Labels[tools::file_path_sans_ext(ReadFiles)]]

patho.fails <- pathogenic[!file.copy(pathogenic, file.path("pathogenic", pathogenic))]
nonpatho.fails <- nonpathogenic[!file.copy(nonpathogenic, file.path("nonpathogenic", nonpathogenic))]
print("Pathogenic copying failed for: ")
patho.fails
print("NonPathogenic copying failed for: ")
nonpatho.fails