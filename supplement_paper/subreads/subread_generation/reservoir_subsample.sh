bioawk -c fastx -v k=10000000 -v seed=100 'BEGIN{srand(seed);} {y=x++<k?x-1:int(rand()*x);if(y<k)a[y]=">"$name"\n"$seq}END{for(z in a)print a[z]}' nonpathogenic_rn_train_bacvir.fasta > nonpathogenic_rn_train_bacvir_half.fasta
bioawk -c fastx -v k=1250000 -v seed=101 'BEGIN{srand(seed);} {y=x++<k?x-1:int(rand()*x);if(y<k)a[y]=">"$name"\n"$seq}END{for(z in a)print a[z]}' nonpathogenic_rn_val_bacvir.fasta > nonpathogenic_rn_val_bacvir_half.fasta
bioawk -c fastx -v k=625000 -v seed=102 'BEGIN{srand(seed);} {y=x++<k?x-1:int(rand()*x);if(y<k)a[y]=">"$name"\n"$seq}END{for(z in a)print a[z]}' nonpathogenic_test_bacvir_raw_1.fasta > nonpathogenic_test_bacvir_half_1.fasta
bioawk -c fastx -v k=625000 -v seed=102 'BEGIN{srand(seed);} {y=x++<k?x-1:int(rand()*x);if(y<k)a[y]=">"$name"\n"$seq}END{for(z in a)print a[z]}' nonpathogenic_test_bacvir_raw_2.fasta > nonpathogenic_test_bacvir_half_2.fasta
awk -F ">" '/^>/ {close(F) ; F = "out_1/"gensub("fq.*", "fa", "g", gensub(".*nonpathogenic/", "", "g", $2))} {print >> F}' nonpathogenic_test_bacvir_half_1.fasta
awk -F ">" '/^>/ {close(F) ; F = "out_2/"gensub("fq.*", "fa", "g", gensub(".*nonpathogenic/", "", "g", $2))} {print >> F}' nonpathogenic_test_bacvir_half_2.fasta

