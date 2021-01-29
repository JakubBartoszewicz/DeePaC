bioawk -c fastx -v min=25 -v max=250 -v seed=0 'BEGIN{srand(seed);} {print ">"$name; print substr($seq,0,int(min+rand()*(max-min+1)))}' pathogenic_train.fasta > pathogenic_rn_train.fasta

bioawk -c fastx 'BEGIN{fname="";} {lname=substr($name,60,15); if (fname!=lname) {fname=lname; outname=fname".fa"; print outname;}; print ">"$name >> outname; print $seq >> outname}' pathogenic_rn_train.fasta

bioawk -c fastx -v min=25 -v max=250 -v seed=1 'BEGIN{srand(seed);} {print ">"$name; print substr($seq,0,int(min+rand()*(max-min+1)))}' pathogenic_train.fasta > pathogenic_rn_train.fasta

bioawk -c fastx 'BEGIN{fname="";D} {lname=substr($name,63,15); if (fname!=lname) {fname=lname; outname=fname".fa"; print outname;}; print ">"$name >> outname; print $seq >> outname}' ../nonpathogenic_rn_train.fasta

bioawk -c fastx -v min=25 -v max=250 -v seed=3 'BEGIN{srand(seed);} {print ">"$name; print substr($seq,0,int(min+rand()*(max-min+1)))}' nonpathogenic_val.fasta > nonpathogenic_rn_val.fasta

bioawk -c fastx -v min=25 -v max=250 -v seed=4 'BEGIN{srand(seed);} {print ">"$name; print substr($seq,0,int(min+rand()*(max-min+1)))}' pathogenic_val.fasta > pathogenic_rn_val.fasta

bioawk -c fastx -v min=25 -v max=250 -v seed=5 'BEGIN{srand(seed);} {print ">"$name; print substr($seq,0,int(min+rand()*(max-min+1)))}' nonpathogenic_train_all.fasta > nonpathogenic_rn_train_all.fasta

bioawk -c fastx -v min=25 -v max=250 -v seed=6 'BEGIN{srand(seed);} {print ">"$name; print substr($seq,0,int(min+rand()*(max-min+1)))}' pathogenic_train_hum.fasta > pathogenic_rn_train_hum.fasta

bioawk -c fastx -v min=25 -v max=250 -v seed=7 'BEGIN{srand(seed);} {print ">"$name; print substr($seq,0,int(min+rand()*(max-min+1)))}' nonpathogenic_val_all.fasta > nonpathogenic_rn_val.fasta

bioawk -c fastx -v min=25 -v max=250 -v seed=8 'BEGIN{srand(seed);} {print ">"$name; print substr($seq,0,int(min+rand()*(max-min+1)))}' pathogenic_val_hum.fasta > pathogenic_rn_val_hum.fasta

