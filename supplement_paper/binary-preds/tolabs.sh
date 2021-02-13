while read l; do if grep -qF "${l}" trainvhdb_pos_ids.txt; then echo 1 >> bowtie_pos_1_labs.txt; elif grep -qF "${l}" trainvhdb_neg_ids.txt; then echo 0 >> bowtie_pos_1_labs.txt; else echo "na" >> bowtie_pos_1_labs.txt; fi; done < bowtie_pos_1_ids.txt
while read l; do if grep -qF "${l}" trainvhdb_pos_ids.txt; then echo 1 >> bowtie_pos_2_labs.txt; elif grep -qF "${l}" trainvhdb_neg_ids.txt; then echo 0 >> bowtie_pos_2_labs.txt; else echo "na" >> bowtie_pos_2_labs.txt; fi; done < bowtie_pos_2_ids.txt
while read l; do if grep -qF "${l}" trainvhdb_neg_ids.txt; then echo 0 >> bowtie_neg_1_labs.txt; elif grep -qF "${l}" trainvhdb_pos_ids.txt; then echo 1 >> bowtie_neg_1_labs.txt; else echo "na" >> bowtie_neg_1_labs.txt; fi; done < bowtie_neg_1_ids.txt
while read l; do if grep -qF "${l}" trainvhdb_neg_ids.txt; then echo 0 >> bowtie_neg_2_labs.txt; elif grep -qF "${l}" trainvhdb_pos_ids.txt; then echo 1 >> bowtie_neg_2_labs.txt; else echo "na" >> bowtie_neg_2_labs.txt; fi; done < bowtie_neg_2_ids.txt


