# path to predictions to test
pred_folders="$(ls -d /home/uli/Dropbox/HPI_RKI/DeePaC/model_evaluation_subreads/DeePaC/predictions/* | grep 'img' | grep -v 'nano' )"

# test all possible comb (no repetions)
for pred_1 in $pred_folders
do
    for pred_2 in $pred_folders
    do 
        # check if comb was already tested
        filename_1=$(basename $pred_1)"_"$(basename $pred_2)
        filename_2=$(basename $pred_2)"_"$(basename $pred_1)
        if [ $pred_1 == $pred_2 ] || [ -f ./"$filename_1"-metrics.csv ] || [ -f ./"$filename_2"-metrics.csv ]
        then
            echo $filename_1
        else
            ./runEval_ens.sh $pred_1 $pred_2 test_1
        fi
    done
done
