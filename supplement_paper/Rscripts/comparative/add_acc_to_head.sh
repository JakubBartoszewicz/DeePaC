for FILE in *fna; do arrFILE=(${FILE//_/ }); NAME=${arrFILE[0]}_${arrFILE[1]}; sed "s/^>/>${NAME}|/g" ${FILE} > out/${FILE}; done
