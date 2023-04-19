#!/bin/bash


declare -A scriptlist

scriptlist=( ["rtx2070"]="2 0" ["i57400"]="1" ["rtx2070-i57400"]="rtx2070-i57400 21" )

for folder in "${!scriptlist[@]}"
do
    echo "Executing $folder with args ${scriptlist[$folder]}"
    
    if [ ! -d "$folder" ]; then
        mkdir $folder
    fi
    
    cd $folder
    
    ./../run.sh ${scriptlist[$folder]}
    
    cd ..
done