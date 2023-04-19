#!/bin/bash


declare -A scriptlist

scriptlist=( ["rtx2070"]="2 0" ["i57400"]="1" ["rtx2070-i57400"]="rtx2070-i57400 21" )

for folder in "${!scriptlist[@]}"
do
    echo "Executing $folder with args ${scriptlist[$script]}"
    
    if [ ! -d "$folder" ]; then
        mkdir $folder
    fi
    
    cd $folder
    
    sh ../run.sh ${scriptlist[$script]}
    
    cd ..
done