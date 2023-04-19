#!/bin/bash

declare -A scriptlist

scriptlist=( ["gtx1080"]="2 0" ["gtx980"]="4 2" ["E52695"]="1" ["gtx1080-gtx1080"]="23" ["gtx1080-E52695"]="21" )


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