#!/bin/bash

folder="CMMSE"
output_file="times.csv"

files=($folder/*.txt)

declare -A file_times

max_count=0

for file in "${files[@]}"; do
    filename=$(basename "$file" .txt)
    
    i=0
    while read -r line; do
        if [[ "$line" =~ TIME:\ ([0-9]*\.[0-9]*) ]]; then
            file_times[$filename,$i]=${BASH_REMATCH[1]}
            ((i++))
        fi
    done < "$file"
    
    if ((i > max_count)); then
        max_count=$i
    fi
done

{
    for file in "${files[@]}"; do
        filename=$(basename "$file" .txt)
        printf "%s," "$filename"
    done
    printf "\n"
    
    for ((i=0; i<max_count; i++)); do
        for file in "${files[@]}"; do
            filename=$(basename "$file" .txt)
            printf "%s," "${file_times[$filename,$i]}"
        done
        printf "\n"
    done
} > "$output_file"

