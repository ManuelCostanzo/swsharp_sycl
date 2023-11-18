#!/bin/bash

declare -A configs
configs["acpp_generic_rtx3090"]="acpp_generic:0:acpp_generic_rtx3090"
configs["acpp_sm_rtx3090"]="acpp_rtx3090:0:acpp_sm_rtx3090"
configs["intel_rtx3090"]="intel_nvidia:3:intel_rtx3090"

execute_command() {
    local bin=$1
    local card=$2
    local file=$3

    for i in {1..5}; do
        echo "Ejecución número $i para $file"
        ../SYCL/bin/swsharpdb_$bin -i ../databases/protein/queries/query_sequences_20 -j ../databases/protein/uniprot_sprot.fasta -g 10 -e 2 -T 0  --nocache --max-aligns=10 --cards=$card >> CMMSE/$file.txt
    done
}


mkdir -p CMMSE
for key in "${!configs[@]}"; do
    echo "Ejecutando para la configuración: $key"

    IFS=':' read -r -a params <<< "${configs[$key]}"
    bin=${params[0]}
    card=${params[1]}
    file=${params[2]}

    execute_command $bin $card $file
done