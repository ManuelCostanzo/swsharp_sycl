#!/bin/bash

default_values() {
    ITERS=5
    CARD_ID=0
    GAP=10
    EXTEND=2
    THREADS=1
    MATRIX=BLOSUM_62
    ALGORITHM=SW
    MAX_ALIGNS=10
    
    unset MAX_THREADS #for dynamic configuration
}

debug() {
    echo "$FOLDER $DB => $QUERY"
}


exec_prot() {
    sleep 5
    ./../../$1/bin/swsharpdb -i ../../databases/protein/queries/$QUERY -j $DB -g $GAP -e $EXTEND -T $THREADS --matrix=$MATRIX --algorithm=$ALGORITHM --cards=$CARD_ID --max-aligns=$MAX_ALIGNS --nocache >> $FOLDER/"$QUERY"_cuda.txt
}

exec_dna() {
    sleep 5
    ./../../$1/bin/swsharpn -i ../../databases/adn/queries/$QUERY -j ../../databases/adn/queries/$TARGET -g $GAP -e $EXTEND --cards=$CARD_ID --algorithm=$ALGORITHM   --score >> $FOLDER/"$FILE"_cuda.txt
}


G1() {
    default_values
    mkdir -p G1
    DB=../../databases/protein/uniprot_sprot.fasta
    for MAX_THREADS in ORIGINAL 64 128 256 512 1024 dynamic
    do
        FOLDER=G1/sprot/"$MAX_THREADS"
        mkdir -p $FOLDER
        
        
        for ALGORITHM in SW
        do
            FOLDER=G1/sprot/"$MAX_THREADS"/"$ALGORITHM"
            mkdir -p $FOLDER
            export MAX_THREADS=$MAX_THREADS
            if [ "$MAX_THREADS" == "dynamic" ]; then
                unset MAX_THREADS
            fi
            
            for i in `seq 1 $ITERS`;
            do
                QUERY=query_sequences_20.fasta
                debug
                exec_prot CUDA
                exec_prot SYCL
            done
        done
    done
}


G2() {
    default_values
    DB=../../databases/protein/env_nr.fasta
    FOLDER=G2/nr/SW
    mkdir -p $FOLDER
    ALGORITHM=SW
    for i in `seq 1 $ITERS`;
    do
        QUERY=query_sequences_20.fasta
        debug
        exec_prot CUDA
        exec_prot SYCL
    done
}

G3() {
    default_values
    FOLDER=G3/sprot/SW
    DB=../../databases/protein/uniprot_sprot.fasta
    mkdir -p $FOLDER
    
    for i in `seq 1 $ITERS`;
    do
        for j in `seq 1 20`
        do
            QUERY=`head -n $j ../../databases/protein/queries/queries.txt | tail -n 1`
            debug
            exec_prot CUDA
            exec_prot SYCL
        done
    done
}


G4() {
    default_values
    FOLDER=G4/sprot
    DB=../../databases/protein/uniprot_sprot.fasta
    mkdir -p $FOLDER
    for ALGORITHM in SW HW NW OV
    do
        FOLDER=G4/sprot/"$ALGORITHM"
        mkdir -p $FOLDER
        for i in `seq 1 $ITERS`;
        do
            QUERY=query_sequences_20.fasta
            debug
            exec_prot CUDA
            exec_prot SYCL
        done
    done
}


G5() {
    default_values
    FOLDER=G5/sprot
    DB=../../databases/protein/uniprot_sprot.fasta
    mkdir -p $FOLDER
    for M in "BLOSUM_45-10-3" "BLOSUM_45-14-2" "BLOSUM_45-19-1" "BLOSUM_62-06-2" "BLOSUM_62-10-2" "BLOSUM_62-13-1" "BLOSUM_90-06-2" "BLOSUM_90-09-1" "BLOSUM_90-11-1"
    do
        FOLDER=G5/sprot/"$M"
        mkdir -p $FOLDER
        MATRIX=${M:0:9}
        GAP=$((10#${M:10:2}))
        EXTEND=${M:13:1}
        
        for i in `seq 1 $ITERS`;
        do
            QUERY=query_sequences_20.fasta
            debug
            exec_prot CUDA
            exec_prot SYCL
        done
    done
}


G6() {
    default_values
    GAP=5
    EXTEND=2
    FOLDER=G6
    mkdir -p $FOLDER
    for ALGORITHM in SW
    do
        FOLDER="$FOLDER"/$ALGORITHM
        mkdir -p $FOLDER
        for i in `seq 1 $ITERS`;
        do
            for j in `seq 1 9`;
            do
                QUERY=`head -n $j ../../databases/adn/queries.txt | tail -n 1`
                TARGET=`head -n $j ../../databases/adn/targets.txt | tail -n 1`
                FILE=$j
                echo "$FOLDER $QUERY | $TARGET"
                exec_dna CUDA
                exec_dna SYCL
            done
        done
    done
}


G9() {
    default_values
    FOLDER=G9/sprot/SW
    DB=../../databases/protein/uniprot_sprot.fasta
    mkdir -p $FOLDER
    
    for i in `seq 1 $ITERS`;
    do
        for j in `seq 1 20`
        do
            QUERY=`head -n $j ../../databases/protein/queries/queries.txt | tail -n 1`
            debug
            exec_prot SYCL
        done
    done
}


#MAIN
G1
G2
G3
G4
G5
G6
G9
