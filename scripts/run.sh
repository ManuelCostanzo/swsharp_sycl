#!/bin/bash

DATABASE_FOLDER=/app/swsharp_oneapi/databases/
SWSYCL_FOLDER=../../

default_values() {
    ITERS=1
    SYCL_CARD_ID=1
    CUDA_CARD_ID=0
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
    if [ "$1" = "CUDA" ]; then
        CARD_ID=$CUDA_CARD_ID
    else
        CARD_ID=$SYCL_CARD_ID
    fi
    
    sleep 5
    ./$SWSYCL_FOLDER/$1/bin/swsharpdb -i $DATABASE_FOLDER/protein/queries/$QUERY -j $DATABASE_FOLDER/protein/$DB -g $GAP -e $EXTEND -T $THREADS --matrix=$MATRIX --algorithm=$ALGORITHM --cards=$CARD_ID --max-aligns=$MAX_ALIGNS --nocache >> $FOLDER/"$QUERY"_$1.txt
}

exec_dna() {
    if [ "$1" = "CUDA" ]; then
        CARD_ID=$CUDA_CARD_ID
    else
        CARD_ID=$SYCL_CARD_ID
    fi
    
    sleep 5
    ./$SWSYCL_FOLDER/$1/bin/swsharpn -i $DATABASE_FOLDER/adn/queries/$QUERY -j $DATABASE_FOLDER/adn/queries/$TARGET -g $GAP -e $EXTEND --cards=$CARD_ID --algorithm=$ALGORITHM   --score >> $FOLDER/"$FILE"_$1.txt
}

GMULTIPLE() {
    default_values
    DB=uniprot_sprot.fasta
    FOLDER=G2/sprot/SW
    mkdir -p $FOLDER
    ALGORITHM=SW
    for i in `seq 1 $ITERS`;
    do
        QUERY=query_sequences_20.fasta
        debug
        # exec_prot CUDA
        exec_prot SYCL
    done
}



G1() {
    default_values
    mkdir -p G1
    DB=uniprot_sprot.fasta
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
    DB=env_nr.fasta
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
    DB=uniprot_sprot.fasta
    mkdir -p $FOLDER
    
    for i in `seq 1 $ITERS`;
    do
        for j in `seq 1 20`
        do
            QUERY=`head -n $j $DATABASE_FOLDER/protein/queries/queries.txt | tail -n 1`
            debug
            exec_prot CUDA
            exec_prot SYCL
        done
    done
}


G4() {
    default_values
    FOLDER=G4/sprot
    DB=uniprot_sprot.fasta
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
    DB=uniprot_sprot.fasta
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
                QUERY=`head -n $j $DATABASE_FOLDER/adn/queries.txt | tail -n 1`
                TARGET=`head -n $j $DATABASE_FOLDER/adn/targets.txt | tail -n 1`
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
    CARD_ID=1
    FOLDER=G9/sprot/SW
    DB=uniprot_sprot.fasta
    mkdir -p $FOLDER
    
    for i in `seq 1 $ITERS`;
    do
        for j in `seq 1 20`
        do
            QUERY=`head -n $j $DATABASE_FOLDER/protein/queries/queries.txt | tail -n 1`
            debug
            exec_prot SYCL
        done
    done
}


#MAIN
G1 #different work groups (20)
G2 #ENV_NR database (20)
G3 #SWIS_PROT (individidual)
G4 #SWIS_PROT different algorithms (20)
G5 #SWIS_PROT different matrices (20)
G6 #DNA small and medium
G9 #CPU SWIS_PROT (individual)
# GMULTIPLE
