run_both() {
    echo "$D;QUERY;CUDA (seg);SYCL (seg);CUDA (gflop);SYCL (gflop); CUDA PROM; SYCL PROM" >> $FILE
    for i in `seq 1 $ITERS`;
    do
        CUDA_SEG=`cat "$CURRENT_PATH/$query"_CUDA.txt | grep  "TIME" | egrep -o '[0-9.]+' | head -$i | tail -1`
        SYCL_SEG=`cat "$CURRENT_PATH/$query"_SYCL.txt | grep  "TIME" | egrep -o '[0-9.]+' | head -$i | tail -1`
        GFLOPS_CUDA="=((MULTIPLY($ID, $DB_LEN)) / MULTIPLY(VALUE(AVERAGE(INDIRECT(ADDRESS(ROW(),COLUMN() -2)))), 1000000000))"
        GFLOPS_SYCL="=((MULTIPLY($ID, $DB_LEN)) / MULTIPLY(VALUE(AVERAGE(INDIRECT(ADDRESS(ROW(),COLUMN() -2)))), 1000000000))"
        PROM="=VALUE(AVERAGE(INDIRECT(ADDRESS(ROW(),COLUMN() -2)): INDIRECT(ADDRESS(ROW() + $ITERS,COLUMN() -2))))"
        ROW=";$ID;$CUDA_SEG;$SYCL_SEG;$GFLOPS_CUDA;$GFLOPS_SYCL"
        if [ "$i" == 1 ]; then
            ROW="$ROW; $PROM; $PROM"
        fi
        echo $ROW >> $FILE
    done
    echo "" >> $FILE
}

run_single() {
    echo "$D;QUERY;SYCL (seg);SYCL (gflop); SYCL PROM" >> $FILE
    for i in `seq 1 $ITERS`;
    do
        SYCL_SEG=`cat "$CURRENT_PATH/$query"_SYCL.txt | grep  "TIME" | egrep -o '[0-9.]+' | head -$i | tail -1`
        GFLOPS_SYCL="=((MULTIPLY($ID, $DB_LEN)) / MULTIPLY($SYCL_SEG, 1000000000))"
        PROM="=VALUE(AVERAGE(INDIRECT(ADDRESS(ROW(),COLUMN() -1)): INDIRECT(ADDRESS(ROW() + $ITERS,COLUMN() -1))))"
        ROW=";$ID;$SYCL_SEG;$GFLOPS_SYCL"
        if [ "$i" == 1 ]; then
            ROW="$ROW; $PROM"
        fi
        echo $ROW >> $FILE
    done
    echo "" >> $FILE
}

G1() {
    ITERS=10
    DB_LEN=251837611
    query=query_sequences_20.fasta
    ID=44068
    FILE=g1.txt
    
    if [ -f $FILE ] ; then
        rm $FILE
    fi
    
    for D in ORIGINAL 64 128 256 512 1024 dynamic
    do
        CURRENT_PATH=G1/sprot/"$D"/SW
        run_both
    done
}


G2() {
    ITERS=10
    DB_LEN=995210546
    query=query_sequences_20.fasta
    ID=44068
    FILE=g2.txt
    
    if [ -f $FILE ] ; then
        rm $FILE
    fi
    
    CURRENT_PATH=G2/nr/SW
    D=NR
    run_both
}

G3() {
    ITERS=10
    DB_LEN=251837611
    FILE=g3.txt
    
    if [ -f $FILE ] ; then
        rm $FILE
    fi
    
    for j in `seq 1 20`
    do
        query=`head -n $j ../../databases/protein/queries/queries.txt | tail -n 1`
        D=`echo $query | egrep -o '[0-9]+'`
        ID=$D
        CURRENT_PATH=G3/sprot/SW
        run_both
    done
}


G4() {
    ITERS=10
    DB_LEN=251837611
    query=query_sequences_20.fasta
    ID=44068
    FILE=g4.txt
    
    if [ -f $FILE ] ; then
        rm $FILE
    fi
    
    for D in SW NW HW OV
    do
        CURRENT_PATH=G4/sprot/$D
        run_both
    done
}


G5() {
    ITERS=10
    DB_LEN=251837611
    query=query_sequences_20.fasta
    ID=44068
    FILE=g5.txt
    
    if [ -f $FILE ] ; then
        rm $FILE
    fi
    
    for D in "BLOSUM_45-10-3" "BLOSUM_45-14-2" "BLOSUM_45-19-1" "BLOSUM_62-06-2" "BLOSUM_62-10-2" "BLOSUM_62-13-1" "BLOSUM_90-06-2" "BLOSUM_90-09-1" "BLOSUM_90-11-1"
    do
        CURRENT_PATH=G5/sprot/$D
        run_both
    done
}

G6() {
    ITERS=10
    FILE=g6.txt
    CURRENT_PATH=G6/SW
    
    if [ -f $FILE ] ; then
        rm $FILE
    fi
    
    for j in `seq 1 9`;
    do
        QUERY=`head -n $j ../../databases/adn/queries.txt | tail -n 1 | sed s/".fasta"//`
        TARGET=`head -n $j ../../databases/adn/targets.txt | tail -n 1 | sed s/".fasta"//`
        QUERY_LEN=`cat "$CURRENT_PATH"/"$j"_cuda.txt | grep "Query length:" | egrep -o '[0-9.]+' | head -1`
        TARGET_LEN=`cat "$CURRENT_PATH"/"$j"_cuda.txt | grep "Query length:" | egrep -o '[0-9.]+' | head -1`
        echo "QUERY;TARGET;CUDA (seg);SYCL (seg);CUDA (gflop);SYCL (gflop); CUDA PROM; SYCL PROM" >> $FILE
        for i in `seq 1 $ITERS`;
        do
            CUDA_SEG=`cat "$CURRENT_PATH"/"$j"_CUDA.txt | grep  "TIME" | egrep -o '[0-9.]+' |  head -$i | tail -1`
            SYCL_SEG=`cat "$CURRENT_PATH"/"$j"_SYCL.txt | grep  "TIME" | egrep -o '[0-9.]+' | head -$i | tail -1`
            GFLOPS_CUDA=`perl -e "print (($QUERY_LEN * $TARGET_LEN ) / ($CUDA_SEG * 1000000000))"`
            GFLOPS_SYCL=`perl -e "print (($QUERY_LEN * $TARGET_LEN ) / ($SYCL_SEG * 1000000000))"`
            PROM="=VALUE(AVERAGE(INDIRECT(ADDRESS(ROW(),COLUMN() -2)): INDIRECT(ADDRESS(ROW() + $ITERS,COLUMN() -2))))"
            ROW="$QUERY;$TARGET;$CUDA_SEG;$SYCL_SEG;$GFLOPS_CUDA;$GFLOPS_SYCL"
            if [ "$i" == 1 ]; then
                ROW="$ROW; $PROM; $PROM"
            fi
            echo $ROW >> $FILE
        done
        
        echo "" >> $FILE
    done
    
}

G9() {
    ITERS=10
    DB_LEN=251837611
    FILE=g9.txt
    CURRENT_PATH=G9/sprot/SW
    
    if [ -f $FILE ] ; then
        rm $FILE
    fi
    for j in `seq 1 20` ;
    do
        query=`head -n $j ../../databases/protein/queries/queries.txt | tail -n 1`
        ID=`echo $query | egrep -o '[0-9]+'`
        echo "QUERY;SYCL (seg);SYCL (gflop); SYCL PROM" >> $FILE
        for i in `seq 1 $ITERS`;
        do
            SYCL_SEG=`cat "$CURRENT_PATH/$query"_sycl.txt | grep  "TIME" | egrep -o '[0-9.]+' | head -$i | tail -1`
            GFLOPS_SYCL=`perl -e "print ((('$ID' + 0) * $DB_LEN) / ($SYCL_SEG * 1000000000))"`
            PROM="=VALUE(AVERAGE(INDIRECT(ADDRESS(ROW(),COLUMN() -1)): INDIRECT(ADDRESS(ROW() + $ITERS,COLUMN() -1))))"
            ROW="$ID;$SYCL_SEG;$GFLOPS_SYCL"
            if [ "$i" == 1 ]; then
                ROW="$ROW; $PROM"
            fi
            echo $ROW >> $FILE
        done
        echo "" >> $FILE
    done
}

G99() {
    ITERS=10
    DB_LEN=251837611
    query=query_sequences_20.fasta
    ID=44068
    FILE=G9.txt
    
    if [ -f $FILE ] ; then
        rm $FILE
    fi
    
    CURRENT_PATH=G9/sprot/SW
    D=SPROT
    run_single
}



if [ -n "$2" ]; then
    G1 #different work groups (20)
    G2 #ENV_NR database (20)
    G3 #SWIS_PROT (individidual)
    G4 #SWIS_PROT different algorithms (20)
    G5 #SWIS_PROT different matrices (20)
    G6 #DNA small and medium
else
    G99 #CPU SWIS_PROT (individual)
fi
