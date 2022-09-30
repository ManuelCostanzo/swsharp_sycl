/*
swsharp - CUDA parallelized Smith Waterman with applying Hirschberg's and 
Ukkonen's algorithm and dynamic cell pruning.
Copyright (C) 2013 Matija Korpar, contributor Mile Šikić

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Contact the author by mkorpar@gmail.com.
*/
/**
@file

@brief GPU implementations of common functions.
*/

#ifndef __SW_SHARP_GPU_MODULEH__
#define __SW_SHARP_GPU_MODULEH__

#include "alignment.h"
#include "chain.h"
#include "scorer.h"
#include "thread.h"

#ifdef __cplusplus 
extern "C" {
#endif

//******************************************************************************
// SINGLE ALIGNMENT

/*!
@brief GPU implementation of the semiglobal scoring function.

Function provides the semiglobal end data, the position of the maximum score
on the query and target sequences as well as the maximum score. QueryEnd must
be equal to the length of the query minus one.

@param queryEnd output, position of the maximum score on the query sequences
@param targetEnd output, position of the maximum score on the target sequences
@param outScore output, maximum score
@param query query chain
@param target target chain
@param scorer scorer object used for alignment
@param score input alignment score if known, otherwise #NO_SCORE 
@param card CUDA card on which the function will be executed
@param thread thread on which the function will be executed, if NULL function is
    executed on the current thread
*/
extern void hwEndDataGpu(int* queryEnd, int* targetEnd, int* outScore, 
    Chain* query, Chain* target, Scorer* scorer, int score, int card, 
    Thread* thread);
    
/*!
@brief GPU implementation of score finding function.

Method uses Needleman-Wunsch algorithm with all of the start conditions set to
infinity. This assures path contains the first cell and does not start with gaps.
If the score is found it returns the coordinates of the cell with the provided 
score, (-1, -1) otherwise.

@param queryStart output, if found query index of found cell, -1 otherwise
@param targetStart output, if found target index of found cell, -1 otherwise
@param query query chain
@param queryFrontGap indicates that query starts with a gap
@param target target chain
@param scorer scorer object used for alignment
@param score input alignment score
@param card CUDA card on which the function will be executed
@param thread thread on which the function will be executed, if NULL function is
    executed on the current thread
*/                 
extern void nwFindScoreGpu(int* queryStart, int* targetStart, Chain* query, 
    int queryFrontGap, Chain* target, Scorer* scorer, int score, int card, 
    Thread* thread);

/*!
@brief GPU implementation of Needleman-Wunsch scoring function.

If scores and/or affines pointers are not equal to NULL method provides the last
row of the scoring matrix and the affine deletion matrix, respectively. Method
uses Ukkonen's banded optimization with the pLeft and pRight margins. PLeft 
margin is defined as diagonal left of the diagonal from the (0, 0) cell, and 
pRight as the right diagonal, respectively. Only the cells lying between those 
diagonals are calculated.

@param scores output, if not NULL the last row of the scoring matrix, 
    new array is created
@param affines output, if not NULL the last row of the affine deletion matrix, 
    new array is created
@param query query chain
@param queryFrontGap if not 0, force that alignments start with a query gap
@param target target chain
@param targetFrontGap if not 0, force that alignments start with a target gap
@param scorer scorer object used for alignment
@param pLeft left Ukkonen's margin
@param pRight right Ukkonen's margin
@param card CUDA card on which the function will be executed
@param thread thread on which the function will be executed, if NULL function is
    executed on the current thread
*/
extern void nwLinearDataGpu(int** scores, int** affines, Chain* query, 
    int queryFrontGap, Chain* target, int targetFrontGap, Scorer* scorer, 
    int pLeft, int pRight, int card, Thread* thread);

/*!
@brief GPU implementation of the overlap scoring function.

Function provides the overlap end data, the position of the maximum score
on the query and target sequences as well as the maximum score. QueryEnd must
be equal to the length of the query minus one or targetEnd must be equal to the
length of the target minus one.

@param queryEnd output, position of the maximum score on the query sequences
@param targetEnd output, position of the maximum score on the target sequences
@param outScore output, maximum score
@param query query chain
@param target target chain
@param scorer scorer object used for alignment
@param score input alignment score if known, otherwise #NO_SCORE 
@param card CUDA card on which the function will be executed
@param thread thread on which the function will be executed, if NULL function is
    executed on the current thread
*/
extern void ovEndDataGpu(int* queryEnd, int* targetEnd, int* outScore, 
    Chain* query, Chain* target, Scorer* scorer, int score, int card, 
    Thread* thread);

/*!
@brief GPU implementation of score finding function.

Method uses Needleman-Wunsch algorithm. If the score is found and the indicies 
of the coresponding cell are on the border of the solving matrix, functions 
returns the coordinates of the cell with the provided score, (-1, -1) otherwise.

@param queryStart output, if found query index of found cell, -1 otherwise
@param targetStart output, if found target index of found cell, -1 otherwise
@param query query chain
@param target target chain
@param scorer scorer object used for alignment
@param score input alignment score
@param card CUDA card on which the function will be executed
@param thread thread on which the function will be executed, if NULL function is
    executed on the current thread
*/                 
extern void ovFindScoreGpu(int* queryStart, int* targetStart, Chain* query, 
    Chain* target, Scorer* scorer, int score, int card, Thread* thread);

/*!
@brief GPU implementation of Smith-Waterman scoring function.

Function provides the Smith-Waterman end data, the position of the maximum score
on the query and target sequences as well as the maximum score. Additionally if
scores and/or affines pointers are not equal to NULL method provides the last
row of the scoring matrix and the affine deletion matrix, respectively.

@param queryEnd output, position of the maximum score on the query sequences
@param targetEnd output, position of the maximum score on the target sequences
@param scores output, if not NULL the last row of the scoring matrix,
    new array is created
@param affines output, if not NULL the last row of the affine deletion matrix,
    new array is created
@param outScore output, maximum score
@param query query chain
@param target target chain
@param scorer scorer object used for alignment
@param score input alignment score if known, otherwise #NO_SCORE 
@param card CUDA card on which the function will be executed
@param thread thread on which the function will be executed, if NULL function is
    executed on the current thread
*/
extern void swEndDataGpu(int* queryEnd, int* targetEnd, int* outScore, 
    int** scores, int** affines, Chain* query, Chain* target, Scorer* scorer, 
    int score, int card, Thread* thread);

//******************************************************************************

//******************************************************************************
// DATABASE ALIGNMENT

/*!
@brief GPU database scoring object.

In the database aligning, queries are often changed and the database is fairly
static. ChainDatabaseGpu is the chain database prepared for GPU usage to reduce
the preperation time in repetitive aligning.
*/
typedef struct ChainDatabaseGpu ChainDatabaseGpu;

/*!
@brief ChainDatabaseGpu constructor.

@param database chain array
@param databaseLen chain array length
@param cards cuda cards index array which the database will be available on
@param cardsLen cuda cards index array length, greater or equal to 1

@return chainDatabaseGpu object
*/
extern ChainDatabaseGpu* chainDatabaseGpuCreate(Chain** database, int databaseLen, 
    int* cards, int cardsLen);

/*!
@brief ChainDatabaseGpu destructor.

@param chainDatabaseGpu chainDatabaseGpu object
*/
extern void chainDatabaseGpuDelete(ChainDatabaseGpu* chainDatabaseGpu);

/*!
@brief ChainDatabaseGpu memory consumption getter

@param database chain array
@param databaseLen chain array length

@return memory needed for the database to be stored on the gpu
*/
extern size_t chainDatabaseGpuMemoryConsumption(Chain** database, int databaseLen);

/*!
@brief GPU database aligning function.

Function scores the query with every target in the chainDatabaseGpu, in other 
words with every chain in the database array with witch the chainDatabaseGpu
was created. The new score array has legnth of databaseLen, where databaseLen is
the argument with which the chainDatabaseGpu was created. If the indexes array
is given only the targets with given indexes will be scored, other targets will 
have the #NO_SCORE score. CUDA cards are necessary for this function to work.

@param scores output, array of scores coresponding to every chain in the 
    database array with which the chainDatabaseGpu was created, new array is 
    created
@param type aligning type, can be #SW_ALIGN, #NW_ALIGN, #HW_ALIGN or #OV_ALIGN
@param query query chain
@param chainDatabaseGpu gpu chain database object
@param scorer scorer object used for alignment
@param indexes array of indexes of which chains from the database to score, 
    if NULL all are solved
@param indexesLen indexes array length
@param cards cuda cards index array
@param cardsLen cuda cards index array length, greater or equal to 1
@param thread thread on which the function will be executed, if NULL function is
    executed on the current thread
*/
extern void scoreDatabaseGpu(int** scores, int type, Chain* query, 
    ChainDatabaseGpu* chainDatabaseGpu, Scorer* scorer, int* indexes, 
    int indexesLen, int* cards, int cardsLen, Thread* thread);

/*!
@brief GPU shotgun database aligning function.

Function scores every query with every target in the chainDatabaseGpu, in other 
words with every chain in the database array with witch the chainDatabaseGpu
was created. The new score array has legnth of databaseLen * queriesLen, where 
databaseLen is the argument with which the chainDatabaseGpu was created. Array
is organized as a table of databaseLen columns and queriesLen rows, where rows
correspond to queries and columns to targets in the database. If the indexes 
array is given only the targets with given indexes will be scored, other 
targets will have the #NO_SCORE score. CUDA cards are necessary for this 
function to work. This function is faster than calling scoreDatabaseGpu() for
every query separately.

@param scores output, array of scores coresponding to every query scored with 
    every chain in the database array with which the chainDatabaseGpu was 
    created, new array is created
@param type aligning type, can be #SW_ALIGN, #NW_ALIGN, #HW_ALIGN or #OV_ALIGN
@param queries query chains array
@param queriesLen query chains array length
@param chainDatabaseGpu gpu chain database object
@param scorer scorer object used for alignment
@param indexes array of indexes of which chains from the database to score, 
    if NULL all are solved
@param indexesLen indexes array length
@param cards cuda cards index array
@param cardsLen cuda cards index array length, greater or equal to 1
@param thread thread on which the function will be executed, if NULL function is
    executed on the current thread
*/
extern void scoreDatabasesGpu(int** scores, int type, Chain** queries, 
    int queriesLen, ChainDatabaseGpu* chainDatabaseGpu, Scorer* scorer, 
    int* indexes, int indexesLen, int* cards, int cardsLen, Thread* thread);
    
//******************************************************************************

#ifdef __cplusplus 
}
#endif
#endif // __SW_SHARP_GPU_MODULEH__
