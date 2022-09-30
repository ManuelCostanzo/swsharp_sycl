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

@brief Database alignment oriented functions header.
*/

#ifndef __SW_SHARP_DATABASEH__
#define __SW_SHARP_DATABASEH__

#include "chain.h"
#include "db_alignment.h"
#include "scorer.h"
#include "thread.h"

#ifdef __cplusplus 
extern "C" {
#endif

/*!
@brief Ddatabase scoring object.

In the database aligning, queries are often changed and the database is fairly
static. ChainDatabase is the chain database prepared for both CPU and GPU usage
to reduce the preperation time in repetitive aligning.
*/
typedef struct ChainDatabase ChainDatabase;

/*!
Database alignments are often scored by other methods than the alignment score.
ValueFunction defines function type for valueing the align scores between a 
query and the database. Function should calculate the values and store them in
the values array which has length equal to databaseLen. Better alignment scores 
should have smaller value.

@param values output, values of the align scores
@param scores scores between the query and targets in the database
@param query query chain
@param database target chain array
@param databaseLen target chain array length
@param cards cuda cards index array which the database will be available on
@param cardsLen cuda cards index array length, greater or equal to 1
@param param additional parameters for the value function
*/
typedef void (*ValueFunction)(double* values, int* scores, Chain* query, 
    Chain** database, int databaseLen, int* cards, int cardsLen, void* param);

/*!
@brief ChainDatabase constructor.

@param database chain array
@param databaseStart index of the first chain to solve
@param databaseLen length offset from databaseStart to last chain that needs to 
    be solved
@param cards cuda cards index array which the database will be available on
@param cardsLen cuda cards index array length, greater or equal to 1

@return chainDatabase object
*/
extern ChainDatabase* chainDatabaseCreate(Chain** database, int databaseStart, 
    int databaseLen, int* cards, int cardsLen);

/*!
@brief ChainDatabase destructor.

@param chainDatabase chainDatabase object
*/
extern void chainDatabaseDelete(ChainDatabase* chainDatabase);

/*!
@brief Database aligning function.

Function scores the query with every target in the chainDatabase, in other 
words with every chain in the database array with witch the chainDatabase
was created. After the scoring function values the scores with the 
valueFunction and every pair with value over the valueThreshold is discarded.
If there is more than maxAlignments pairs left only the best maxAlignments
pairs are aligned and returned. If the indexes array is given only the targets 
with given indexes will be considered, other will be ignored. 

@param dbAlignments output dbAlignments array, new array is created
@param dbAlignmentsLen output, length of the output dbAlignments array
@param type aligning type, can be #SW_ALIGN, #NW_ALIGN, #HW_ALIGN or #OV_ALIGN
@param query query chain
@param chainDatabase chain database object
@param scorer scorer object used for alignment
@param maxAlignments maximum number of alignments to return, if negative number
    of alignments wont be limited
@param valueFunction function for valueing the alignment scores
@param valueThreshold maximum value of returned alignments
@param valueFunctionParam additional parameters for the value function
@param indexes array of indexes of which chains from the database to score, 
    if NULL all are solved
@param indexesLen indexes array length
@param cards cuda cards index array
@param cardsLen cuda cards index array length, greater or equal to 1
@param thread thread on which the function will be executed, if NULL function is
    executed on the current thread
*/
extern void alignDatabase(DbAlignment*** dbAlignments, int* dbAlignmentsLen, 
    int type, Chain* query, ChainDatabase* chainDatabase, Scorer* scorer, 
    int maxAlignments, ValueFunction valueFunction, void* valueFunctionParam, 
    double valueThreshold, int* indexes, int indexesLen, int* cards, 
    int cardsLen, Thread* thread);
    
/*!
@brief Shotgun aligning function.

Function is the same as the alignDatabase() but it works on the array of 
queries. As result of the an array of arrays of alignments is outputed as well
as the array of coresponding lengths. This function is faster than calling 
alignDatabase() for every query separately.

@param dbAlignments output dbAlignments array of arrays, one for each query,
    new array of arrays is created
@param dbAlignmentsLen output, lengths of the output dbAlignments arrays, 
    one for each query, new array is created
@param type aligning type, can be #SW_ALIGN, #NW_ALIGN, #HW_ALIGN or #OV_ALIGN
@param queries query chains array
@param queriesLen query chains array length
@param chainDatabase chain database object
@param scorer scorer object used for alignment
@param maxAlignments maximum number of alignments to return, if negative number
    of alignments wont be limited
@param valueFunction function for valueing the alignment scores
@param valueThreshold maximum value of returned alignments
@param valueFunctionParam additional parameters for the value function
@param indexes array of indexes of which chains from the database to score, 
    if NULL all are solved
@param indexesLen indexes array length
@param cards cuda cards index array
@param cardsLen cuda cards index array length, greater or equal to 1
@param thread thread on which the function will be executed, if NULL function is
    executed on the current thread
*/
extern void shotgunDatabase(DbAlignment**** dbAlignments, int** dbAlignmentsLen, 
    int type, Chain** queries, int queriesLen, ChainDatabase* chainDatabase, 
    Scorer* scorer, int maxAlignments, ValueFunction valueFunction, 
    void* valueFunctionParam, double valueThreshold, int* indexes, 
    int indexesLen, int* cards, int cardsLen, Thread* thread);

#ifdef __cplusplus 
}
#endif
#endif // __SW_SHARP_DATABASEH__
