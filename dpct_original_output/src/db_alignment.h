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

@brief Database sequnce alignment result storage header.
*/

#ifndef __SW_SHARP_DBALIGNMENTH__
#define __SW_SHARP_DBALIGNMENTH__

#include "alignment.h"
#include "chain.h"
#include "scorer.h"

#ifdef __cplusplus 
extern "C" {
#endif

/*!
@brief Database sequnce alignment result storage object.

Database alignment object is fairly similiar to ::Alignment object. In addition
it stores query index in the query database and the target index in the target
database. Also database alignments are often scored by other methods than the
alignment score, database alignment value is also stored. Database alignment 
value representation is user defined.
*/
typedef struct DbAlignment DbAlignment;

/*!
@brief DbAlignment object constructor.

Alignment object is constructed from the query and target sequence aligned and 
their coresponding start and stop positions, alignment score, scorer which was
used for alignment and the alignment path. None of the input objects are copied
via the constructor.

@param query query sequnce
@param queryStart query start position
@param queryEnd query end position, inclusive
@param queryIdx query index
@param target target sequnce
@param targetStart target start position
@param targetEnd target end position, inclusive
@param targetIdx target index
@param value alignment value
@param score alignment score
@param scorer scorer object used for alignment
@param path alignment path
@param pathLen alignment path length

@return alignment object
*/
extern DbAlignment* dbAlignmentCreate(Chain* query, int queryStart, int queryEnd,
    int queryIdx, Chain* target, int targetStart, int targetEnd, int targetIdx, 
    double value, int score, Scorer* scorer, char* path, int pathLen);

extern DbAlignment* dbAlignmentCopy(DbAlignment* dbAlignment);

/*!
@brief DbAlignment destructor.

@param dbAlignment dbAlignment object 
*/
extern void dbAlignmentDelete(DbAlignment* dbAlignment);

/*!
@brief Move getter.

Given index must be greater or equal to zero and less than dbAlignment path 
length.

@param dbAlignment dbAlignment object 
@param index path move index

@return path move
*/
extern char dbAlignmentGetMove(DbAlignment* dbAlignment, int index);

/*!
@brief Path length getter.

@param dbAlignment dbAlignment object 

@return path length 
*/
extern int dbAlignmentGetPathLen(DbAlignment* dbAlignment);

/*!
@brief Query getter.

@param dbAlignment dbAlignment object 

@return query 
*/
extern Chain* dbAlignmentGetQuery(DbAlignment* dbAlignment);

/*!
@brief Query end getter.

@param dbAlignment dbAlignment object 

@return query end 
*/
extern int dbAlignmentGetQueryEnd(DbAlignment* dbAlignment);

/*!
@brief Query index getter.

@param dbAlignment dbAlignment object 

@return query index 
*/
extern int dbAlignmentGetQueryIdx(DbAlignment* dbAlignment);

/*!
@brief Query start getter.

@param dbAlignment dbAlignment object 

@return query start 
*/
extern int dbAlignmentGetQueryStart(DbAlignment* dbAlignment);

/*!
@brief Score getter.

@param dbAlignment dbAlignment object 

@return score
*/
extern int dbAlignmentGetScore(DbAlignment* dbAlignment);

/*!
@brief Scorer getter.

@param dbAlignment dbAlignment object 

@return scorer 
*/
extern Scorer* dbAlignmentGetScorer(DbAlignment* dbAlignment);

/*!
@brief Target getter.

@param dbAlignment dbAlignment object 

@return target 
*/
extern Chain* dbAlignmentGetTarget(DbAlignment* dbAlignment);

/*!
@brief Target end getter.

@param dbAlignment dbAlignment object 

@return target end
*/
extern int dbAlignmentGetTargetEnd(DbAlignment* dbAlignment);

/*!
@brief Target index getter.

@param dbAlignment dbAlignment object 

@return target index
*/
extern int dbAlignmentGetTargetIdx(DbAlignment* dbAlignment);

/*!
@brief Target start getter.

@param dbAlignment dbAlignment object 

@return target start
*/
extern int dbAlignmentGetTargetStart(DbAlignment* dbAlignment);

/*!
@brief Value getter.

@param dbAlignment dbAlignment object 

@return value
*/
extern double dbAlignmentGetValue(DbAlignment* dbAlignment);

/*!
@brief Copies path to the destination buffer.

Method copies path to the destination buffer which should be at least long as 
the database alignment path length.

@param dbAlignment dbAlignment object 
@param dest destination buffer
*/
extern void dbAlignmentCopyPath(DbAlignment* dbAlignment, char* dest);

/*!
@brief Creates alignment object from the dbAlignment object.

@param dbAlignment dbAlignment object 

@return alignment object
*/
extern Alignment* dbAlignmentToAlignment(DbAlignment* dbAlignment);

#ifdef __cplusplus 
}
#endif
#endif // __SW_SHARP_DBALIGNMENTH__
