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

@brief Pairwise sequnce alignment result storage header.
*/

#ifndef __SW_SHARP_ALIGNMENTH__
#define __SW_SHARP_ALIGNMENTH__

#include "chain.h"
#include "scorer.h"

#ifdef __cplusplus 
extern "C" {
#endif

/*!
@brief Stop move.

Stop move is not used in alignment path. However it is used internaly in 
sequnce alignment algorithms.
*/
#define MOVE_STOP 0

/*!
@brief Path insertion move.

Named diagonal move because of the up move in matrix done while backtracking.
*/
#define MOVE_LEFT 2

/*!
@brief Path deletion move.

Named diagonal move because of the up move in matrix done while backtracking.
*/
#define MOVE_UP 3

/*!
@brief Path alignment move.

Named diagonal move because of the diagonal move in matrix done while
backtracking.
*/
#define MOVE_DIAG 1

/*!
@brief Pairwise sequnce alignment result storage object.

All off the pairwise sequnce alignment algorithms produce a similiar result.
Input query and target sequences are stored as ::Chain object. Algorithm scoring
system is stored as ::Scorer object. Alignment stores the query and target
start and endpoints as well as the alignment score. Alignment path is stored 
as a character array. Every character represents one move and can be #MOVE_LEFT,
#MOVE_UP or #MOVE_DIAG. Alignment path is stored in format convinient for 
backtracking, in other words the moves are named by the matrix movement while
backtracking.
*/
typedef struct Alignment Alignment;

/*!
@brief Alignment object constructor.

Alignment object is constructed from the query and target sequence aligned and 
their coresponding start and stop positions, alignment score, scorer which was
used for alignment and the alignment path. None of the input objects are copied
via the constructor.

@param query query sequnce
@param queryStart query start position
@param queryEnd query end position, inclusive
@param target target sequnce
@param targetStart target start position
@param targetEnd target end position, inclusive
@param score alignment score
@param scorer scorer object used for alignment
@param path alignment path
@param pathLen alignment path length

@return alignment object
*/
extern Alignment* alignmentCreate(Chain* query, int queryStart, int queryEnd,
    Chain* target, int targetStart, int targetEnd, int score, Scorer* scorer,
    char* path, int pathLen);

/*!
@brief Alignment destructor.

@param alignment alignment object 
*/
extern void alignmentDelete(Alignment* alignment);

/*!
@brief Move getter.

Given index must be greater or equal to zero and less than alignment path 
length.

@param alignment alignment object 
@param index path move index

@return path move
*/
extern char alignmentGetMove(Alignment* alignment, int index);

/*!
@brief Path len getter.

@param alignment alignment object 

@return path len
*/
extern int alignmentGetPathLen(Alignment* alignment);

/*!
@brief Query getter.

@param alignment alignment object 

@return query
*/
extern Chain* alignmentGetQuery(Alignment* alignment);

/*!
@brief Query end getter.

@param alignment alignment object 

@return query end
*/
extern int alignmentGetQueryEnd(Alignment* alignment);

/*!
@brief Query start getter.

@param alignment alignment object 

@return query start
*/
extern int alignmentGetQueryStart(Alignment* alignment);

/*!
@brief Score getter.

@param alignment alignment object 

@return score
*/
extern int alignmentGetScore(Alignment* alignment);

/*!
@brief Scorer getter.

@param alignment alignment object 

@return scorer
*/
extern Scorer* alignmentGetScorer(Alignment* alignment);

/*!
@brief Target getter.

@param alignment alignment object 

@return target
*/
extern Chain* alignmentGetTarget(Alignment* alignment);

/*!
@brief Target end getter.

@param alignment alignment object 

@return target end
*/
extern int alignmentGetTargetEnd(Alignment* alignment);

/*!
@brief Target start getter.

@param alignment alignment object 

@return target start
*/
extern int alignmentGetTargetStart(Alignment* alignment);

/*!
@brief Copies path to the destination buffer.

Method copies path to the destination buffer which should be at least long as 
the alignment path length.

@param alignment alignment object 
@param dest destination buffer
*/
extern void alignmentCopyPath(Alignment* alignment, char* dest);

/*!
@brief Alignment deserialization method.

Method deserializes alignment object from a byte buffer.

@param bytes byte buffer

@return alignment object
*/
extern Alignment* alignmentDeserialize(char* bytes);

/*!
@brief Alignment serialization method.

Method serializes alignment object to a byte buffer.

@param bytes output byte buffer
@param bytesLen output byte buffer length
@param alignment alignment object
*/
extern void alignmentSerialize(char** bytes, int* bytesLen, Alignment* alignment);

#ifdef __cplusplus 
}
#endif
#endif // __SW_SHARP_ALIGNMENTH__
