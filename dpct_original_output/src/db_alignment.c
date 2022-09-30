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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "alignment.h"
#include "chain.h"
#include "scorer.h"

#include "db_alignment.h"

struct DbAlignment {
    Chain* query;
    int queryStart;
    int queryEnd;
    int queryIdx;
    Chain* target;
    int targetStart;
    int targetEnd;
    int targetIdx;
    double value;
    int score;
    Scorer* scorer;
    char* path;
    int pathLen;
};

//******************************************************************************
// PUBLIC

//******************************************************************************

//******************************************************************************
// PRIVATE
//******************************************************************************

//******************************************************************************
// PUBLIC

//------------------------------------------------------------------------------
// CONSTRUCTOR, DESTRUCTOR

extern DbAlignment* dbAlignmentCreate(Chain* query, int queryStart, int queryEnd,
    int queryIdx, Chain* target, int targetStart, int targetEnd, int targetIdx, 
    double value, int score, Scorer* scorer, char* path, int pathLen) {
    
    DbAlignment* dbAlignment = (DbAlignment*) malloc(sizeof(struct DbAlignment));

    dbAlignment->query = query;
    dbAlignment->queryStart = queryStart;
    dbAlignment->queryEnd = queryEnd;
    dbAlignment->queryIdx = queryIdx;
    dbAlignment->target = target;
    dbAlignment->targetStart = targetStart;
    dbAlignment->targetEnd = targetEnd;
    dbAlignment->targetIdx = targetIdx;
    dbAlignment->value = value;
    dbAlignment->score = score;
    dbAlignment->scorer = scorer;
    dbAlignment->path = path;
    dbAlignment->pathLen = pathLen;
    
    return dbAlignment;
}

extern DbAlignment* dbAlignmentCopy(DbAlignment* other) {

    DbAlignment* dbAlignment = (DbAlignment*) malloc(sizeof(struct DbAlignment));

    dbAlignment->query = other->query;
    dbAlignment->queryStart = other->queryStart;
    dbAlignment->queryEnd = other->queryEnd;
    dbAlignment->queryIdx = other->queryIdx;
    dbAlignment->target = other->target;
    dbAlignment->targetStart = other->targetStart;
    dbAlignment->targetEnd = other->targetEnd;
    dbAlignment->targetIdx = other->targetIdx;
    dbAlignment->value = other->value;
    dbAlignment->score = other->score;
    dbAlignment->scorer = other->scorer;
    dbAlignment->pathLen = other->pathLen;
    
    size_t pathSize = other->pathLen * sizeof(char);
    dbAlignment->path = (char*) malloc(pathSize);
    memcpy(dbAlignment->path, other->path, pathSize);
    
    return dbAlignment;
}

extern void dbAlignmentDelete(DbAlignment* dbAlignment) {
    free(dbAlignment->path);
    free(dbAlignment); 
    dbAlignment = NULL;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// GETTERS

extern char dbAlignmentGetMove(DbAlignment* dbAlignment, int index) {
    return dbAlignment->path[index];
}

extern int dbAlignmentGetPathLen(DbAlignment* dbAlignment) {
    return dbAlignment->pathLen;
}

extern Chain* dbAlignmentGetQuery(DbAlignment* dbAlignment) {
    return dbAlignment->query;
}

extern int dbAlignmentGetQueryEnd(DbAlignment* dbAlignment) {
    return dbAlignment->queryEnd;
}

extern int dbAlignmentGetQueryIdx(DbAlignment* dbAlignment) {
    return dbAlignment->queryIdx;
}

extern int dbAlignmentGetQueryStart(DbAlignment* dbAlignment) {
    return dbAlignment->queryStart;
}

extern int dbAlignmentGetScore(DbAlignment* dbAlignment) {
    return dbAlignment->score;
}

extern Scorer* dbAlignmentGetScorer(DbAlignment* dbAlignment) {
    return dbAlignment->scorer;
}

extern Chain* dbAlignmentGetTarget(DbAlignment* dbAlignment) {
    return dbAlignment->target;
}

extern int dbAlignmentGetTargetEnd(DbAlignment* dbAlignment) {
    return dbAlignment->targetEnd;
}

extern int dbAlignmentGetTargetIdx(DbAlignment* dbAlignment) {
    return dbAlignment->targetIdx;
}

extern int dbAlignmentGetTargetStart(DbAlignment* dbAlignment) {
    return dbAlignment->targetStart;
}

extern double dbAlignmentGetValue(DbAlignment* dbAlignment) {
    return dbAlignment->value;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// FUNCTIONS

extern void dbAlignmentCopyPath(DbAlignment* dbAlignment, char* dest) {
    memcpy(dest, dbAlignment->path, dbAlignment->pathLen);
}

extern Alignment* dbAlignmentToAlignment(DbAlignment* dbAlignment) {

    char* path = (char*) malloc(dbAlignment->pathLen);
    memcpy(path, dbAlignment->path, dbAlignment->pathLen);

    return alignmentCreate(dbAlignment->query, dbAlignment->queryStart, 
        dbAlignment->queryEnd, dbAlignment->target, dbAlignment->targetStart, 
        dbAlignment->targetEnd, dbAlignment->score, dbAlignment->scorer,
        path, dbAlignment->pathLen);
}

//------------------------------------------------------------------------------

//******************************************************************************

//******************************************************************************
// PRIVATE

//******************************************************************************
