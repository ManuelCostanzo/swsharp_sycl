/*
This code represents a SYCL-compatible, DPC++-based version of SW#.
Copyright (C) 2022 Manuel Costanzo, contributor Enzo Rucci.

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

Contact SW# author by mkorpar@gmail.com.

Contact SW#-SYCL authors by mcostanzo@lidi.info.unlp.edu.ar, erucci@lidi.info.unlp.edu.ar
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chain.h"
#include "scorer.h"

#include "alignment.h"

struct Alignment
{
    Chain *query;
    int queryStart;
    int queryEnd;
    Chain *target;
    int targetStart;
    int targetEnd;
    int score;
    Scorer *scorer;
    char *path;
    int pathLen;
};

//******************************************************************************
// PUBLIC

//******************************************************************************

//******************************************************************************
// PRIVATE

static void unzipPath(char **path, int *pathLen, char *bytes, int bytesLen);

static void zipPath(char **bytes, int *bytesLen, char *path, int pathLen);

//******************************************************************************

//******************************************************************************
// PUBLIC

//------------------------------------------------------------------------------
// CONSTRUCTOR, DESTRUCTOR

extern Alignment *alignmentCreate(Chain *query, int queryStart, int queryEnd,
                                  Chain *target, int targetStart, int targetEnd, int score, Scorer *scorer,
                                  char *path, int pathLen)
{

    Alignment *alignment = (Alignment *)malloc(sizeof(struct Alignment));

    alignment->query = query;
    alignment->queryStart = queryStart;
    alignment->queryEnd = queryEnd;
    alignment->target = target;
    alignment->targetStart = targetStart;
    alignment->targetEnd = targetEnd;
    alignment->score = score;
    alignment->scorer = scorer;
    alignment->path = path;
    alignment->pathLen = pathLen;

    return alignment;
}

extern void alignmentDelete(Alignment *alignment)
{
    free(alignment->path);
    free(alignment);
    alignment = NULL;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// GETTERS

extern char alignmentGetMove(Alignment *alignment, int index)
{
    return alignment->path[index];
}

extern int alignmentGetPathLen(Alignment *alignment)
{
    return alignment->pathLen;
}

extern Chain *alignmentGetQuery(Alignment *alignment)
{
    return alignment->query;
}

extern int alignmentGetQueryEnd(Alignment *alignment)
{
    return alignment->queryEnd;
}

extern int alignmentGetQueryStart(Alignment *alignment)
{
    return alignment->queryStart;
}

extern int alignmentGetScore(Alignment *alignment)
{
    return alignment->score;
}

extern Scorer *alignmentGetScorer(Alignment *alignment)
{
    return alignment->scorer;
}

extern Chain *alignmentGetTarget(Alignment *alignment)
{
    return alignment->target;
}

extern int alignmentGetTargetEnd(Alignment *alignment)
{
    return alignment->targetEnd;
}

extern int alignmentGetTargetStart(Alignment *alignment)
{
    return alignment->targetStart;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// FUNCTIONS

extern void alignmentCopyPath(Alignment *alignment, char *dest)
{
    memcpy(dest, alignment->path, alignment->pathLen);
}

extern Alignment *alignmentDeserialize(char *bytes)
{

    int ptr = 0;

    int queryBytesLen;
    memcpy(&queryBytesLen, bytes + ptr, sizeof(int));
    ptr += sizeof(int);

    Chain *query = chainDeserialize(bytes + ptr);
    ptr += queryBytesLen;

    int queryStart;
    memcpy(&queryStart, bytes + ptr, sizeof(int));
    ptr += sizeof(int);

    int queryEnd;
    memcpy(&queryEnd, bytes + ptr, sizeof(int));
    ptr += sizeof(int);

    int targetBytesLen;
    memcpy(&targetBytesLen, bytes + ptr, sizeof(int));
    ptr += sizeof(int);

    Chain *target = chainDeserialize(bytes + ptr);
    ptr += targetBytesLen;

    int targetStart;
    memcpy(&targetStart, bytes + ptr, sizeof(int));
    ptr += sizeof(int);

    int targetEnd;
    memcpy(&targetEnd, bytes + ptr, sizeof(int));
    ptr += sizeof(int);

    int score;
    memcpy(&score, bytes + ptr, sizeof(int));
    ptr += sizeof(int);

    int scorerBytesLen;
    memcpy(&scorerBytesLen, bytes + ptr, sizeof(int));
    ptr += sizeof(int);

    Scorer *scorer = scorerDeserialize(bytes + ptr);
    ptr += scorerBytesLen;

    int pathBytesLen;
    memcpy(&pathBytesLen, bytes + ptr, sizeof(int));
    ptr += sizeof(int);

    char *pathBytes = (char *)malloc(pathBytesLen);
    memcpy(pathBytes, bytes + ptr, pathBytesLen);
    ptr += pathBytesLen;

    char *path;
    int pathLen;
    unzipPath(&path, &pathLen, pathBytes, pathBytesLen);

    free(pathBytes);

    Alignment *alignment = (Alignment *)malloc(sizeof(struct Alignment));

    alignment->query = query;
    alignment->queryStart = queryStart;
    alignment->queryEnd = queryEnd;
    alignment->target = target;
    alignment->targetStart = targetStart;
    alignment->targetEnd = targetEnd;
    alignment->score = score;
    alignment->scorer = scorer;
    alignment->path = path;
    alignment->pathLen = pathLen;

    return alignment;
}

extern void alignmentSerialize(char **bytes, int *bytesLen, Alignment *alignment)
{

    char *queryBytes;
    int queryBytesLen;
    chainSerialize(&queryBytes, &queryBytesLen, alignment->query);

    char *targetBytes;
    int targetBytesLen;
    chainSerialize(&targetBytes, &targetBytesLen, alignment->target);

    char *scorerBytes;
    int scorerBytesLen;
    scorerSerialize(&scorerBytes, &scorerBytesLen, alignment->scorer);

    char *pathBytes;
    int pathBytesLen;
    zipPath(&pathBytes, &pathBytesLen, alignment->path, alignment->pathLen);

    *bytesLen = 0;
    *bytesLen += sizeof(int);    // query
    *bytesLen += queryBytesLen;  // query
    *bytesLen += sizeof(int);    // queryStart
    *bytesLen += sizeof(int);    // queryEnd
    *bytesLen += sizeof(int);    // target
    *bytesLen += targetBytesLen; // target
    *bytesLen += sizeof(int);    // targetStart
    *bytesLen += sizeof(int);    // targetEnd
    *bytesLen += sizeof(int);    // score
    *bytesLen += sizeof(int);    // scorer
    *bytesLen += scorerBytesLen; // scorer
    *bytesLen += sizeof(int);    // path
    *bytesLen += pathBytesLen;   // path

    *bytes = (char *)malloc(*bytesLen);

    int ptr = 0;

    memcpy(*bytes + ptr, &queryBytesLen, sizeof(int));
    ptr += sizeof(int);

    memcpy(*bytes + ptr, queryBytes, queryBytesLen);
    ptr += queryBytesLen;

    memcpy(*bytes + ptr, &alignment->queryStart, sizeof(int));
    ptr += sizeof(int);

    memcpy(*bytes + ptr, &alignment->queryEnd, sizeof(int));
    ptr += sizeof(int);

    memcpy(*bytes + ptr, &targetBytesLen, sizeof(int));
    ptr += sizeof(int);

    memcpy(*bytes + ptr, targetBytes, targetBytesLen);
    ptr += targetBytesLen;

    memcpy(*bytes + ptr, &alignment->targetStart, sizeof(int));
    ptr += sizeof(int);

    memcpy(*bytes + ptr, &alignment->targetEnd, sizeof(int));
    ptr += sizeof(int);

    memcpy(*bytes + ptr, &alignment->score, sizeof(int));
    ptr += sizeof(int);

    memcpy(*bytes + ptr, &scorerBytesLen, sizeof(int));
    ptr += sizeof(int);

    memcpy(*bytes + ptr, scorerBytes, scorerBytesLen);
    ptr += scorerBytesLen;

    memcpy(*bytes + ptr, &pathBytesLen, sizeof(int));
    ptr += sizeof(int);

    memcpy(*bytes + ptr, pathBytes, pathBytesLen);
    ptr += pathBytesLen;

    free(pathBytes);
    free(queryBytes);
    free(targetBytes);
    free(scorerBytes);
}

//------------------------------------------------------------------------------
//******************************************************************************

//******************************************************************************
// PRIVATE

static void unzipPath(char **path, int *pathLen, char *bytes, int bytesLen)
{

    int ptr = 0;

    memcpy(pathLen, bytes + ptr, sizeof(int));
    ptr += sizeof(int);

    *path = (char *)malloc(*pathLen);

    int offset = 0;

    while (ptr < bytesLen)
    {

        char move;
        memcpy(&move, bytes + ptr, 1);
        ptr += 1;

        int n;
        memcpy(&n, bytes + ptr, sizeof(int));
        ptr += sizeof(int);

        memset(*path + offset, move, n);
        offset += n;
    }
}

static void zipPath(char **bytes, int *bytesLen, char *path, int pathLen)
{

    size_t size = pathLen + pathLen * sizeof(int) + sizeof(int);
    *bytes = (char *)malloc(size);

    int ptr = 0;

    memcpy(*bytes + ptr, &pathLen, sizeof(int));
    ptr += sizeof(int);

    int i = 0;
    while (i < pathLen)
    {

        char move = path[i];
        int n = 1;

        while (i + 1 < pathLen && move == path[i + 1])
        {
            i++;
            n++;
        }

        memcpy(*bytes + ptr, &move, 1);
        ptr += 1;

        memcpy(*bytes + ptr, &n, sizeof(int));
        ptr += sizeof(int);

        i++;
    }

    *bytesLen = ptr;
}

//******************************************************************************
