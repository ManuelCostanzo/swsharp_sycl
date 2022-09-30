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

#ifndef __CUDACC__

#include "chain.h"
#include "error.h"
#include "scorer.h"
#include "thread.h"

#include "gpu_module.h"

static const char *errorMessage = "CUDA not available";

struct ChainDatabaseGpu
{
};

extern void hwEndDataGpu(int *queryEnd, int *targetEnd, int *outScore,
                         Chain *query, Chain *target, Scorer *scorer, int score, int card,
                         Thread *thread)
{
    ERROR("%s", errorMessage);
}

extern void nwFindScoreGpu(int *queryStart, int *targetStart, Chain *query,
                           int queryFrontGap, Chain *target, Scorer *scorer, int score, int card,
                           Thread *thread)
{
    ERROR("%s", errorMessage);
}

extern void nwLinearDataGpu(int **scores, int **affines, Chain *query,
                            int queryFrontGap, Chain *target, int targetFrontGap, Scorer *scorer,
                            int pLeft, int pRight, int card, Thread *thread)
{
    ERROR("%s", errorMessage);
}

extern void ovEndDataGpu(int *queryEnd, int *targetEnd, int *outScore,
                         Chain *query, Chain *target, Scorer *scorer, int score, int card,
                         Thread *thread)
{
    ERROR("%s", errorMessage);
}

extern void ovFindScoreGpu(int *queryStart, int *targetStart, Chain *query,
                           Chain *target, Scorer *scorer, int score, int card, Thread *thread)
{
    ERROR("%s", errorMessage);
}

extern void swEndDataGpu(int *queryEnd, int *targetEnd, int *outScore,
                         int **scores, int **affines, Chain *query, Chain *target, Scorer *scorer,
                         int score, int card, Thread *thread)
{
    ERROR("%s", errorMessage);
}

extern ChainDatabaseGpu *chainDatabaseGpuCreate(Chain **database, int databaseLen,
                                                int *cards, int cardsLen)
{
    return NULL;
}

extern void chainDatabaseGpuDelete(ChainDatabaseGpu *chainDatabaseGpu)
{
}

extern size_t chainDatabaseGpuMemoryConsumption(Chain **database, int databaseLen)
{
    return 0;
}

extern void scoreDatabaseGpu(int **scores, int type, Chain *query,
                             ChainDatabaseGpu *chainDatabaseGpu, Scorer *scorer, int *indexes,
                             int indexesLen, int *cards, int cardsLen, Thread *thread)
{
    ERROR("%s", errorMessage);
}

extern void scoreDatabasesGpu(int **scores, int type, Chain **queries,
                              int queriesLen, ChainDatabaseGpu *chainDatabaseGpu, Scorer *scorer,
                              int *indexes, int indexesLen, int *cards, int cardsLen, Thread *thread)
{
    ERROR("%s", errorMessage);
}

#endif // __CUDACC__
