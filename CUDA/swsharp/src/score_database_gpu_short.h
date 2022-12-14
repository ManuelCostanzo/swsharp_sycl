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

#ifndef __SW_SHARP_SCORE_DATABASE_SHORTH__
#define __SW_SHARP_SCORE_DATABASE_SHORTH__

#include "alignment.h"
#include "chain.h"
#include "constants.h"
#include "scorer.h"
#include "thread.h"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct ShortDatabase ShortDatabase;

    extern ShortDatabase *shortDatabaseCreate(Chain **database, int databaseLen,
                                              int minLen, int maxLen, int *cards, int cardsLen);

    extern void shortDatabaseDelete(ShortDatabase *shortDatabase);

    extern size_t shortDatabaseGpuMemoryConsumption(Chain **database,
                                                    int databaseLen, int minLen, int maxLen);

    extern void scoreShortDatabaseGpu(int *scores, int type, Chain *query,
                                      ShortDatabase *shortDatabase, Scorer *scorer, int *indexes, int indexesLen,
                                      int *cards, int cardsLen, Thread *thread);

    extern void scoreShortDatabasesGpu(int *scores, int type, Chain **queries,
                                       int queriesLen, ShortDatabase *shortDatabase, Scorer *scorer, int *indexes,
                                       int indexesLen, int *cards, int cardsLen, Thread *thread);

    extern void scoreShortDatabasePartiallyGpu(int *scores, int type, Chain *query,
                                               ShortDatabase *shortDatabase, Scorer *scorer, int *indexes, int indexesLen,
                                               int maxScore, int *cards, int cardsLen, Thread *thread);

    extern void scoreShortDatabasesPartiallyGpu(int *scores, int type,
                                                Chain **queries, int queriesLen, ShortDatabase *shortDatabase,
                                                Scorer *scorer, int *indexes, int indexesLen, int maxScore, int *cards,
                                                int cardsLen, Thread *thread);

#ifdef __cplusplus
}
#endif
#endif // __SW_SHARP_SCORE_DATABASE_SHORTH__
