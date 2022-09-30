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

#ifndef __SW_SHARP_SCORE_DATABASE_LONGH__
#define __SW_SHARP_SCORE_DATABASE_LONGH__

#include "alignment.h"
#include "chain.h"
#include "constants.h"
#include "scorer.h"
#include "thread.h"

#ifdef __cplusplus 
extern "C" {
#endif

typedef struct LongDatabase LongDatabase;

extern LongDatabase* longDatabaseCreate(Chain** database, int databaseLen, 
    int minLen, int maxLen, int* cards, int cardsLen);
extern void longDatabaseDelete(LongDatabase* longDatabase);

extern size_t longDatabaseGpuMemoryConsumption(Chain** database, int databaseLen,
    int minLen, int maxLen);

extern void scoreLongDatabaseGpu(int* scores, int type, Chain* query, 
    LongDatabase* longDatabase, Scorer* scorer, int* indexes, int indexesLen, 
    int* cards, int cardsLen, Thread* thread);

extern void scoreLongDatabasesGpu(int* scores, int type, Chain** queries, 
    int queriesLen, LongDatabase* longDatabase, Scorer* scorer, int* indexes, 
    int indexesLen, int* cards, int cardsLen, Thread* thread);
    
#ifdef __cplusplus 
}
#endif
#endif // __SW_SHARP_SCORE_DATABASE_LONGH__
