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

#include <CL/sycl.hpp>
#ifdef HIP
namespace sycl = cl::sycl;
#endif
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chain.h"
#include "constants.h"
#include "cpu_module.h"
#include "cuda_utils.h"
#include "error.h"
#include "scorer.h"
#include "score_database_gpu_long.h"
#include "score_database_gpu_short.h"
#include "thread.h"
#include "threadpool.h"
#include "utils.h"

#include "gpu_module.h"
#include <cmath>

#include <algorithm>

#define MAX_SHORT_LEN 2800

#define CPU_WORKER_STEP 100
#define CPU_THREADPOOL_STEP 100

typedef struct Context
{
    int **scores;
    int type;
    Chain **queries;
    int queriesLen;
    ChainDatabaseGpu *chainDatabaseGpu;
    Scorer *scorer;
    int *indexes;
    int indexesLen;
    int *cards;
    int cardsLen;
} Context;

typedef struct ContextCpu
{
    int *scores;
    int type;
    Chain **queries;
    int queriesLen;
    Chain **database;
    int databaseLen;
    Scorer *scorer;
    int *indexes;
    int indexesLen;
    int useSimd;
    Mutex *mutex;
    int lastIndexSolved;
    int cancelled;
} ContextCpu;

typedef struct ContextWorkerCpu
{
    int *scores;
    int type;
    Chain **queries;
    int queriesLen;
    Chain **database;
    int databaseLen;
    Scorer *scorer;
    int useSimd;
    Mutex *mutex;
    int *lastQuery;
    int *lastTarget;
    int *cancelled;
} ContextWorkerCpu;

struct ChainDatabaseGpu
{
    Chain **database;
    int databaseLen;
    ShortDatabase *shortDatabase;
    LongDatabase *longDatabase;
    int *longIndexes;
    int longIndexesLen;
};

//******************************************************************************
// PUBLIC

extern ChainDatabaseGpu *chainDatabaseGpuCreate(Chain **database, int databaseLen,
                                                int *cards, int cardsLen);

extern void chainDatabaseGpuDelete(ChainDatabaseGpu *chainDatabaseGpu);

extern void scoreDatabaseGpu(int **scores, int type, Chain *query,
                             ChainDatabaseGpu *chainDatabaseGpu, Scorer *scorer, int *indexes,
                             int indexesLen, int *cards, int cardsLen, Thread *thread);

extern void scoreDatabasesGpu(int **scores, int type, Chain **queries,
                              int queriesLen, ChainDatabaseGpu *chainDatabaseGpu, Scorer *scorer,
                              int *indexes, int indexesLen, int *cards, int cardsLen, Thread *thread);

//******************************************************************************

//******************************************************************************
// PRIVATE

static void scoreDatabase(int **scores, int type, Chain **queries,
                          int queriesLen, ChainDatabaseGpu *chainDatabaseGpu, Scorer *scorer,
                          int *indexes, int indexesLen, int *cards, int cardsLen, Thread *thread);

static void *scoreDatabaseThreadWrapper(void *param);

static void scoreDatabaseThread(int *scores, int type, Chain **queries,
                                int queriesLen, ChainDatabaseGpu *chainDatabaseGpu, Scorer *scorer,
                                int *indexes, int indexesLen, int *cards, int cardsLen, int useSimd);

// cpu workers
static void *scoreCpu(void *param);

static void *scoreCpuWorker(void *param);

// utils
static void filterIndexesArray(int **indexesNew, int *indexesNewLen,
                               int *indexes, int indexesLen, int minIndex, int maxIndex);

static void filterLongIndexesArray(int **longIndexesNew, int *longIndexesNewLen,
                                   int *longIndexes, int longIndexesLen, int *indexes, int indexesLen,
                                   int maxIndex);

static int int2CmpY(const void *a_, const void *b_);

//******************************************************************************

//******************************************************************************
// PUBLIC

extern ChainDatabaseGpu *chainDatabaseGpuCreate(Chain **database, int databaseLen,
                                                int *cards, int cardsLen)
{

    if (cardsLen == 0 || databaseLen == 0)
    {
        return NULL;
    }

    //**************************************************************************
    // CREATE LONG INDEXES

    sycl::int2 *packed = (sycl::int2 *)malloc(databaseLen * sizeof(sycl::int2));
    int packedLen = 0;

    for (int i = 0; i < databaseLen; ++i)
    {

        const int n = chainGetLength(database[i]);

        if (n >= MAX_SHORT_LEN)
        {
            packed[packedLen].x() = i;
            packed[packedLen].y() = n;
            packedLen++;
        }
    }

    qsort(packed, packedLen, sizeof(sycl::int2), int2CmpY);

    int longIndexesLen = packedLen;
    int *longIndexes = (int *)malloc(longIndexesLen * sizeof(int));

    for (int i = 0; i < longIndexesLen; ++i)
    {
        longIndexes[i] = packed[i].x();
    }

    free(packed);

    //**************************************************************************

    //**************************************************************************
    // CREATE GPU DATABASES

    ShortDatabase *shortDatabase = shortDatabaseCreate(database, databaseLen,
                                                       0, MAX_SHORT_LEN, cards, cardsLen);

    LongDatabase *longDatabase = longDatabaseCreate(database, databaseLen,
                                                    MAX_SHORT_LEN, INT_MAX, cards, cardsLen);

    //**************************************************************************

    //**************************************************************************
    // SAVE DATA

    ChainDatabaseGpu *chainDatabaseGpu =
        (ChainDatabaseGpu *)malloc(sizeof(struct ChainDatabaseGpu));

    chainDatabaseGpu->database = database;
    chainDatabaseGpu->databaseLen = databaseLen;
    chainDatabaseGpu->shortDatabase = shortDatabase;
    chainDatabaseGpu->longDatabase = longDatabase;
    chainDatabaseGpu->longIndexes = longIndexes;
    chainDatabaseGpu->longIndexesLen = longIndexesLen;

    //**************************************************************************

    return chainDatabaseGpu;
}

extern void chainDatabaseGpuDelete(ChainDatabaseGpu *chainDatabaseGpu)
{

    if (chainDatabaseGpu != NULL)
    {

        shortDatabaseDelete(chainDatabaseGpu->shortDatabase);
        longDatabaseDelete(chainDatabaseGpu->longDatabase);
        free(chainDatabaseGpu->longIndexes);

        free(chainDatabaseGpu);
    }
}

extern size_t chainDatabaseGpuMemoryConsumption(Chain **database, int databaseLen)
{

    size_t mem1 = shortDatabaseGpuMemoryConsumption(database, databaseLen,
                                                    0, MAX_SHORT_LEN);
    size_t mem2 = longDatabaseGpuMemoryConsumption(database, databaseLen,
                                                   MAX_SHORT_LEN, INT_MAX);

    return mem1 + mem2;
}

extern void scoreDatabaseGpu(int **scores, int type, Chain *query,
                             ChainDatabaseGpu *chainDatabaseGpu, Scorer *scorer, int *indexes,
                             int indexesLen, int *cards, int cardsLen, Thread *thread)
{
    scoreDatabase(scores, type, &query, 1, chainDatabaseGpu, scorer, indexes,
                  indexesLen, cards, cardsLen, thread);
}

extern void scoreDatabasesGpu(int **scores, int type, Chain **queries,
                              int queriesLen, ChainDatabaseGpu *chainDatabaseGpu, Scorer *scorer,
                              int *indexes, int indexesLen, int *cards, int cardsLen, Thread *thread)
{
    scoreDatabase(scores, type, queries, queriesLen, chainDatabaseGpu, scorer,
                  indexes, indexesLen, cards, cardsLen, thread);
}

//******************************************************************************
// PRIVATE

//------------------------------------------------------------------------------
// ENTRY

static void scoreDatabase(int **scores, int type, Chain **queries,
                          int queriesLen, ChainDatabaseGpu *chainDatabaseGpu, Scorer *scorer,
                          int *indexes, int indexesLen, int *cards, int cardsLen, Thread *thread)
{

    ASSERT(cardsLen > 0, "no GPUs available");

    Context *param = (Context *)malloc(sizeof(Context));

    param->scores = scores;
    param->type = type;
    param->queries = queries;
    param->queriesLen = queriesLen;
    param->chainDatabaseGpu = chainDatabaseGpu;
    param->scorer = scorer;
    param->indexes = indexes;
    param->indexesLen = indexesLen;
    param->cards = cards;
    param->cardsLen = cardsLen;

    if (thread == NULL)
    {
        scoreDatabaseThreadWrapper(param);
    }
    else
    {
        threadCreate(thread, scoreDatabaseThreadWrapper, (void *)param);
    }
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// SOLVE

static void *scoreDatabaseThreadWrapper(void *param)
{

    Context *context = (Context *)param;

    int **scores = context->scores;
    int type = context->type;
    Chain **queries = context->queries;
    int queriesLen = context->queriesLen;
    ChainDatabaseGpu *chainDatabaseGpu = context->chainDatabaseGpu;
    Scorer *scorer = context->scorer;
    int *indexes = context->indexes;
    int indexesLen = context->indexesLen;
    int *cards = context->cards;
    int cardsLen = context->cardsLen;

    int gapOpen = scorerGetGapOpen(scorer);
    int databaseLen = chainDatabaseGpu->databaseLen;

    //**************************************************************************
    // INIT RESULTS

    *scores = (int *)malloc(queriesLen * databaseLen * sizeof(int));

    for (int i = 0; i < queriesLen; ++i)
    {
        for (int j = 0; j < databaseLen; ++j)
        {
            (*scores)[i * databaseLen + j] = NO_SCORE;
        }
    }

    //**************************************************************************

    //**************************************************************************
    // FILTER INDEXES

    int *indexesNew = NULL;
    int indexesNewLen;

    filterIndexesArray(&indexesNew, &indexesNewLen, indexes, indexesLen,
                       0, databaseLen - 1);

    //**************************************************************************

    //**************************************************************************
    // CHECK IF SIMD IS AVAILABLE AND USABLE

    int useSimd = false;

    //**************************************************************************

    //**************************************************************************
    // SOLVE

    scoreDatabaseThread(*scores, type, queries, queriesLen, chainDatabaseGpu,
                        scorer, indexesNew, indexesNewLen, cards, cardsLen, useSimd);

    //**************************************************************************

    //**************************************************************************
    // CLEAN MEMORY

    free(indexesNew);
    free(param);

    //**************************************************************************

    return NULL;
}

static void scoreDatabaseThread(int *scores, int type, Chain **queries,
                                int queriesLen, ChainDatabaseGpu *chainDatabaseGpu, Scorer *scorer,
                                int *indexes, int indexesLen, int *cards, int cardsLen, int useSimd)
{

    ShortDatabase *shortDatabase = chainDatabaseGpu->shortDatabase;
    LongDatabase *longDatabase = chainDatabaseGpu->longDatabase;

    int *longIndexes = chainDatabaseGpu->longIndexes;
    int longIndexesLen = chainDatabaseGpu->longIndexesLen;

    Chain **database = chainDatabaseGpu->database;
    int databaseLen = chainDatabaseGpu->databaseLen;

    if (indexes != NULL && indexesLen == 0)
    {
        return;
    }

    int useGpu = indexes == NULL || indexesLen > 100;

    int withThread = !useGpu || withThreads();

    //**************************************************************************
    // FILTER LONG INDEXES

    int *
        longIndexesNew;
    int longIndexesNewLen;

    filterLongIndexesArray(&longIndexesNew, &longIndexesNewLen, longIndexes,
                           longIndexesLen, indexes, indexesLen, databaseLen - 1);

    //**************************************************************************

    //**************************************************************************
    // PREPARE CPU

    Mutex mutex;
    mutexCreate(&mutex);

    ContextCpu contextCpu;
    contextCpu.scores = scores;
    contextCpu.type = type;
    contextCpu.queries = queries;
    contextCpu.queriesLen = queriesLen;
    contextCpu.database = database;
    contextCpu.databaseLen = databaseLen;
    contextCpu.scorer = scorer;
    contextCpu.indexes = useGpu ? longIndexesNew : indexes;
    contextCpu.indexesLen = useGpu ? longIndexesNewLen : indexesLen;
    contextCpu.useSimd = useSimd;
    contextCpu.mutex = &mutex;
    contextCpu.lastIndexSolved = 0;
    contextCpu.cancelled = 0;

    //**************************************************************************

    //**************************************************************************
    // SOLVE MULTICARDED

    TIMER_START("Database solving GPU");

    Thread thread;

    if (withThread)
    {
        threadCreate(&thread, scoreCpu, (void *)&contextCpu);
    }

    if (useGpu)
    {

        TIMER_START("Short solve");

        if (useSimd)
        {
            scoreShortDatabasesPartiallyGpu(scores, type, queries, queriesLen,
                                            shortDatabase, scorer, indexes, indexesLen, 128, cards, cardsLen, NULL);
        }
        else
        {
            scoreShortDatabasesGpu(scores, type, queries, queriesLen,
                                   shortDatabase, scorer, indexes, indexesLen, cards, cardsLen, NULL);
        }

        TIMER_STOP;

        int longInexesSolved = 0;
        if (withThread)
        {
            mutexLock(contextCpu.mutex);

            longInexesSolved = contextCpu.lastIndexSolved;
            contextCpu.cancelled = 1;

            mutexUnlock(contextCpu.mutex);
        }

        LOG("Long indexes solved CPU: %d", longInexesSolved);

        TIMER_START("Long solve");

        printf("%d %d\n", longIndexesLen, longIndexesNewLen);
        if (!withThread || (withThread && longInexesSolved < longIndexesNewLen))
        {
            scoreLongDatabasesGpu(scores, type, queries, queriesLen,
                                  longDatabase, scorer, longIndexesNew + longInexesSolved,
                                  longIndexesNewLen - longInexesSolved, cards, cardsLen, NULL);
        }

        TIMER_STOP;
    }

    if (withThread)
    {
        threadJoin(thread);
    }

    TIMER_STOP;

    //**************************************************************************

    //**************************************************************************
    // SOLVE OVERFLOWS

    if (useSimd)
    {

        TIMER_START("Solving overflows");

        int maxScore = scorerGetMaxScore(scorer);

        for (int i = 0; i < queriesLen; ++i)
        {

            int *overflows = (int *)malloc(databaseLen * sizeof(int));
            int overflowsLen = 0;

            for (int j = 0; j < databaseLen; ++j)
            {
                if (scores[i * databaseLen + j] == 128)
                {
                    overflows[overflowsLen++] = j;
                }
            }

            scoreDatabaseThread(scores + i * databaseLen, type, queries + i, 1,
                                chainDatabaseGpu, scorer, overflows, overflowsLen, cards, cardsLen, 0);

            free(overflows);
        }

        TIMER_STOP;
    }
    else
    {

#ifdef DEBUG

        int overflows = 0;
        for (int i = 0; i < queriesLen; ++i)
        {
            for (int j = 0; j < databaseLen; ++j)
            {
                if (scores[i * databaseLen + j] >= 128)
                {
                    overflows++;
                }
            }
        }

        LOG("Scores over 127: %d", overflows);
#endif
    }

    //**************************************************************************

    //**************************************************************************
    // CLEAN MEMORY

    mutexDelete(&mutex);

    if (longIndexesNew != longIndexes)
    {
        free(longIndexesNew);
    }

    //**************************************************************************
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// CPU WORKER

static void *scoreCpu(void *param)
{

    ContextCpu *context = (ContextCpu *)param;

    int *scores = context->scores;
    int type = context->type;
    Chain **queries = context->queries;
    int queriesLen = context->queriesLen;
    Chain **database_ = context->database;
    int databaseLen = context->databaseLen;
    Scorer *scorer = context->scorer;
    int *indexes = context->indexes;
    int indexesLen = context->indexesLen;
    int useSimd = context->useSimd;
    Mutex *mutex = context->mutex;

    if (indexesLen == 0)
    {
        return NULL;
    }

    TIMER_START("Outter solving CPU: %d", indexesLen);

    //**************************************************************************
    // CREATE DATABASE

    Chain **database = (Chain **)malloc(indexesLen * sizeof(Chain *));

    for (int i = 0; i < indexesLen; ++i)
    {
        database[i] = database_[indexes[i]];
    }

    //**************************************************************************

    //**************************************************************************
    // PREPARE WORKER CONTEXT

    int *scoresCpu = (int *)malloc(queriesLen * indexesLen * sizeof(int));
    int lastQuery = 0;

    ContextWorkerCpu workerContext;
    workerContext.scores = scoresCpu;
    workerContext.type = type;
    workerContext.queries = queries;
    workerContext.queriesLen = queriesLen;
    workerContext.database = database;
    workerContext.databaseLen = indexesLen;
    workerContext.scorer = scorer;
    workerContext.useSimd = useSimd;
    workerContext.mutex = mutex;
    workerContext.lastQuery = &lastQuery;
    workerContext.lastTarget = &(context->lastIndexSolved);
    workerContext.cancelled = &(context->cancelled);

    //**************************************************************************

    //**************************************************************************
    // SOLVE MULTITHREADED

    int tasksNmr = CPU_THREADPOOL_STEP;
    ThreadPoolTask **tasks = (ThreadPoolTask **)malloc(tasksNmr * sizeof(ThreadPoolTask *));

    int over = 0;
    while (!over)
    {

        for (int i = 0; i < tasksNmr; ++i)
        {
            tasks[i] = threadPoolSubmitToFront(scoreCpuWorker, &workerContext);
        }

        for (int i = 0; i < tasksNmr; ++i)
        {
            threadPoolTaskWait(tasks[i]);
            threadPoolTaskDelete(tasks[i]);
        }

        mutexLock(mutex);

        if (context->cancelled || context->lastIndexSolved >= indexesLen)
        {
            over = 1;
        }

        mutexUnlock(mutex);
    }

    //**************************************************************************

    //**************************************************************************
    // SAVE SCORES

    int lastIndexSolved = context->lastIndexSolved;

    for (int i = 0; i < queriesLen; ++i)
    {
        for (int j = 0; j < lastIndexSolved; ++j)
        {
            scores[i * databaseLen + indexes[j]] = scoresCpu[i * indexesLen + j];
        }
    }

    //**************************************************************************

    //**************************************************************************
    // CLEAN MEMORY

    free(database);
    free(tasks);
    free(scoresCpu);

    //**************************************************************************

    TIMER_STOP;

    return NULL;
}

static void *scoreCpuWorker(void *param)
{

    ContextWorkerCpu *context = (ContextWorkerCpu *)param;

    int *scores_ = context->scores;
    int type = context->type;
    Chain **queries = context->queries;
    int queriesLen = context->queriesLen;
    Chain **database_ = context->database;
    int databaseLen = context->databaseLen;
    Scorer *scorer = context->scorer;
    int useSimd = context->useSimd;
    Mutex *mutex = context->mutex;
    int *lastQuery = context->lastQuery;
    int *lastTarget = context->lastTarget;
    int *cancelled = context->cancelled;

    mutexLock(mutex);

    if (*lastQuery >= queriesLen)
    {
        *lastQuery = 0;
        *lastTarget += std::min(CPU_WORKER_STEP, databaseLen - *lastTarget);
    }

    int queryIdx = *lastQuery;
    int start = *lastTarget;
    int length = std::min(CPU_WORKER_STEP, databaseLen - start);

    if (start >= databaseLen || *cancelled)
    {
        mutexUnlock(mutex);
        return NULL;
    }

    (*lastQuery)++;

    mutexUnlock(mutex);

    int *scores = scores_ + queryIdx * databaseLen + start;

    Chain *query = queries[queryIdx];
    Chain **database = database_ + start;

    if (useSimd)
    {
        scoreDatabasePartiallyCpu(scores, type, query, database, length, scorer, 128);
    }
    else
    {
        scoreDatabaseCpu(scores, type, query, database, length, scorer);
    }

    return NULL;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// UTILS

static void filterIndexesArray(int **indexesNew, int *indexesNewLen,
                               int *indexes, int indexesLen, int minIndex, int maxIndex)
{

    if (indexes == NULL)
    {
        *indexesNew = NULL;
        *indexesNewLen = 0;
        return;
    }

    *indexesNew = (int *)malloc(indexesLen * sizeof(int));
    *indexesNewLen = 0;

    for (int i = 0; i < indexesLen; ++i)
    {

        int idx = indexes[i];

        if (idx >= minIndex && idx <= maxIndex)
        {
            (*indexesNew)[*indexesNewLen] = idx;
            (*indexesNewLen)++;
        }
    }
}

static void filterLongIndexesArray(int **longIndexesNew, int *longIndexesNewLen,
                                   int *longIndexes, int longIndexesLen, int *indexes, int indexesLen,
                                   int maxIndex)
{

    if (indexes == NULL)
    {
        *longIndexesNew = longIndexes;
        *longIndexesNewLen = longIndexesLen;
        return;
    }

    int *mask = (int *)calloc(maxIndex + 1, sizeof(int));

    for (int i = 0; i < indexesLen; ++i)
    {
        mask[indexes[i]] = 1;
    }

    *longIndexesNew = (int *)malloc(longIndexesLen * sizeof(int));
    *longIndexesNewLen = 0;

    for (int i = 0; i < longIndexesLen; ++i)
    {
        if (mask[longIndexes[i]])
        {
            (*longIndexesNew)[*longIndexesNewLen] = longIndexes[i];
            (*longIndexesNewLen)++;
        }
    }

    free(mask);
}

static int int2CmpY(const void *a_, const void *b_)
{

    sycl::int2 a = *((sycl::int2 *)a_);
    sycl::int2 b = *((sycl::int2 *)b_);

    return a.y() - b.y();
}

//------------------------------------------------------------------------------
//******************************************************************************
