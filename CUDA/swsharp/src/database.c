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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "align.h"
#include "alignment.h"
#include "cpu_module.h"
#include "chain.h"
#include "constants.h"
#include "cuda_utils.h"
#include "db_alignment.h"
#include "error.h"
#include "gpu_module.h"
#include "post_proc.h"
#include "scorer.h"
#include "thread.h"
#include "threadpool.h"
#include "utils.h"

#include "database.h"

#define CPU_THREAD_CHUNK 1000
#define CPU_PACKED_CHUNK 25

#define GPU_DB_MIN_CELLS 49000000ll
#define GPU_MIN_CELLS 40000000ll
#define GPU_MIN_LEN 256

typedef struct Context
{
    DbAlignment ***dbAlignments;
    int *dbAlignmentsLen;
    int type;
    Chain **queries;
    int queriesLen;
    ChainDatabase *chainDatabase;
    Scorer *scorer;
    int maxAlignments;
    ValueFunction valueFunction;
    void *valueFunctionParam;
    double valueThreshold;
    int *indexes;
    int indexesLen;
    int *cards;
    int cardsLen;
    Thread *thread;
} Context;

typedef struct DbAlignmentData
{
    int idx;
    int score;
    double value;
    const char *name;
} DbAlignmentData;

typedef struct ExtractContext
{
    DbAlignmentData **dbAlignmentData;
    int *dbAlignmentLen;
    Chain *query;
    Chain **database;
    int databaseLen;
    int *scores;
    int maxAlignments;
    ValueFunction valueFunction;
    void *valueFunctionParam;
    double valueThreshold;
    int *cards;
    int cardsLen;
} ExtractContext;

typedef struct ExtractContexts
{
    ExtractContext *contexts;
    int contextsLen;
} ExtractContexts;

typedef struct AlignContext
{
    DbAlignment **dbAlignment;
    int type;
    Chain *query;
    int queryIdx;
    Chain *target;
    int targetIdx;
    double value;
    int score;
    Scorer *scorer;
    int *cards;
    int cardsLen;
    long long cells;
} AlignContext;

typedef struct AlignContexts
{
    AlignContext **contexts;
    int contextsLen;
    long long cells;
} AlignContexts;

typedef struct AlignContextsPacked
{
    AlignContext *contexts;
    int contextsLen;
} AlignContextsPacked;

typedef struct ScoreCpuContext
{
    int *scores;
    int type;
    Chain *query;
    Chain **database;
    int databaseLen;
    Scorer *scorer;
} ScoreCpuContext;

struct ChainDatabase
{
    ChainDatabaseGpu *chainDatabaseGpu;
    Chain **database;
    int databaseStart;
    int databaseLen;
    long databaseElems;
};

//******************************************************************************
// PUBLIC

extern ChainDatabase *chainDatabaseCreate(Chain **database, int databaseStart,
                                          int databaseLen, int *cards, int cardsLen);

extern void chainDatabaseDelete(ChainDatabase *chainDatabase);

extern void alignDatabase(DbAlignment ***dbAlignments, int *dbAlignmentsLen,
                          int type, Chain *query, ChainDatabase *chainDatabase, Scorer *scorer,
                          int maxAlignments, ValueFunction valueFunction, void *valueFunctionParam,
                          double valueThreshold, int *indexes, int indexesLen, int *cards,
                          int cardsLen, Thread *thread);

extern void shotgunDatabase(DbAlignment ****dbAlignments, int **dbAlignmentsLen,
                            int type, Chain **queries, int queriesLen, ChainDatabase *chainDatabase,
                            Scorer *scorer, int maxAlignments, ValueFunction valueFunction,
                            void *valueFunctionParam, double valueThreshold, int *indexes,
                            int indexesLen, int *cards, int cardsLen, Thread *thread);

//******************************************************************************

//******************************************************************************
// PRIVATE

static void databaseSearch(DbAlignment ***dbAlignments, int *dbAlignmentsLen,
                           int type, Chain **queries, int queriesLen, ChainDatabase *chainDatabase,
                           Scorer *scorer, int maxAlignments, ValueFunction valueFunction,
                           void *valueFunctionParam, double valueThreshold, int *indexes,
                           int indexesLen, int *cards, int cardsLen, Thread *thread);

static void *databaseSearchThread(void *param);

static void databaseSearchStep(DbAlignment ***dbAlignments,
                               int *dbAlignmentsLen, int type, Chain **queries, int queriesStart,
                               int queriesLen, ChainDatabase *chainDatabase, Scorer *scorer,
                               int maxAlignments, ValueFunction valueFunction, void *valueFunctionParam,
                               double valueThreshold, int *indexes, int indexesLen, int *cards,
                               int cardsLen);

static void *alignThread(void *param);

static void *alignsThread(void *param);

static void *alignsPackedThread(void *param);

static void *extractThread(void *param);

static void *extractsThread(void *param);

static void scoreCpu(int **scores, int type, Chain **queries,
                     int queriesLen, Chain **database, int databaseLen, Scorer *scorer,
                     int *indexes, int indexesLen);

static void *scoreCpuThread(void *param);

static void filterIndexesArray(int **indexesNew, int *indexesNewLen,
                               int *indexes, int indexesLen, int minIndex, int maxIndex);

static int dbAlignmentDataCmp(const void *a_, const void *b_);

//******************************************************************************

//******************************************************************************
// PUBLIC

extern ChainDatabase *chainDatabaseCreate(Chain **database, int databaseStart,
                                          int databaseLen, int *cards, int cardsLen)
{

    ChainDatabase *db = (ChainDatabase *)malloc(sizeof(struct ChainDatabase));

    TIMER_START("Creating database");

    db->database = database + databaseStart;
    db->databaseStart = databaseStart;
    db->databaseLen = databaseLen;

    int i;
    long databaseElems = 0;
    for (i = 0; i < databaseLen; ++i)
    {
        databaseElems += chainGetLength(db->database[i]);
    }
    db->databaseElems = databaseElems;

    db->chainDatabaseGpu = chainDatabaseGpuCreate(db->database, databaseLen,
                                                  cards, cardsLen);

    TIMER_STOP;

    return db;
}

extern void chainDatabaseDelete(ChainDatabase *chainDatabase)
{

    chainDatabaseGpuDelete(chainDatabase->chainDatabaseGpu);

    free(chainDatabase);
    chainDatabase = NULL;
}

extern void alignDatabase(DbAlignment ***dbAlignments, int *dbAlignmentsLen,
                          int type, Chain *query, ChainDatabase *chainDatabase, Scorer *scorer,
                          int maxAlignments, ValueFunction valueFunction, void *valueFunctionParam,
                          double valueThreshold, int *indexes, int indexesLen, int *cards,
                          int cardsLen, Thread *thread)
{

    databaseSearch(dbAlignments, dbAlignmentsLen, type, &query, 1,
                   chainDatabase, scorer, maxAlignments, valueFunction, valueFunctionParam,
                   valueThreshold, indexes, indexesLen, cards, cardsLen, thread);
}

extern void shotgunDatabase(DbAlignment ****dbAlignments, int **dbAlignmentsLen,
                            int type, Chain **queries, int queriesLen, ChainDatabase *chainDatabase,
                            Scorer *scorer, int maxAlignments, ValueFunction valueFunction,
                            void *valueFunctionParam, double valueThreshold, int *indexes,
                            int indexesLen, int *cards, int cardsLen, Thread *thread)
{

    *dbAlignments = (DbAlignment ***)malloc(queriesLen * sizeof(DbAlignment **));
    *dbAlignmentsLen = (int *)malloc(queriesLen * sizeof(int));

    databaseSearch(*dbAlignments, *dbAlignmentsLen, type, queries, queriesLen,
                   chainDatabase, scorer, maxAlignments, valueFunction, valueFunctionParam,
                   valueThreshold, indexes, indexesLen, cards, cardsLen, thread);
}

//******************************************************************************

//******************************************************************************
// PRIVATE

//------------------------------------------------------------------------------
// SEARCH

static void databaseSearch(DbAlignment ***dbAlignments, int *dbAlignmentsLen,
                           int type, Chain **queries, int queriesLen, ChainDatabase *chainDatabase,
                           Scorer *scorer, int maxAlignments, ValueFunction valueFunction,
                           void *valueFunctionParam, double valueThreshold, int *indexes,
                           int indexesLen, int *cards, int cardsLen, Thread *thread)
{

    Context *param = (Context *)malloc(sizeof(Context));

    param->dbAlignments = dbAlignments;
    param->dbAlignmentsLen = dbAlignmentsLen;
    param->type = type;
    param->queries = queries;
    param->queriesLen = queriesLen;
    param->chainDatabase = chainDatabase;
    param->scorer = scorer;
    param->maxAlignments = maxAlignments;
    param->valueFunction = valueFunction;
    param->valueFunctionParam = valueFunctionParam;
    param->valueThreshold = valueThreshold;
    param->indexes = indexes;
    param->indexesLen = indexesLen;
    param->cards = cards;
    param->cardsLen = cardsLen;

    if (thread == NULL)
    {
        databaseSearchThread(param);
    }
    else
    {
        threadCreate(thread, databaseSearchThread, (void *)param);
    }
}

static void *databaseSearchThread(void *param)
{

    Context *context = (Context *)param;

    DbAlignment ***dbAlignments = context->dbAlignments;
    int *dbAlignmentsLen = context->dbAlignmentsLen;
    int type = context->type;
    Chain **queries = context->queries;
    int queriesLen = context->queriesLen;
    ChainDatabase *chainDatabase = context->chainDatabase;
    Scorer *scorer = context->scorer;
    int maxAlignments = context->maxAlignments;
    ValueFunction valueFunction = context->valueFunction;
    void *valueFunctionParam = context->valueFunctionParam;
    double valueThreshold = context->valueThreshold;
    int *cards = context->cards;
    int cardsLen = context->cardsLen;

    int databaseStart = chainDatabase->databaseStart;
    int databaseLen = chainDatabase->databaseLen;

    TIMER_START("Database search");

    int i;

    //**************************************************************************
    // FIX INDEXES

    int *indexes;
    int indexesLen;

    filterIndexesArray(&indexes, &indexesLen, context->indexes,
                       context->indexesLen, databaseStart, databaseStart + databaseLen - 1);

    for (i = 0; i < indexesLen; ++i)
    {
        indexes[i] -= databaseStart;
    }

    //**************************************************************************

    //**************************************************************************
    // FIX ARGUMENTS

    if (indexes != NULL)
    {
        maxAlignments = MIN(indexesLen, maxAlignments);
    }

    if (maxAlignments < 0)
    {
        maxAlignments = databaseLen;
    }

    //**************************************************************************

    //**************************************************************************
    // DO THE ALIGN

    double memory = (double)databaseLen * queriesLen * sizeof(int); // scores
    memory = (memory * 1.15) / 1024.0 / 1024.0;                     // 15% offset and to MB

    // chop in pieces
    int steps = MIN(queriesLen, (int)ceil(memory / (1 * 1024.0)));
    int queriesChunk = queriesLen / steps;
    int queriesAdd = queriesLen % steps;
    int offset = 0;

    LOG("need %.2lfMB total, %d queries, solving in %d steps", memory,
        queriesLen, steps);

    for (i = 0; i < steps; ++i)
    {

        int length = queriesChunk + (i < queriesAdd);

        LOG("Solving %d-%d", offset, offset + length);

        databaseSearchStep(dbAlignments + offset, dbAlignmentsLen + offset,
                           type, queries + offset, offset, length, chainDatabase, scorer,
                           maxAlignments, valueFunction, valueFunctionParam, valueThreshold,
                           indexes, indexesLen, cards, cardsLen);

        offset += length;
    }

    //**************************************************************************

    //**************************************************************************
    // CLEAN MEMORY

    free(indexes); // copy

    free(param);

    //**************************************************************************

    TIMER_STOP;

    return NULL;
}

static void databaseSearchStep(DbAlignment ***dbAlignments,
                               int *dbAlignmentsLen, int type, Chain **queries, int queriesStart,
                               int queriesLen, ChainDatabase *chainDatabase, Scorer *scorer,
                               int maxAlignments, ValueFunction valueFunction, void *valueFunctionParam,
                               double valueThreshold, int *indexes, int indexesLen, int *cards,
                               int cardsLen)
{

    Chain **database = chainDatabase->database;
    int databaseStart = chainDatabase->databaseStart;
    int databaseLen = chainDatabase->databaseLen;
    long databaseElems = chainDatabase->databaseElems;
    ChainDatabaseGpu *chainDatabaseGpu = chainDatabase->chainDatabaseGpu;

    int i, j, k;

    //**************************************************************************
    // CALCULATE CELL NUMBER

    long queriesElems = 0;
    for (i = 0; i < queriesLen; ++i)
    {
        queriesElems += chainGetLength(queries[i]);
    }

    if (indexes != NULL)
    {

        databaseElems = 0;

        for (i = 0; i < indexesLen; ++i)
        {
            databaseElems += chainGetLength(database[indexes[i]]);
        }
    }

    long long cells = (long long)queriesElems * databaseElems;

    //**************************************************************************

    //**************************************************************************
    // CALCULATE SCORES

    int *scores;

    if (cells < GPU_DB_MIN_CELLS || cardsLen == 0)
    {
        scoreCpu(&scores, type, queries, queriesLen, database,
                 databaseLen, scorer, indexes, indexesLen);
    }
    else
    {
        scoreDatabasesGpu(&scores, type, queries, queriesLen, chainDatabaseGpu,
                          scorer, indexes, indexesLen, cards, cardsLen, NULL);
    }

    //**************************************************************************

    //**************************************************************************
    // EXTRACT BEST CHAINS AND SAVE THEIR DATA MULTITHREADED

    TIMER_START("Extract best");

    DbAlignmentData **dbAlignmentsData =
        (DbAlignmentData **)malloc(queriesLen * sizeof(DbAlignmentData *));

    ExtractContext *eContexts =
        (ExtractContext *)malloc(queriesLen * sizeof(ExtractContext));

    for (i = 0; i < queriesLen; ++i)
    {
        eContexts[i].dbAlignmentData = &(dbAlignmentsData[i]);
        eContexts[i].dbAlignmentLen = &(dbAlignmentsLen[i]);
        eContexts[i].query = queries[i];
        eContexts[i].database = database;
        eContexts[i].databaseLen = databaseLen;
        eContexts[i].scores = scores + i * databaseLen;
        eContexts[i].maxAlignments = maxAlignments;
        eContexts[i].valueFunction = valueFunction;
        eContexts[i].valueFunctionParam = valueFunctionParam;
        eContexts[i].valueThreshold = valueThreshold;
        eContexts[i].cards = cards;
        eContexts[i].cardsLen = cardsLen;
    }

    if (cardsLen == 0)
    {

        size_t tasksSize = queriesLen * sizeof(ThreadPoolTask *);
        ThreadPoolTask **tasks = (ThreadPoolTask **)malloc(tasksSize);

        for (i = 0; i < queriesLen; ++i)
        {
            tasks[i] = threadPoolSubmit(extractThread, (void *)&(eContexts[i]));
        }

        for (i = 0; i < queriesLen; ++i)
        {
            threadPoolTaskWait(tasks[i]);
            threadPoolTaskDelete(tasks[i]);
        }

        free(tasks);
    }
    else
    {

        int chunks = MIN(queriesLen, cardsLen);

        int cardsChunk = cardsLen / chunks;
        int cardsAdd = cardsLen % chunks;
        int cardsOff = 0;

        int contextsChunk = queriesLen / chunks;
        int contextsAdd = queriesLen % chunks;
        int contextsOff = 0;

        size_t contextsSize = chunks * sizeof(ExtractContexts);
        ExtractContexts *contexts = (ExtractContexts *)malloc(contextsSize);

        size_t tasksSize = chunks * sizeof(Thread);
        Thread *tasks = (Thread *)malloc(tasksSize);

        for (i = 0; i < chunks; ++i)
        {

            int *cards_ = cards + cardsOff;
            int cardsLen_ = cardsChunk + (i < cardsAdd);
            cardsOff += cardsLen_;

            ExtractContext *contexts_ = eContexts + contextsOff;
            int contextsLen_ = contextsChunk + (i < contextsAdd);
            contextsOff += contextsLen_;

            for (j = 0; j < contextsLen_; ++j)
            {
                contexts_[j].cards = cards_;
                contexts_[j].cardsLen = cardsLen_;
            }

            contexts[i].contexts = contexts_;
            contexts[i].contextsLen = contextsLen_;

            threadCreate(&(tasks[i]), extractsThread, &(contexts[i]));
        }

        for (i = 0; i < chunks; ++i)
        {
            threadJoin(tasks[i]);
        }

        free(tasks);
        free(contexts);
    }

    free(eContexts);
    free(scores); // this is big, release immediately

    TIMER_STOP;

    //**************************************************************************

    //**************************************************************************
    // ALIGN BEST TARGETS MULTITHREADED

    TIMER_START("Database aligning");

    // create structure
    for (i = 0; i < queriesLen; ++i)
    {
        size_t dbAlignmentsSize = dbAlignmentsLen[i] * sizeof(DbAlignment *);
        dbAlignments[i] = (DbAlignment **)malloc(dbAlignmentsSize);
    }

    // count tasks
    int aTasksLen = 0;
    for (i = 0; i < queriesLen; ++i)
    {
        aTasksLen += dbAlignmentsLen[i];
    }

    size_t aTasksSize = aTasksLen * sizeof(ThreadPoolTask *);
    ThreadPoolTask **aTasks = (ThreadPoolTask **)malloc(aTasksSize);

    size_t aContextsSize = aTasksLen * sizeof(AlignContext);
    AlignContext *aContextsCpu = (AlignContext *)malloc(aContextsSize);
    AlignContext *aContextsGpu = (AlignContext *)malloc(aContextsSize);
    int aContextsCpuLen = 0;
    int aContextsGpuLen = 0;

    for (i = 0, k = 0; i < queriesLen; ++i, ++k)
    {

        Chain *query = queries[i];
        int rows = chainGetLength(query);

        for (j = 0; j < dbAlignmentsLen[i]; ++j, ++k)
        {

            DbAlignmentData data = dbAlignmentsData[i][j];
            Chain *target = database[data.idx];

            int cols = chainGetLength(target);
            long long cells = (long long)rows * cols;

            AlignContext *context;
            if (cols < GPU_MIN_LEN || cells < GPU_MIN_CELLS || cardsLen == 0)
            {
                context = &(aContextsCpu[aContextsCpuLen++]);
                context->cards = NULL;
                context->cardsLen = 0;
            }
            else
            {
                context = &(aContextsGpu[aContextsGpuLen++]);
            }

            context->dbAlignment = &(dbAlignments[i][j]);
            context->type = type;
            context->query = query;
            context->queryIdx = i;
            context->target = target;
            context->targetIdx = data.idx + databaseStart;
            context->value = data.value;
            context->score = data.score;
            context->scorer = scorer;
            context->cells = cells;
        }
    }

    LOG("Aligning %d cpu, %d gpu", aContextsCpuLen, aContextsGpuLen);

    // run cpu tasks
    int aCpuTasksLen;
    AlignContextsPacked *aContextsCpuPacked;

    if (aContextsCpuLen < 10000)
    {

        aCpuTasksLen = aContextsCpuLen;
        aContextsCpuPacked = NULL;

        for (i = 0; i < aCpuTasksLen; ++i)
        {
            aTasks[i] = threadPoolSubmit(alignThread, &(aContextsCpu[i]));
        }
    }
    else
    {

        aCpuTasksLen = aContextsCpuLen / CPU_PACKED_CHUNK;
        aCpuTasksLen += (aContextsCpuLen % CPU_PACKED_CHUNK) != 0;

        size_t contextsSize = aCpuTasksLen * sizeof(AlignContextsPacked);
        AlignContextsPacked *contexts = (AlignContextsPacked *)malloc(contextsSize);

        for (i = 0; i < aCpuTasksLen; ++i)
        {

            int length = MIN(CPU_PACKED_CHUNK, aContextsCpuLen - i * CPU_PACKED_CHUNK);

            contexts[i].contexts = aContextsCpu + i * CPU_PACKED_CHUNK;
            contexts[i].contextsLen = length;
        }

        for (i = 0; i < aCpuTasksLen; ++i)
        {
            aTasks[i] = threadPoolSubmit(alignsPackedThread, &(contexts[i]));
        }

        aContextsCpuPacked = contexts;
    }

    if (aContextsGpuLen)
    {

        int chunks = MIN(aContextsGpuLen, cardsLen);

        size_t contextsSize = chunks * sizeof(AlignContexts);
        AlignContexts *contexts = (AlignContexts *)malloc(contextsSize);

        size_t balancedSize = chunks * aContextsGpuLen * sizeof(AlignContext *);
        AlignContext **balanced = (AlignContext **)malloc(balancedSize);

        // set phony contexts, init data
        for (i = 0; i < chunks; ++i)
        {
            contexts[i].contexts = balanced + i * aContextsGpuLen;
            contexts[i].contextsLen = 0;
            contexts[i].cells = 0;
        }

        // balance tasks by round roobin, chunks are pretty small (CUDA cards)
        for (i = 0; i < aContextsGpuLen; ++i)
        {

            int minIdx = 0;
            long long min = contexts[0].cells;
            for (j = 1; j < chunks; ++j)
            {
                if (contexts[j].cells < min)
                {
                    min = contexts[j].cells;
                    minIdx = j;
                }
            }

            AlignContext *context = &(aContextsGpu[i]);
            contexts[minIdx].contexts[contexts[minIdx].contextsLen++] = context;
            contexts[minIdx].cells += context->cells;
        }

        // set context cards
        int cardsChunk = cardsLen / chunks;
        int cardsAdd = cardsLen % chunks;
        int cardsOff = 0;

        for (i = 0; i < chunks; ++i)
        {

            int cCardsLen = cardsChunk + (i < cardsAdd);
            int *cCards = cards + cardsOff;
            cardsOff += cCardsLen;

            for (j = 0; j < contexts[i].contextsLen; ++j)
            {
                contexts[i].contexts[j]->cards = cCards;
                contexts[i].contexts[j]->cardsLen = cCardsLen;
            }
        }

        size_t tasksSize = chunks * sizeof(Thread);
        Thread *tasks = (Thread *)malloc(tasksSize);

        // run gpu tasks first
        for (i = 0; i < chunks; ++i)
        {
            threadCreate(&(tasks[i]), alignsThread, &(contexts[i]));
        }

        // wait for gpu tasks to finish
        for (i = 0; i < chunks; ++i)
        {
            threadJoin(tasks[i]);
        }

        free(balanced);
        free(contexts);
    }

    // wait for cpu tasks
    for (i = 0; i < aCpuTasksLen; ++i)
    {
        threadPoolTaskWait(aTasks[i]);
        threadPoolTaskDelete(aTasks[i]);
    }

    free(aContextsCpuPacked);
    free(aContextsCpu);
    free(aContextsGpu);
    free(aTasks);

    TIMER_STOP;

    //**************************************************************************

    //**************************************************************************
    // CLEAN MEMORY

    for (i = 0; i < queriesLen; ++i)
    {
        free(dbAlignmentsData[i]);
    }
    free(dbAlignmentsData);

    //**************************************************************************
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// THREADS

static void *alignThread(void *param)
{

    AlignContext *context = (AlignContext *)param;

    DbAlignment **dbAlignment = context->dbAlignment;
    int type = context->type;
    Chain *query = context->query;
    int queryIdx = context->queryIdx;
    Chain *target = context->target;
    int targetIdx = context->targetIdx;
    double value = context->value;
    int score = context->score;
    Scorer *scorer = context->scorer;
    int *cards = context->cards;
    int cardsLen = context->cardsLen;

    // align
    Alignment *alignment;
    alignScoredPair(&alignment, type, query, target, scorer, score, cards, cardsLen, NULL);

    // check scores
    int s1 = alignmentGetScore(alignment);
    int s2 = score;

    ASSERT(s1 == s2, "Scores don't match %d %d, (%s %s)", s1, s2,
           chainGetName(query), chainGetName(target));

    // extract info
    int queryStart = alignmentGetQueryStart(alignment);
    int queryEnd = alignmentGetQueryEnd(alignment);
    int targetStart = alignmentGetTargetStart(alignment);
    int targetEnd = alignmentGetTargetEnd(alignment);
    int pathLen = alignmentGetPathLen(alignment);

    char *path = (char *)malloc(pathLen);
    alignmentCopyPath(alignment, path);

    alignmentDelete(alignment);

    // create db alignment
    *dbAlignment = dbAlignmentCreate(query, queryStart, queryEnd, queryIdx,
                                     target, targetStart, targetEnd, targetIdx, value, score, scorer, path,
                                     pathLen);

    return NULL;
}

static void *alignsThread(void *param)
{

    AlignContexts *context = (AlignContexts *)param;
    AlignContext **contexts = context->contexts;
    int contextsLen = context->contextsLen;

    int i = 0;
    for (i = 0; i < contextsLen; ++i)
    {
        alignThread(contexts[i]);
    }

    return NULL;
}

static void *alignsPackedThread(void *param)
{

    AlignContextsPacked *context = (AlignContextsPacked *)param;
    AlignContext *contexts = context->contexts;
    int contextsLen = context->contextsLen;

    int i = 0;
    for (i = 0; i < contextsLen; ++i)
    {
        alignThread(&(contexts[i]));
    }

    return NULL;
}

static void *extractThread(void *param)
{

    ExtractContext *context = (ExtractContext *)param;

    DbAlignmentData **dbAlignmentData = context->dbAlignmentData;
    int *dbAlignmentLen = context->dbAlignmentLen;
    Chain *query = context->query;
    Chain **database = context->database;
    int databaseLen = context->databaseLen;
    int *scores = context->scores;
    int maxAlignments = context->maxAlignments;
    ValueFunction valueFunction = context->valueFunction;
    void *valueFunctionParam = context->valueFunctionParam;
    double valueThreshold = context->valueThreshold;
    int *cards = context->cards;
    int cardsLen = context->cardsLen;

    int i;

    size_t packedSize = databaseLen * sizeof(DbAlignmentData);
    DbAlignmentData *packed = (DbAlignmentData *)malloc(packedSize);
    double *values = (double *)malloc(databaseLen * sizeof(double));

    valueFunction(values, scores, query, database, databaseLen,
                  cards, cardsLen, valueFunctionParam);

    int thresholded = 0;
    for (i = 0; i < databaseLen; ++i)
    {

        packed[i].idx = i;
        packed[i].value = values[i];
        packed[i].score = scores[i];
        packed[i].name = chainGetName(database[i]);

        if (packed[i].value <= valueThreshold)
        {
            thresholded++;
        }
    }

    int k = MIN(thresholded, maxAlignments);
    qselect((void *)packed, databaseLen, sizeof(DbAlignmentData), k, dbAlignmentDataCmp);
    qsort((void *)packed, k, sizeof(DbAlignmentData), dbAlignmentDataCmp);

    *dbAlignmentData = (DbAlignmentData *)malloc(k * sizeof(DbAlignmentData));
    *dbAlignmentLen = k;

    for (i = 0; i < k; ++i)
    {
        (*dbAlignmentData)[i] = packed[i];
    }

    free(packed);
    free(values);

    return NULL;
}

static void *extractsThread(void *param)
{

    ExtractContexts *context = (ExtractContexts *)param;
    ExtractContext *contexts = context->contexts;
    int contextsLen = context->contextsLen;

    int i = 0;
    for (i = 0; i < contextsLen; ++i)
    {
        extractThread(contexts + i);
    }

    return NULL;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// CPU MODULES

static void scoreCpu(int **scores_, int type, Chain **queries,
                     int queriesLen, Chain **database_, int databaseLen_, Scorer *scorer,
                     int *indexes, int indexesLen)
{

    TIMER_START("CPU database scoring");

    *scores_ = (int *)malloc(queriesLen * databaseLen_ * sizeof(int));

    int i, j;

    int *scores;

    Chain **database;
    int databaseLen;

    //**************************************************************************
    // INIT STRUCTURES

    if (indexes == NULL)
    {

        scores = *scores_;

        database = database_;
        databaseLen = databaseLen_;
    }
    else
    {

        scores = (int *)malloc(indexesLen * queriesLen * sizeof(int));

        database = (Chain **)malloc(indexesLen * sizeof(Chain *));
        databaseLen = indexesLen;

        for (i = 0; i < indexesLen; ++i)
        {
            database[i] = database_[indexes[i]];
        }
    }

    //**************************************************************************

    //**************************************************************************
    // SOLVE MULTITHREADED

    int maxLen = (queriesLen * databaseLen) / CPU_THREAD_CHUNK + queriesLen;
    int length = 0;

    size_t contextsSize = maxLen * sizeof(ScoreCpuContext);
    ScoreCpuContext *contexts = (ScoreCpuContext *)malloc(contextsSize);

    size_t tasksSize = maxLen * sizeof(ThreadPoolTask *);
    ThreadPoolTask **tasks = (ThreadPoolTask **)malloc(tasksSize);

    for (i = 0; i < queriesLen; ++i)
    {
        for (j = 0; j < databaseLen; j += CPU_THREAD_CHUNK)
        {

            contexts[length].scores = scores + i * databaseLen + j;
            contexts[length].type = type;
            contexts[length].query = queries[i];
            contexts[length].database = database + j;
            contexts[length].databaseLen = MIN(CPU_THREAD_CHUNK, databaseLen - j);
            contexts[length].scorer = scorer;

            tasks[length] = threadPoolSubmit(scoreCpuThread, &(contexts[length]));

            length++;
        }
    }

    for (i = 0; i < length; ++i)
    {
        threadPoolTaskWait(tasks[i]);
        threadPoolTaskDelete(tasks[i]);
    }

    free(tasks);
    free(contexts);

    //**************************************************************************

    //**************************************************************************
    // SAVE RESULTS

    if (indexes != NULL)
    {

        for (i = 0; i < queriesLen; ++i)
        {

            for (j = 0; j < databaseLen_; ++j)
            {
                (*scores_)[i * databaseLen_ + j] = NO_SCORE;
            }

            for (j = 0; j < indexesLen; ++j)
            {
                (*scores_)[i * databaseLen_ + indexes[j]] = scores[i * indexesLen + j];
            }
        }

        free(database);
        free(scores);
    }

    //**************************************************************************

    TIMER_STOP;
}

static void *scoreCpuThread(void *param)
{

    ScoreCpuContext *context = (ScoreCpuContext *)param;

    int *scores = context->scores;
    int type = context->type;
    Chain *query = context->query;
    Chain **database = context->database;
    int databaseLen = context->databaseLen;
    Scorer *scorer = context->scorer;

    scoreDatabaseCpu(scores, type, query, database, databaseLen, scorer);

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

    int i;
    for (i = 0; i < indexesLen; ++i)
    {

        int idx = indexes[i];

        if (idx >= minIndex && idx <= maxIndex)
        {
            (*indexesNew)[*indexesNewLen] = idx;
            (*indexesNewLen)++;
        }
    }
}

static int dbAlignmentDataCmp(const void *a_, const void *b_)
{

    DbAlignmentData *a = (DbAlignmentData *)a_;
    DbAlignmentData *b = (DbAlignmentData *)b_;

    if (a->value == b->value)
    {

        if (a->score == b->score)
        {
            return strcmp(a->name, b->name);
        }

        return b->score - a->score;
    }

    if (a->value < b->value)
        return -1;
    return 1;
}

//------------------------------------------------------------------------------
//******************************************************************************
