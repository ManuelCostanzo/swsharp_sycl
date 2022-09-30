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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chain.h"
#include "constants.h"
#include "cuda_utils.h"
#include "error.h"
#include "scorer.h"
#include "thread.h"
#include "utils.h"

#include "score_database_gpu_long.h"
#include <cmath>

#include <algorithm>

#define THREADS 64
#define BLOCKS 240

#define MAX_THREADS THREADS

#define INT4_ZERO sycl::int4(0, 0, 0, 0)
#define INT4_SCORE_MIN sycl::int4(SCORE_MIN, SCORE_MIN, SCORE_MIN, SCORE_MIN)

typedef struct GpuDatabase
{
    int card;
    char *codes;
    int *starts;
    int *lengths;
    int *indexes;
    int *scores;
    sycl::int2 *hBus;
} GpuDatabase;

struct LongDatabase
{
    Chain **database;
    int databaseLen;
    int length;
    int *order;
    int *positions;
    int *indexes;
    GpuDatabase *gpuDatabases;
    int gpuDatabasesLen;
};

typedef struct Context
{
    int *scores;
    int type;
    Chain **queries;
    int queriesLen;
    LongDatabase *longDatabase;
    Scorer *scorer;
    int *indexes;
    int indexesLen;
    int *cards;
    int cardsLen;
} Context;

typedef struct QueryProfile
{
    int height;
    int width;
    int length;
    sycl::char4 *data;
    size_t size;
} QueryProfile;

typedef struct QueryProfileGpu
{
    sycl::char4 *data;
} QueryProfileGpu;

typedef void (*ScoringFunction)(char *, int *, int *, int *, int *,
                                sycl::int2 *);

typedef struct KernelContext
{
    int *scores;
    ScoringFunction scoringFunction;
    QueryProfile *queryProfile;
    Chain *query;
    LongDatabase *longDatabase;
    Scorer *scorer;
    int *indexes;
    int indexesLen;
    int card;
} KernelContext;

typedef struct KernelContexts
{
    KernelContext *contexts;
    int contextsLen;
    long long cells;
} KernelContexts;

typedef struct Atom
{
    int mch;
    sycl::int2 up;
    sycl::int4 lScr;
    sycl::int4 lAff;
    sycl::int4 rScr;
    sycl::int4 rAff;
} Atom;

static int type;

//******************************************************************************
// PUBLIC

extern LongDatabase *longDatabaseCreate(Chain **database, int databaseLen,
                                        int minLen, int maxLen, int *cards, int cardsLen);

extern void longDatabaseDelete(LongDatabase *longDatabase);

extern void scoreLongDatabaseGpu(int *scores, int type, Chain *query,
                                 LongDatabase *longDatabase, Scorer *scorer, int *indexes, int indexesLen,
                                 int *cards, int cardsLen, Thread *thread);

extern void scoreLongDatabasesGpu(int *scores, int type, Chain **queries,
                                  int queriesLen, LongDatabase *longDatabase, Scorer *scorer, int *indexes,
                                  int indexesLen, int *cards, int cardsLen, Thread *thread);

//******************************************************************************

//******************************************************************************
// PRIVATE

// constructor
static LongDatabase *createDatabase(Chain **database, int databaseLen,
                                    int minLen, int maxLen, int *cards, int cardsLen);

// destructor
static void deleteDatabase(LongDatabase *database);

// scoring
static void scoreDatabase(int *scores, int type, Chain **queries,
                          int queriesLen, LongDatabase *longDatabase, Scorer *scorer, int *indexes,
                          int indexesLen, int *cards, int cardsLen, Thread *thread);

static void *scoreDatabaseGpuLongThread(void *param);

static void scoreDatabaseMulti(int *scores, ScoringFunction scoringFunction,
                               Chain **queries, int queriesLen, LongDatabase *longDatabase, Scorer *scorer,
                               int *indexes, int indexesLen, int *cards, int cardsLen);

static void scoreDatabaseSingle(int *scores, ScoringFunction scoringFunction,
                                Chain **queries, int queriesLen, LongDatabase *longDatabase, Scorer *scorer,
                                int *indexes, int indexesLen, int *cards, int cardsLen);

// cpu kernels
static void *kernelThread(void *param);

static void *kernelsThread(void *param);

// gpu kernels
void hwSolve(char *codes, int *starts, int *lengths, int *indexes, int *scores,
             sycl::int2 *hBus, sycl::nd_item<1> item_ct1, int gapOpen_,
             int gapExtend_, int rows_, int rowsPadded_, int length_,
             int iters_, int *scoresShr, int *hBusScrShr, int *hBusAffShr, int qpWidth_,
             sycl::char4 *qpGpu);

void nwSolve(char *codes, int *starts, int *lengths, int *indexes, int *scores,
             sycl::int2 *hBus, sycl::nd_item<1> item_ct1, int gapOpen_,
             int gapExtend_, int rows_, int rowsPadded_, int length_,
             int iters_, int *scoresShr, int *hBusScrShr, int *hBusAffShr, int qpWidth_,
             sycl::char4 *qpGpu);

void ovSolve(char *codes, int *starts, int *lengths, int *indexes, int *scores,
             sycl::int2 *hBus, sycl::nd_item<1> item_ct1, int gapOpen_,
             int gapExtend_, int rows_, int rowsPadded_, int length_,
             int iters_, int *scoresShr, int *hBusScrShr, int *hBusAffShr, int qpWidth_,
             sycl::char4 *qpGpu);

void swSolve(char *codes, int *starts, int *lengths, int *indexes, int *scores,
             sycl::int2 *hBus, sycl::nd_item<1> item_ct1, int gapOpen_,
             int gapExtend_, int rows_, int rowsPadded_, int length_,
             int iters_, int *scoresShr, int *hBusScrShr, int *hBusAffShr, int qpWidth_,
             sycl::char4 *qpGpu);

static int gap(int index, int gapOpen_, int gapExtend_);

void hwSolveSingle(int id, char *codes, int *starts, int *lengths, int *scores,
                   sycl::int2 *hBus, sycl::nd_item<1> item_ct1, int gapOpen_,
                   int gapExtend_, int rows_, int rowsPadded_, int iters_,
                   int *scoresShr, int *hBusScrShr, int *hBusAffShr, int qpWidth_,
                   sycl::char4 *qpGpu);

void nwSolveSingle(int id, char *codes, int *starts, int *lengths, int *scores,
                   sycl::int2 *hBus, sycl::nd_item<1> item_ct1, int gapOpen_,
                   int gapExtend_, int rows_, int rowsPadded_, int iters_,
                   int *scoresShr, int *hBusScrShr, int *hBusAffShr, int qpWidth_,
                   sycl::char4 *qpGpu);

void ovSolveSingle(int id, char *codes, int *starts, int *lengths, int *scores,
                   sycl::int2 *hBus, sycl::nd_item<1> item_ct1, int gapOpen_,
                   int gapExtend_, int rows_, int rowsPadded_, int iters_,
                   int *scoresShr, int *hBusScrShr, int *hBusAffShr, int qpWidth_,
                   sycl::char4 *qpGpu);

void swSolveSingle(int id, char *codes, int *starts, int *lengths, int *scores,
                   sycl::int2 *hBus, sycl::nd_item<1> item_ct1, int gapOpen_,
                   int gapExtend_, int rows_, int rowsPadded_, int iters_,
                   int *scoresShr, int *hBusScrShr, int *hBusAffShr, int qpWidth_,
                   sycl::char4 *qpGpu);

// query profile
static QueryProfile *createQueryProfile(Chain *query, Scorer *scorer);

static void deleteQueryProfile(QueryProfile *queryProfile);

static QueryProfileGpu *createQueryProfileGpu(QueryProfile *queryProfile, sycl::queue &dev_q);

static void deleteQueryProfileGpu(QueryProfileGpu *queryProfileGpu, sycl::queue &dev_q);

//******************************************************************************

//******************************************************************************
// PUBLIC

//------------------------------------------------------------------------------
// CONSTRUCTOR, DESTRUCTOR

extern LongDatabase *longDatabaseCreate(Chain **database, int databaseLen,
                                        int minLen, int maxLen, int *cards, int cardsLen)
{
    return createDatabase(database, databaseLen, minLen, maxLen, cards, cardsLen);
}

extern void longDatabaseDelete(LongDatabase *longDatabase)
{
    deleteDatabase(longDatabase);
}

extern size_t longDatabaseGpuMemoryConsumption(Chain **database, int databaseLen,
                                               int minLen, int maxLen)
{

    int length = 0;
    long codesLen = 0;

    for (int i = 0; i < databaseLen; ++i)
    {

        const int n = chainGetLength(database[i]);

        if (n >= minLen && n < maxLen)
        {
            codesLen += n;
            length++;
        }
    }

    size_t lengthsSize = length * sizeof(int);
    size_t startsSize = length * sizeof(int);
    size_t codesSize = codesLen * sizeof(char);
    size_t indexesSize = length * sizeof(int);
    size_t scoresSize = length * sizeof(int);
    size_t hBusSize = codesLen * sizeof(sycl::int2);

    size_t memory = codesSize + startsSize + lengthsSize + indexesSize +
                    scoresSize + hBusSize;

    return memory;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// CPU KERNELS

extern void scoreLongDatabaseGpu(int *scores, int type, Chain *query,
                                 LongDatabase *longDatabase, Scorer *scorer, int *indexes, int indexesLen,
                                 int *cards, int cardsLen, Thread *thread)
{
    scoreDatabase(scores, type, &query, 1, longDatabase, scorer, indexes,
                  indexesLen, cards, cardsLen, thread);
}

extern void scoreLongDatabasesGpu(int *scores, int type, Chain **queries,
                                  int queriesLen, LongDatabase *longDatabase, Scorer *scorer, int *indexes,
                                  int indexesLen, int *cards, int cardsLen, Thread *thread)
{
    scoreDatabase(scores, type, queries, queriesLen, longDatabase, scorer,
                  indexes, indexesLen, cards, cardsLen, thread);
}

//------------------------------------------------------------------------------

//******************************************************************************

//******************************************************************************
// PRIVATE

//------------------------------------------------------------------------------
// CONSTRUCTOR, DESTRUCTOR

static LongDatabase *createDatabase(Chain **database, int databaseLen,
                                    int minLen, int maxLen, int *cards,
                                    int cardsLen)
try
{

    //**************************************************************************
    // FILTER DATABASE AND REMEBER ORDER

    int length = 0;

    for (int i = 0; i < databaseLen; ++i)
    {

        const int n = chainGetLength(database[i]);

        if (n >= minLen && n < maxLen)
        {
            length++;
        }
    }

    if (length == 0)
    {
        return NULL;
    }

    int *order = (int *)malloc(length * sizeof(int));

    for (int i = 0, j = 0; i < databaseLen; ++i)
    {

        const int n = chainGetLength(database[i]);

        if (n >= minLen && n < maxLen)
        {
            order[j++] = i;
        }
    }

    LOG("Long database length: %d", length);

    //**************************************************************************

    //**************************************************************************
    // CALCULATE DIMENSIONS

    long codesLen = 0;
    for (int i = 0; i < length; ++i)
    {
        codesLen += chainGetLength(database[order[i]]);
    }

    LOG("Long database cells: %ld", codesLen);

    //**************************************************************************

    //**************************************************************************
    // INIT STRUCTURES

    size_t lengthsSize = length * sizeof(int);
    int *lengths = (int *)malloc(lengthsSize);

    size_t startsSize = length * sizeof(int);
    int *starts = (int *)malloc(startsSize);

    size_t codesSize = codesLen * sizeof(char);
    char *codes = (char *)malloc(codesSize);

    //**************************************************************************

    //**************************************************************************
    // CREATE STRUCTURES

    long codesOff = 0;
    for (int i = 0; i < length; ++i)
    {

        Chain *chain = database[order[i]];
        int n = chainGetLength(chain);

        lengths[i] = n;
        starts[i] = codesOff;
        chainCopyCodes(chain, codes + codesOff);

        codesOff += n;
    }

    //**************************************************************************

    //**************************************************************************
    // CREATE DEFAULT INDEXES

    size_t indexesSize = length * sizeof(int);
    int *indexes = (int *)malloc(indexesSize);

    for (int i = 0; i < length; ++i)
    {
        indexes[i] = i;
    }

    //**************************************************************************

    //**************************************************************************
    // CREATE POSITION ARRAY

    int *positions = (int *)malloc(databaseLen * sizeof(int));

    for (int i = 0; i < databaseLen; ++i)
    {
        positions[i] = -1;
    }

    for (int i = 0; i < length; ++i)
    {
        positions[order[i]] = i;
    }

    //**************************************************************************

    //**************************************************************************
    // CREATE GPU DATABASES

    size_t gpuDatabasesSize = cardsLen * sizeof(GpuDatabase);
    GpuDatabase *gpuDatabases = (GpuDatabase *)malloc(gpuDatabasesSize);

    for (int i = 0; i < cardsLen; ++i)
    {

        int card = cards[i];

        sycl::queue dev_q((sycl::device::get_devices()[card]));

        char *codesGpu = sycl::malloc_device<char>(codesLen, dev_q);
        dev_q.memcpy(codesGpu, codes, codesSize).wait();

        int *startsGpu = sycl::malloc_device<int>(length, dev_q);
        dev_q.memcpy(startsGpu, starts, startsSize).wait();

        int *lengthsGpu = sycl::malloc_device<int>(length, dev_q);
        dev_q.memcpy(lengthsGpu, lengths, lengthsSize).wait();

        int *indexesGpu = sycl::malloc_device<int>(length, dev_q);
        dev_q.memcpy(indexesGpu, indexes, indexesSize).wait();

        // additional structures

        int *scoresGpu = sycl::malloc_shared<int>(length, dev_q);
        dev_q.memset(scoresGpu, 0, length * sizeof(int)).wait();

        sycl::int2 *hBusGpu = sycl::malloc_device<sycl::int2>(codesLen, dev_q);
        dev_q.memset(hBusGpu, 0, codesLen * sizeof(sycl::int2)).wait();

        gpuDatabases[i].card = card;
        gpuDatabases[i].codes = codesGpu;
        gpuDatabases[i].starts = startsGpu;
        gpuDatabases[i].lengths = lengthsGpu;
        gpuDatabases[i].indexes = indexesGpu;
        gpuDatabases[i].scores = scoresGpu;
        gpuDatabases[i].hBus = hBusGpu;

#ifdef DEBUG
        size_t memory = codesSize + startsSize + lengthsSize + indexesSize +
                        scoresSize + hBusSize;

        LOG("Long database using %.2lfMBs on card %d", memory / 1024.0 / 1024.0, card);
#endif
    }

    //**************************************************************************

    //**************************************************************************
    // CLEAN MEMORY

    free(codes);
    free(starts);
    free(lengths);

    //**************************************************************************

    size_t longDatabaseSize = sizeof(struct LongDatabase);
    LongDatabase *longDatabase = (LongDatabase *)malloc(longDatabaseSize);

    longDatabase->database = database;
    longDatabase->databaseLen = databaseLen;
    longDatabase->length = length;
    longDatabase->order = order;
    longDatabase->positions = positions;
    longDatabase->indexes = indexes;
    longDatabase->gpuDatabases = gpuDatabases;
    longDatabase->gpuDatabasesLen = cardsLen;

    return longDatabase;
}
catch (sycl::exception const &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void deleteDatabase(LongDatabase *database)
try
{

    if (database == NULL)
    {
        return;
    }

    for (int i = 0; i < database->gpuDatabasesLen; ++i)
    {

        GpuDatabase *gpuDatabase = &(database->gpuDatabases[i]);

        sycl::queue dev_q((sycl::device::get_devices()[gpuDatabase->card]));

        sycl::free(gpuDatabase->codes, dev_q);
        sycl::free(gpuDatabase->starts, dev_q);
        sycl::free(gpuDatabase->lengths, dev_q);
        sycl::free(gpuDatabase->indexes, dev_q);
        sycl::free(gpuDatabase->scores, dev_q);
        sycl::free(gpuDatabase->hBus, dev_q);
    }

    free(database->gpuDatabases);
    free(database->order);
    free(database->positions);
    free(database->indexes);

    free(database);
}
catch (sycl::exception const &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// SCORING

static void scoreDatabase(int *scores, int type, Chain **queries,
                          int queriesLen, LongDatabase *longDatabase, Scorer *scorer, int *indexes,
                          int indexesLen, int *cards, int cardsLen, Thread *thread)
{

    ASSERT(cardsLen > 0, "no GPUs available");

    Context *param = (Context *)malloc(sizeof(Context));

    param->scores = scores;
    param->type = type;
    param->queries = queries;
    param->queriesLen = queriesLen;
    param->longDatabase = longDatabase;
    param->scorer = scorer;
    param->indexes = indexes;
    param->indexesLen = indexesLen;
    param->cards = cards;
    param->cardsLen = cardsLen;

    if (thread == NULL)
    {
        scoreDatabaseGpuLongThread(param);
    }
    else
    {
        threadCreate(thread, scoreDatabaseGpuLongThread, (void *)param);
    }
}

static void *scoreDatabaseGpuLongThread(void *param)
{

    Context *context = (Context *)param;

    int *scores = context->scores;
    type = context->type;
    Chain **queries = context->queries;
    int queriesLen = context->queriesLen;
    LongDatabase *longDatabase = context->longDatabase;
    Scorer *scorer = context->scorer;
    int *indexes = context->indexes;
    int indexesLen = context->indexesLen;
    int *cards = context->cards;
    int cardsLen = context->cardsLen;

    if (longDatabase == NULL)
    {
        free(param);
        return NULL;
    }

    //**************************************************************************
    // CREATE NEW INDEXES ARRAY IF NEEDED

    int *newIndexes = NULL;
    int newIndexesLen = 0;

    int deleteIndexes;

    if (indexes != NULL)
    {

        // translate and filter indexes

        int databaseLen = longDatabase->databaseLen;
        int *positions = longDatabase->positions;

        newIndexes = (int *)malloc(indexesLen * sizeof(int));
        newIndexesLen = 0;

        for (int i = 0; i < indexesLen; ++i)
        {

            int idx = indexes[i];
            if (idx < 0 || idx > databaseLen || positions[idx] == -1)
            {
                continue;
            }

            newIndexes[newIndexesLen++] = positions[idx];
        }

        deleteIndexes = 1;
    }
    else
    {
        // load prebuilt defaults
        newIndexes = longDatabase->indexes;
        newIndexesLen = longDatabase->length;
        deleteIndexes = 0;
    }

    //**************************************************************************

    //**************************************************************************
    // CHOOSE SOLVING FUNCTION

    ScoringFunction function;
    // switch (type)
    // {
    // case SW_ALIGN:
    //     function = swSolve;
    //     break;
    // case NW_ALIGN:
    //     function = nwSolve;
    //     break;
    // case HW_ALIGN:
    //     function = hwSolve;
    //     break;
    // case OV_ALIGN:
    //     function = ovSolve;
    //     break;
    // default:
    //     ERROR("Wrong align type");
    // }

    //**************************************************************************

    //**************************************************************************
    // SCORE MULTITHREADED

    if (queriesLen < cardsLen)
    {
        scoreDatabaseMulti(scores, function, queries, queriesLen, longDatabase,
                           scorer, newIndexes, newIndexesLen, cards, cardsLen);
    }
    else
    {
        scoreDatabaseSingle(scores, function, queries, queriesLen, longDatabase,
                            scorer, newIndexes, newIndexesLen, cards, cardsLen);
    }

    //**************************************************************************

    //**************************************************************************
    // CLEAN MEMORY

    if (deleteIndexes)
    {
        free(newIndexes);
    }

    free(param);

    //**************************************************************************

    return NULL;
}

static void scoreDatabaseMulti(int *scores, ScoringFunction scoringFunction,
                               Chain **queries, int queriesLen, LongDatabase *longDatabase, Scorer *scorer,
                               int *indexes, int indexesLen, int *cards, int cardsLen)
{

    //**************************************************************************
    // CREATE QUERY PROFILES

    size_t profilesSize = queriesLen * sizeof(QueryProfile *);
    QueryProfile **profiles = (QueryProfile **)malloc(profilesSize);

    for (int i = 0; i < queriesLen; ++i)
    {
        profiles[i] = createQueryProfile(queries[i], scorer);
    }

    //**************************************************************************

    //**************************************************************************
    // CREATE BALANCING DATA

    Chain **database = longDatabase->database;
    int *order = longDatabase->order;

    size_t weightsSize = indexesLen * sizeof(int);
    int *weights = (int *)malloc(weightsSize);
    memset(weights, 0, weightsSize);

    for (int i = 0; i < indexesLen; ++i)
    {
        weights[i] += chainGetLength(database[order[indexes[i]]]);
    }

    //**************************************************************************

    //**************************************************************************
    // SCORE MULTICARDED

    int contextsLen = cardsLen * queriesLen;
    size_t contextsSize = contextsLen * sizeof(KernelContext);
    KernelContext *contexts = (KernelContext *)malloc(contextsSize);

    size_t tasksSize = contextsLen * sizeof(Thread);
    Thread *tasks = (Thread *)malloc(tasksSize);

    int databaseLen = longDatabase->databaseLen;

    int cardsChunk = cardsLen / queriesLen;
    int cardsAdd = cardsLen % queriesLen;
    int cardsOff = 0;

    int *idxChunksOff = (int *)malloc(cardsLen * sizeof(int));
    int *idxChunksLens = (int *)malloc(cardsLen * sizeof(int));
    int idxChunksLen = 0;

    int length = 0;

    for (int i = 0, k = 0; i < queriesLen; ++i)
    {

        int cCardsLen = cardsChunk + (i < cardsAdd);
        int *cCards = cards + cardsOff;
        cardsOff += cCardsLen;

        QueryProfile *queryProfile = profiles[i];

        int chunks = std::min(cCardsLen, indexesLen);
        if (chunks != idxChunksLen)
        {
            weightChunkArray(idxChunksOff, idxChunksLens, &idxChunksLen,
                             weights, indexesLen, chunks);
        }

        for (int j = 0; j < idxChunksLen; ++j, ++k)
        {

            contexts[k].scores = scores + i * databaseLen;
            contexts[k].scoringFunction = scoringFunction;
            contexts[k].queryProfile = queryProfile;
            contexts[k].longDatabase = longDatabase;
            contexts[k].scorer = scorer;
            contexts[k].indexes = indexes + idxChunksOff[j];
            contexts[k].indexesLen = idxChunksLens[j];
            contexts[k].card = cCards[j];

            threadCreate(&(tasks[k]), kernelThread, &(contexts[k]));
            length++;
        }
    }

    for (int i = 0; i < length; ++i)
    {
        threadJoin(tasks[i]);
    }

    free(tasks);
    free(contexts);

    //**************************************************************************

    //**************************************************************************
    // CLEAN MEMORY

    for (int i = 0; i < queriesLen; ++i)
    {
        deleteQueryProfile(profiles[i]);
    }

    free(profiles);
    free(weights);
    free(idxChunksOff);
    free(idxChunksLens);

    //**************************************************************************
}

static void scoreDatabaseSingle(int *scores, ScoringFunction scoringFunction,
                                Chain **queries, int queriesLen, LongDatabase *longDatabase, Scorer *scorer,
                                int *indexes, int indexesLen, int *cards, int cardsLen)
{

    //**************************************************************************
    // CREATE CONTEXTS

    size_t contextsSize = cardsLen * sizeof(KernelContext);
    KernelContexts *contexts = (KernelContexts *)malloc(contextsSize);

    for (int i = 0; i < cardsLen; ++i)
    {
        size_t size = queriesLen * sizeof(KernelContext);
        contexts[i].contexts = (KernelContext *)malloc(size);
        contexts[i].contextsLen = 0;
        contexts[i].cells = 0;
    }

    //**************************************************************************

    //**************************************************************************
    // SCORE MULTITHREADED

    size_t tasksSize = cardsLen * sizeof(Thread);
    Thread *tasks = (Thread *)malloc(tasksSize);

    int databaseLen = longDatabase->databaseLen;

    // balance tasks by round roobin, cardsLen is pretty small (CUDA cards)
    for (int i = 0; i < queriesLen; ++i)
    {

        int minIdx = 0;
        long long minVal = contexts[0].cells;
        for (int j = 1; j < cardsLen; ++j)
        {
            if (contexts[j].cells < minVal)
            {
                minVal = contexts[j].cells;
                minIdx = j;
            }
        }

        KernelContext context;
        context.scores = scores + i * databaseLen;
        context.scoringFunction = scoringFunction;
        context.queryProfile = NULL;
        context.query = queries[i];
        context.longDatabase = longDatabase;
        context.scorer = scorer;
        context.indexes = indexes;
        context.indexesLen = indexesLen;
        context.card = cards[minIdx];

        contexts[minIdx].contexts[contexts[minIdx].contextsLen++] = context;
        contexts[minIdx].cells += chainGetLength(queries[i]);
    }

    for (int i = 0; i < cardsLen; ++i)
    {
        threadCreate(&(tasks[i]), kernelsThread, &(contexts[i]));
    }

    for (int i = 0; i < cardsLen; ++i)
    {
        threadJoin(tasks[i]);
    }
    free(tasks);

    //**************************************************************************

    //**************************************************************************
    // CLEAN MEMORY

    for (int i = 0; i < cardsLen; ++i)
    {
        free(contexts[i].contexts);
    }
    free(contexts);

    //**************************************************************************
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// CPU KERNELS

static void *kernelsThread(void *param)
try
{

    KernelContexts *context = (KernelContexts *)param;

    KernelContext *contexts = context->contexts;
    int contextsLen = context->contextsLen;

    for (int i = 0; i < contextsLen; ++i)
    {

        Chain *query = contexts[i].query;
        Scorer *scorer = contexts[i].scorer;
        int card = contexts[i].card;

        int currentCard;

        contexts[i].queryProfile = createQueryProfile(query, scorer);

        kernelThread(&(contexts[i]));

        deleteQueryProfile(contexts[i].queryProfile);
    }

    return NULL;
}
catch (sycl::exception const &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void *kernelThread(void *param)
try
{

    KernelContext *context = (KernelContext *)param;

    int *scores = context->scores;
    ScoringFunction scoringFunction = context->scoringFunction;
    QueryProfile *queryProfile = context->queryProfile;
    LongDatabase *longDatabase = context->longDatabase;
    Scorer *scorer = context->scorer;
    int *indexes = context->indexes;
    int indexesLen = context->indexesLen;
    int card = context->card;

    //**************************************************************************
    // FIND DATABASE

    GpuDatabase *gpuDatabases = longDatabase->gpuDatabases;
    int gpuDatabasesLen = longDatabase->gpuDatabasesLen;

    GpuDatabase *gpuDatabase = NULL;

    for (int i = 0; i < gpuDatabasesLen; ++i)
    {
        if (gpuDatabases[i].card == card)
        {
            gpuDatabase = &(gpuDatabases[i]);
            break;
        }
    }

    ASSERT(gpuDatabase != NULL, "Long database not available on card %d", card);

    //**************************************************************************

    //**************************************************************************
    // CUDA SETUP

    int currentCard;

    sycl::queue dev_q((sycl::device::get_devices()[card]));

    int threads;
    int blocks;

    maxWorkGroups(card, BLOCKS, THREADS, 0, &blocks, &threads);

    //**************************************************************************

    //**************************************************************************
    // FIX INDEXES

    int deleteIndexes;
    int *indexesGpu;

    if (indexesLen == longDatabase->length)
    {
        indexes = longDatabase->indexes;
        indexesLen = longDatabase->length;
        indexesGpu = gpuDatabase->indexes;
        deleteIndexes = 0;
    }
    else
    {
        size_t indexesSize = indexesLen * sizeof(int);
        indexesGpu = sycl::malloc_device<int>(indexesLen, dev_q);
        dev_q.memcpy(indexesGpu, indexes, indexesSize).wait();
        deleteIndexes = 1;
    }

    //**************************************************************************

    //**************************************************************************
    // PREPARE GPU

    QueryProfileGpu *queryProfileGpu = createQueryProfileGpu(queryProfile, dev_q);

    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);
    int rows = queryProfile->length;
    int rowsGpu = queryProfile->height * 4;

    int qpWidth = queryProfile->width;
    sycl::char4 *qpGpu = queryProfileGpu->data;

    int iters = rowsGpu / (threads * 4) + (rowsGpu % (threads * 4) != 0);

    //**************************************************************************

    //**************************************************************************
    // SOLVE

    char *codesGpu = gpuDatabase->codes;
    int *startsGpu = gpuDatabase->starts;
    int *lengthsGpu = gpuDatabase->lengths;
    int *scoresGpu = gpuDatabase->scores;
    sycl::int2 *hBusGpu = gpuDatabase->hBus;

    dev_q
        .submit([&](sycl::handler &cgh)
                {

    sycl::accessor<int, 1, sycl::access_mode::read_write,
                   sycl::access::target::local>
        scoresShr_acc_ct1(sycl::range<1>(threads), cgh);
    sycl::accessor<int, 1, sycl::access_mode::read_write,
                   sycl::access::target::local>
        hBusScrShr_acc_ct1(sycl::range<1>(threads), cgh);
    sycl::accessor<int, 1, sycl::access_mode::read_write,
                   sycl::access::target::local>
        hBusAffShr_acc_ct1(sycl::range<1>(threads), cgh);

            if (type == SW_ALIGN) {
                cgh.parallel_for(sycl::nd_range<1>(blocks * threads, threads), [=](sycl::nd_item<1> item_ct1) {
                    swSolve(codesGpu, startsGpu, lengthsGpu, indexesGpu, scoresGpu,
                        hBusGpu, item_ct1, gapOpen, gapExtend,
                        rows, rowsGpu, indexesLen,
                        iters, scoresShr_acc_ct1.get_pointer(),
                        hBusScrShr_acc_ct1.get_pointer(),
                        hBusAffShr_acc_ct1.get_pointer(), qpWidth, qpGpu);   });
            }  else if (type == NW_ALIGN) {
                cgh.parallel_for(sycl::nd_range<1>(blocks * threads, threads),  [=](sycl::nd_item<1> item_ct1)  {
                    nwSolve(codesGpu, startsGpu, lengthsGpu, indexesGpu, scoresGpu,
                            hBusGpu, item_ct1, gapOpen, gapExtend,
                            rows, rowsGpu, indexesLen,
                            iters, scoresShr_acc_ct1.get_pointer(),
                            hBusScrShr_acc_ct1.get_pointer(),
                            hBusAffShr_acc_ct1.get_pointer(), qpWidth, qpGpu);   });
            }  else if (type == HW_ALIGN) {
                cgh.parallel_for(sycl::nd_range<1>(blocks * threads, threads),  [=](sycl::nd_item<1> item_ct1)  {
                    hwSolve(codesGpu, startsGpu, lengthsGpu, indexesGpu, scoresGpu,
                            hBusGpu, item_ct1, gapOpen, gapExtend,
                            rows, rowsGpu, indexesLen,
                            iters, scoresShr_acc_ct1.get_pointer(),
                            hBusScrShr_acc_ct1.get_pointer(),
                            hBusAffShr_acc_ct1.get_pointer(), qpWidth, qpGpu);   });
                         
             } else if (type == OV_ALIGN) {
                cgh.parallel_for(sycl::nd_range<1>(blocks * threads, threads), [=](sycl::nd_item<1> item_ct1) { 
                    ovSolve(codesGpu, startsGpu, lengthsGpu, indexesGpu, scoresGpu,
                            hBusGpu, item_ct1, gapOpen, gapExtend,
                            rows, rowsGpu, indexesLen,
                            iters, scoresShr_acc_ct1.get_pointer(),
                            hBusScrShr_acc_ct1.get_pointer(),
                            hBusAffShr_acc_ct1.get_pointer(), qpWidth, qpGpu);   });
    } })
        .wait();

    //**************************************************************************

    //**************************************************************************
    // SAVE RESULTS

    int length = longDatabase->length;

    size_t scoresSize = length * sizeof(int);
    int *order = longDatabase->order;
    for (int i = 0; i < indexesLen; ++i)
    {
        scores[order[indexes[i]]] = scoresGpu[indexes[i]];
    }

    // sycl::free(scoresGpu, dev_q);

    //**************************************************************************

    //**************************************************************************
    // CLEAN MEMORY

    deleteQueryProfileGpu(queryProfileGpu, dev_q);

    if (deleteIndexes)
    {
        sycl::free(indexesGpu, dev_q);
    }

    //**************************************************************************

    return NULL;
}
catch (sycl::exception const &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// GPU KERNELS

void hwSolve(char *codes, int *starts, int *lengths, int *indexes, int *scores,
             sycl::int2 *hBus, sycl::nd_item<1> item_ct1, int gapOpen_,
             int gapExtend_, int rows_, int rowsPadded_, int length_,
             int iters_, int *scoresShr, int *hBusScrShr, int *hBusAffShr, int qpWidth_,
             sycl::char4 *qpGpu)
{

    const int groupId = item_ct1.get_group(0);
    const int groupRangeId = item_ct1.get_group_range(0);

    for (int i = groupId; i < length_; i += groupRangeId)
    {
        hwSolveSingle(indexes[i], codes, starts, lengths, scores, hBus, item_ct1,
                      gapOpen_, gapExtend_, rows_, rowsPadded_, iters_, scoresShr,
                      hBusScrShr, hBusAffShr, qpWidth_, qpGpu);
    }
}

void nwSolve(char *codes, int *starts, int *lengths, int *indexes, int *scores,
             sycl::int2 *hBus, sycl::nd_item<1> item_ct1, int gapOpen_,
             int gapExtend_, int rows_, int rowsPadded_, int length_,
             int iters_, int *scoresShr, int *hBusScrShr, int *hBusAffShr, int qpWidth_,
             sycl::char4 *qpGpu)
{

    const int groupId = item_ct1.get_group(0);
    const int groupRangeId = item_ct1.get_group_range(0);

    for (int i = groupId; i < length_; i += groupRangeId)
    {
        nwSolveSingle(indexes[i], codes, starts, lengths, scores, hBus, item_ct1,
                      gapOpen_, gapExtend_, rows_, rowsPadded_, iters_, scoresShr,
                      hBusScrShr, hBusAffShr, qpWidth_, qpGpu);
    }
}

void ovSolve(char *codes, int *starts, int *lengths, int *indexes, int *scores,
             sycl::int2 *hBus, sycl::nd_item<1> item_ct1, int gapOpen_,
             int gapExtend_, int rows_, int rowsPadded_, int length_,
             int iters_, int *scoresShr, int *hBusScrShr, int *hBusAffShr, int qpWidth_,
             sycl::char4 *qpGpu)
{

    const int groupId = item_ct1.get_group(0);
    const int groupRangeId = item_ct1.get_group_range(0);

    for (int i = groupId; i < length_; i += groupRangeId)
    {
        ovSolveSingle(indexes[i], codes, starts, lengths, scores, hBus, item_ct1,
                      gapOpen_, gapExtend_, rows_, rowsPadded_, iters_, scoresShr,
                      hBusScrShr, hBusAffShr, qpWidth_, qpGpu);
    }
}

void swSolve(char *codes, int *starts, int *lengths, int *indexes, int *scores,
             sycl::int2 *hBus, sycl::nd_item<1> item_ct1, int gapOpen_,
             int gapExtend_, int rows_, int rowsPadded_, int length_,
             int iters_, int *scoresShr, int *hBusScrShr, int *hBusAffShr, int qpWidth_,
             sycl::char4 *qpGpu)
{
    const int groupId = item_ct1.get_group(0);
    const int groupRangeId = item_ct1.get_group_range(0);

    for (int i = groupId; i < length_; i += groupRangeId)
    {
        swSolveSingle(indexes[i], codes, starts, lengths, scores, hBus, item_ct1,
                      gapOpen_, gapExtend_, rows_, rowsPadded_, iters_, scoresShr,
                      hBusScrShr, hBusAffShr, qpWidth_, qpGpu);
    }
}

static int gap(int index, int gapOpen_, int gapExtend_)
{
    return (-gapOpen_ - index * gapExtend_) * (index >= 0);
}

void hwSolveSingle(int id, char *codes, int *starts, int *lengths, int *scores,
                   sycl::int2 *hBus, sycl::nd_item<1> item_ct1, int gapOpen_,
                   int gapExtend_, int rows_, int rowsPadded_, int iters_,
                   int *scoresShr, int *hBusScrShr, int *hBusAffShr, int qpWidth_,
                   sycl::char4 *qpGpu)
{
    const int localRangeId = item_ct1.get_local_range(0);
    const int localId = item_ct1.get_local_id(0);

    int off = starts[id];
    int cols = lengths[id];

    int score = SCORE_MIN;

    int width = cols * iters_ + 2 * (localRangeId - 1);
    int col = -localId;
    int row = localId * 4;
    int iter = 0;

    Atom atom;
    atom.mch = gap(row - 1, gapOpen_, gapExtend_);
    atom.lScr = sycl::int4(
        gap(row, gapOpen_, gapExtend_), gap(row + 1, gapOpen_, gapExtend_),
        gap(row + 2, gapOpen_, gapExtend_), gap(row + 3, gapOpen_, gapExtend_));
    atom.lAff = INT4_SCORE_MIN;

    hBusScrShr[localId] = 0;
    hBusAffShr[localId] = SCORE_MIN;

    for (int i = 0; i < width; ++i)
    {

        int del;
        int valid = col >= 0 && row < rowsPadded_;

        if (valid)
        {

            if (iter != 0 && localId == 0)
            {
                atom.up = hBus[off + col];
            }
            else
            {
                atom.up.x() = hBusScrShr[localId];
                atom.up.y() = hBusAffShr[localId];
            }

            char code = codes[off + col];

            sycl::char4 rowScores = qpGpu[(row >> 2) * qpWidth_ + sycl::min(static_cast<int>(code), qpWidth_ - 1)];

            del = sycl::max((atom.up.x() - gapOpen_),
                            (atom.up.y() - gapExtend_));
            int ins = sycl::max((atom.lScr.x() - gapOpen_),
                                (atom.lAff.x() - gapExtend_));
            int mch = atom.mch + rowScores.x();

            atom.rScr.x() = MAX3(mch, del, ins);
            atom.rAff.x() = ins;

            del = sycl::max((atom.rScr.x() - gapOpen_), (del - gapExtend_));
            ins = sycl::max((atom.lScr.y() - gapOpen_),
                            (atom.lAff.y() - gapExtend_));
            mch = atom.lScr.x() + rowScores.y();

            atom.rScr.y() = MAX3(mch, del, ins);
            atom.rAff.y() = ins;

            del = sycl::max((atom.rScr.y() - gapOpen_), (del - gapExtend_));
            ins = sycl::max((atom.lScr.z() - gapOpen_),
                            (atom.lAff.z() - gapExtend_));
            mch = atom.lScr.y() + rowScores.z();

            atom.rScr.z() = MAX3(mch, del, ins);
            atom.rAff.z() = ins;

            del = sycl::max((atom.rScr.z() - gapOpen_), (del - gapExtend_));
            ins = sycl::max((atom.lScr.w() - gapOpen_),
                            (atom.lAff.w() - gapExtend_));
            mch = atom.lScr.z() + rowScores.w();

            atom.rScr.w() = MAX3(mch, del, ins);
            atom.rAff.w() = ins;

            if (row + 0 == rows_ - 1)
                score = sycl::max(score, atom.rScr.x());
            if (row + 1 == rows_ - 1)
                score = sycl::max(score, atom.rScr.y());
            if (row + 2 == rows_ - 1)
                score = sycl::max(score, atom.rScr.z());
            if (row + 3 == rows_ - 1)
                score = sycl::max(score, atom.rScr.w());

            atom.mch = atom.up.x();
            atom.lScr = atom.rScr;
            atom.lAff = atom.rAff;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        if (valid)
        {
            if (iter < iters_ - 1 &&
                localId == localRangeId - 1)
            {
                hBus[off + col] = sycl::int2(atom.rScr.w(), del);
            }
            else
            {
                hBusScrShr[localId + 1] = atom.rScr.w();
                hBusAffShr[localId + 1] = del;
            }
        }

        col++;

        if (col == cols)
        {

            col = 0;
            row += localRangeId * 4;
            iter++;

            atom.mch = gap(row - 1, gapOpen_, gapExtend_);
            atom.lScr = sycl::int4(gap(row, gapOpen_, gapExtend_),
                                   gap(row + 1, gapOpen_, gapExtend_),
                                   gap(row + 2, gapOpen_, gapExtend_),
                                   gap(row + 3, gapOpen_, gapExtend_));
            ;
            atom.lAff = INT4_SCORE_MIN;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    // write all scores
    scoresShr[localId] = score;

    item_ct1.barrier(sycl::access::fence_space::local_space);

    // gather scores
    if (localId == 0)
    {

        for (int i = 1; i < localRangeId; ++i)
        {
            score = sycl::max(score, scoresShr[i]);
        }

        scores[id] = score;
    }
}

void nwSolveSingle(int id, char *codes, int *starts, int *lengths, int *scores,
                   sycl::int2 *hBus, sycl::nd_item<1> item_ct1, int gapOpen_,
                   int gapExtend_, int rows_, int rowsPadded_, int iters_,
                   int *scoresShr, int *hBusScrShr, int *hBusAffShr, int qpWidth_,
                   sycl::char4 *qpGpu)
{

    const int localRangeId = item_ct1.get_local_range(0);
    const int localId = item_ct1.get_local_id(0);

    int off = starts[id];
    int cols = lengths[id];

    int score = SCORE_MIN;

    int width = cols * iters_ + 2 * (localRangeId - 1);
    int col = -localId;
    int row = localId * 4;
    int iter = 0;

    Atom atom;
    atom.mch = gap(row - 1, gapOpen_, gapExtend_);
    atom.lScr = sycl::int4(
        gap(row, gapOpen_, gapExtend_), gap(row + 1, gapOpen_, gapExtend_),
        gap(row + 2, gapOpen_, gapExtend_), gap(row + 3, gapOpen_, gapExtend_));
    atom.lAff = INT4_SCORE_MIN;

    for (int i = 0; i < width; ++i)
    {

        int del;
        int valid = col >= 0 && row < rowsPadded_;

        if (valid)
        {

            if (localId == 0)
            {
                if (iter == 0)
                {
                    atom.up.x() = gap(col, gapOpen_, gapExtend_);
                    atom.up.y() = SCORE_MIN;
                }
                else
                {
                    atom.up = hBus[off + col];
                }
            }
            else
            {
                atom.up.x() = hBusScrShr[localId];
                atom.up.y() = hBusAffShr[localId];
            }

            char code = codes[off + col];

            sycl::char4 rowScores = qpGpu[(row >> 2) * qpWidth_ + sycl::min(static_cast<int>(code), qpWidth_ - 1)];

            del = sycl::max((atom.up.x() - gapOpen_),
                            (atom.up.y() - gapExtend_));
            int ins = sycl::max((atom.lScr.x() - gapOpen_),
                                (atom.lAff.x() - gapExtend_));
            int mch = atom.mch + rowScores.x();

            atom.rScr.x() = MAX3(mch, del, ins);
            atom.rAff.x() = ins;

            del = sycl::max((atom.rScr.x() - gapOpen_), (del - gapExtend_));
            ins = sycl::max((atom.lScr.y() - gapOpen_),
                            (atom.lAff.y() - gapExtend_));
            mch = atom.lScr.x() + rowScores.y();

            atom.rScr.y() = MAX3(mch, del, ins);
            atom.rAff.y() = ins;

            del = sycl::max((atom.rScr.y() - gapOpen_), (del - gapExtend_));
            ins = sycl::max((atom.lScr.z() - gapOpen_),
                            (atom.lAff.z() - gapExtend_));
            mch = atom.lScr.y() + rowScores.z();

            atom.rScr.z() = MAX3(mch, del, ins);
            atom.rAff.z() = ins;

            del = sycl::max((atom.rScr.z() - gapOpen_), (del - gapExtend_));
            ins = sycl::max((atom.lScr.w() - gapOpen_),
                            (atom.lAff.w() - gapExtend_));
            mch = atom.lScr.z() + rowScores.w();

            atom.rScr.w() = MAX3(mch, del, ins);
            atom.rAff.w() = ins;

            atom.mch = atom.up.x();
            atom.lScr = atom.rScr;
            atom.lAff = atom.rAff;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        if (valid)
        {
            if (iter < iters_ - 1 &&
                localId == localRangeId - 1)
            {
                hBus[off + col] = sycl::int2(atom.rScr.w(), del);
            }
            else
            {
                hBusScrShr[localId + 1] = atom.rScr.w();
                hBusAffShr[localId + 1] = del;
            }
        }

        col++;

        if (col == cols)
        {

            if (row + 0 == rows_ - 1)
                score = sycl::max(score, atom.lScr.x());
            if (row + 1 == rows_ - 1)
                score = sycl::max(score, atom.lScr.y());
            if (row + 2 == rows_ - 1)
                score = sycl::max(score, atom.lScr.z());
            if (row + 3 == rows_ - 1)
                score = sycl::max(score, atom.lScr.w());

            col = 0;
            row += localRangeId * 4;
            iter++;

            atom.mch = gap(row - 1, gapOpen_, gapExtend_);
            atom.lScr = sycl::int4(gap(row, gapOpen_, gapExtend_),
                                   gap(row + 1, gapOpen_, gapExtend_),
                                   gap(row + 2, gapOpen_, gapExtend_),
                                   gap(row + 3, gapOpen_, gapExtend_));
            ;
            atom.lAff = INT4_SCORE_MIN;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    // write all scores
    scoresShr[localId] = score;

    item_ct1.barrier(sycl::access::fence_space::local_space);

    // gather scores
    if (localId == 0)
    {

        for (int i = 1; i < localRangeId; ++i)
        {
            score = sycl::max(score, scoresShr[i]);
        }

        scores[id] = score;
    }
}

void ovSolveSingle(int id, char *codes, int *starts, int *lengths, int *scores,
                   sycl::int2 *hBus, sycl::nd_item<1> item_ct1, int gapOpen_,
                   int gapExtend_, int rows_, int rowsPadded_, int iters_,
                   int *scoresShr, int *hBusScrShr, int *hBusAffShr, int qpWidth_,
                   sycl::char4 *qpGpu)
{
    const int localRangeId = item_ct1.get_local_range(0);
    const int localId = item_ct1.get_local_id(0);

    int off = starts[id];
    int cols = lengths[id];

    int score = SCORE_MIN;

    int width = cols * iters_ + 2 * (localRangeId - 1);
    int col = -localId;
    int row = localId * 4;
    int iter = 0;

    Atom atom;
    atom.mch = 0;
    atom.lScr = INT4_ZERO;
    ;
    atom.lAff = INT4_SCORE_MIN;

    hBusScrShr[localId] = 0;
    hBusAffShr[localId] = SCORE_MIN;

    for (int i = 0; i < width; ++i)
    {

        int del;
        int valid = col >= 0 && row < rowsPadded_;

        if (valid)
        {

            if (iter != 0 && localId == 0)
            {
                atom.up = hBus[off + col];
            }
            else
            {
                atom.up.x() = hBusScrShr[localId];
                atom.up.y() = hBusAffShr[localId];
            }

            char code = codes[off + col];

            sycl::char4 rowScores = qpGpu[(row >> 2) * qpWidth_ + sycl::min(static_cast<int>(code), qpWidth_ - 1)];

            del = sycl::max((atom.up.x() - gapOpen_),
                            (atom.up.y() - gapExtend_));
            int ins = sycl::max((atom.lScr.x() - gapOpen_),
                                (atom.lAff.x() - gapExtend_));
            int mch = atom.mch + rowScores.x();

            atom.rScr.x() = MAX3(mch, del, ins);
            atom.rAff.x() = ins;

            del = sycl::max((atom.rScr.x() - gapOpen_), (del - gapExtend_));
            ins = sycl::max((atom.lScr.y() - gapOpen_),
                            (atom.lAff.y() - gapExtend_));
            mch = atom.lScr.x() + rowScores.y();

            atom.rScr.y() = MAX3(mch, del, ins);
            atom.rAff.y() = ins;

            del = sycl::max((atom.rScr.y() - gapOpen_), (del - gapExtend_));
            ins = sycl::max((atom.lScr.z() - gapOpen_),
                            (atom.lAff.z() - gapExtend_));
            mch = atom.lScr.y() + rowScores.z();

            atom.rScr.z() = MAX3(mch, del, ins);
            atom.rAff.z() = ins;

            del = sycl::max((atom.rScr.z() - gapOpen_), (del - gapExtend_));
            ins = sycl::max((atom.lScr.w() - gapOpen_),
                            (atom.lAff.w() - gapExtend_));
            mch = atom.lScr.z() + rowScores.w();

            atom.rScr.w() = MAX3(mch, del, ins);
            atom.rAff.w() = ins;

            if (row + 0 == rows_ - 1)
                score = sycl::max(score, atom.rScr.x());
            if (row + 1 == rows_ - 1)
                score = sycl::max(score, atom.rScr.y());
            if (row + 2 == rows_ - 1)
                score = sycl::max(score, atom.rScr.z());
            if (row + 3 == rows_ - 1)
                score = sycl::max(score, atom.rScr.w());

            atom.mch = atom.up.x();
            atom.lScr = atom.rScr;
            atom.lAff = atom.rAff;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        if (valid)
        {
            if (iter < iters_ - 1 &&
                localId == localRangeId - 1)
            {
                hBus[off + col] = sycl::int2(atom.rScr.w(), del);
            }
            else
            {
                hBusScrShr[localId + 1] = atom.rScr.w();
                hBusAffShr[localId + 1] = del;
            }
        }

        col++;

        if (col == cols)
        {

            if (row < rows_)
            {
                score = sycl::max(score, atom.lScr.x());
                score = sycl::max(score, atom.lScr.y());
                score = sycl::max(score, atom.lScr.z());
                score = sycl::max(score, atom.lScr.w());
            }

            col = 0;
            row += localRangeId * 4;
            iter++;

            atom.mch = 0;
            atom.lScr = INT4_ZERO;
            ;
            atom.lAff = INT4_SCORE_MIN;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    // write all scores
    scoresShr[localId] = score;

    item_ct1.barrier(sycl::access::fence_space::local_space);

    // gather scores
    if (localId == 0)
    {

        for (int i = 1; i < localRangeId; ++i)
        {
            score = sycl::max(score, scoresShr[i]);
        }

        scores[id] = score;
    }
}

void swSolveSingle(int id, char *codes, int *starts, int *lengths, int *scores,
                   sycl::int2 *hBus, sycl::nd_item<1> item_ct1, int gapOpen_,
                   int gapExtend_, int rows_, int rowsPadded_, int iters_,
                   int *scoresShr, int *hBusScrShr, int *hBusAffShr, int qpWidth_,
                   sycl::char4 *qpGpu)
{
    const int localRangeId = item_ct1.get_local_range(0);
    const int localId = item_ct1.get_local_id(0);

    int off = starts[id];
    int cols = lengths[id];

    int score = 0;

    int width = cols * iters_ + 2 * (localRangeId - 1);
    int col = -localId;
    int row = localId * 4;
    int iter = 0;

    Atom atom;
    atom.mch = 0;
    atom.lScr = INT4_ZERO;
    ;
    atom.lAff = INT4_SCORE_MIN;

    hBusScrShr[localId] = 0;
    hBusAffShr[localId] = SCORE_MIN;

    for (int i = 0; i < width; ++i)
    {

        int del;
        int valid = col >= 0 && row < rowsPadded_;

        if (valid)
        {

            if (iter != 0 && localId == 0)
            {
                atom.up = hBus[off + col];
            }
            else
            {
                atom.up.x() = hBusScrShr[localId];
                atom.up.y() = hBusAffShr[localId];
            }

            char code = codes[off + col];

            sycl::char4 rowScores = qpGpu[(row >> 2) * qpWidth_ + sycl::min(static_cast<int>(code), qpWidth_ - 1)];

            del = sycl::max((atom.up.x() - gapOpen_),
                            (atom.up.y() - gapExtend_));
            int ins = sycl::max((atom.lScr.x() - gapOpen_),
                                (atom.lAff.x() - gapExtend_));
            int mch = atom.mch + rowScores.x();

            atom.rScr.x() = MAX4(0, mch, del, ins);
            atom.rAff.x() = ins;

            del = sycl::max((atom.rScr.x() - gapOpen_), (del - gapExtend_));
            ins = sycl::max((atom.lScr.y() - gapOpen_),
                            (atom.lAff.y() - gapExtend_));
            mch = atom.lScr.x() + rowScores.y();

            atom.rScr.y() = MAX4(0, mch, del, ins);
            atom.rAff.y() = ins;

            del = sycl::max((atom.rScr.y() - gapOpen_), (del - gapExtend_));
            ins = sycl::max((atom.lScr.z() - gapOpen_),
                            (atom.lAff.z() - gapExtend_));
            mch = atom.lScr.y() + rowScores.z();

            atom.rScr.z() = MAX4(0, mch, del, ins);
            atom.rAff.z() = ins;

            del = sycl::max((atom.rScr.z() - gapOpen_), (del - gapExtend_));
            ins = sycl::max((atom.lScr.w() - gapOpen_),
                            (atom.lAff.w() - gapExtend_));
            mch = atom.lScr.z() + rowScores.w();

            atom.rScr.w() = MAX4(0, mch, del, ins);
            atom.rAff.w() = ins;

            score = sycl::max(score, atom.rScr.x());
            score = sycl::max(score, atom.rScr.y());
            score = sycl::max(score, atom.rScr.z());
            score = sycl::max(score, atom.rScr.w());

            atom.mch = atom.up.x();
            atom.lScr = atom.rScr;
            atom.lAff = atom.rAff;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        if (valid)
        {
            if (iter < iters_ - 1 &&
                localId == localRangeId - 1)
            {
                hBus[off + col] = sycl::int2(atom.rScr.w(), del);
            }
            else
            {
                hBusScrShr[localId + 1] = atom.rScr.w();
                hBusAffShr[localId + 1] = del;
            }
        }

        col++;

        if (col == cols)
        {

            col = 0;
            row += localRangeId * 4;
            iter++;

            atom.mch = 0;
            atom.lScr = INT4_ZERO;
            ;
            atom.lAff = INT4_SCORE_MIN;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    // write all scores
    scoresShr[localId] = score;

    item_ct1.barrier(sycl::access::fence_space::local_space);

    // gather scores
    if (localId == 0)
    {

        for (int i = 1; i < localRangeId; ++i)
        {
            score = sycl::max(score, scoresShr[i]);
        }

        scores[id] = score;
    }
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// QUERY PROFILE

static QueryProfile *createQueryProfile(Chain *query, Scorer *scorer)
{

    int rows = chainGetLength(query);
    int rowsGpu = rows + (8 - rows % 8) % 8;

    int width = scorerGetMaxCode(scorer) + 1;
    int height = rowsGpu / 4;

    char *row = (char *)malloc(rows * sizeof(char));
    chainCopyCodes(query, row);

    size_t size = width * height * sizeof(sycl::char4);
    sycl::char4 *data = (sycl::char4 *)malloc(size);
    memset(data, 0, size);
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width - 1; ++j)
        {
            sycl::char4 scr;
            scr.x() = i * 4 + 0 >= rows ? 0 : scorerScore(scorer, row[i * 4 + 0], j);
            scr.y() = i * 4 + 1 >= rows ? 0 : scorerScore(scorer, row[i * 4 + 1], j);
            scr.z() = i * 4 + 2 >= rows ? 0 : scorerScore(scorer, row[i * 4 + 2], j);
            scr.w() = i * 4 + 3 >= rows ? 0 : scorerScore(scorer, row[i * 4 + 3], j);
            data[i * width + j] = scr;
        }
    }

    free(row);

    QueryProfile *queryProfile = (QueryProfile *)malloc(sizeof(QueryProfile));
    queryProfile->data = data;
    queryProfile->width = width;
    queryProfile->height = height;
    queryProfile->length = rows;
    queryProfile->size = size;

    return queryProfile;
}

static void deleteQueryProfile(QueryProfile *queryProfile)
{
    free(queryProfile->data);
    free(queryProfile);
}

static QueryProfileGpu *createQueryProfileGpu(QueryProfile *queryProfile, sycl::queue &dev_q)
try
{

    int width = queryProfile->width;
    int height = queryProfile->height;

    size_t size = queryProfile->size;
    sycl::char4 *dataGpu = sycl::malloc_device<sycl::char4>(height * width, dev_q);
    dev_q.memcpy(dataGpu, queryProfile->data, height * width * sizeof(sycl::char4)).wait();

    size_t queryProfileGpuSize = sizeof(QueryProfileGpu);
    QueryProfileGpu *queryProfileGpu =
        (QueryProfileGpu *)malloc(queryProfileGpuSize);

    queryProfileGpu->data = dataGpu;
    return queryProfileGpu;
}
catch (sycl::exception const &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void deleteQueryProfileGpu(QueryProfileGpu *queryProfileGpu, sycl::queue &dev_q)
try
{
    sycl::free(queryProfileGpu->data, dev_q);

    free(queryProfileGpu);
}
catch (sycl::exception const &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

//------------------------------------------------------------------------------
//******************************************************************************
