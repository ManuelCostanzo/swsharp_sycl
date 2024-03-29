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

Contact SW#-SYCL authors by mcostanzo@lidi.info.unlp.edu.ar,
erucci@lidi.info.unlp.edu.ar
*/

#ifdef __CUDACC__

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

#define THREADS 64
#define BLOCKS 240

#define MAX_THREADS 1024

#define INT4_ZERO make_int4(0, 0, 0, 0)
#define INT4_SCORE_MIN make_int4(SCORE_MIN, SCORE_MIN, SCORE_MIN, SCORE_MIN)

typedef struct GpuDatabase {
  int card;
  char *codes;
  int *starts;
  int *lengths;
  int *indexes;
  int *scores;
  int2 *hBus;
} GpuDatabase;

struct LongDatabase {
  Chain **database;
  int databaseLen;
  int length;
  int *order;
  int *positions;
  int *indexes;
  GpuDatabase *gpuDatabases;
  int gpuDatabasesLen;
};

typedef struct Context {
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

typedef struct QueryProfile {
  int height;
  int width;
  int length;
  char4 *data;
  size_t size;
} QueryProfile;

typedef struct QueryProfileGpu {
  cudaArray *data;
} QueryProfileGpu;

typedef void (*ScoringFunction)(char *, int *, int *, int *, int *, int2 *);

typedef struct KernelContext {
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

typedef struct KernelContexts {
  KernelContext *contexts;
  int contextsLen;
  long long cells;
} KernelContexts;

typedef struct Atom {
  int mch;
  int2 up;
  int4 lScr;
  int4 lAff;
  int4 rScr;
  int4 rAff;
} Atom;

static __constant__ int gapOpen_;
static __constant__ int gapExtend_;

static __constant__ int rows_;
static __constant__ int rowsPadded_;
static __constant__ int length_;
static __constant__ int iters_;

texture<char4, 2, cudaReadModeElementType> qpTexture;

//******************************************************************************
// PUBLIC

extern LongDatabase *longDatabaseCreate(Chain **database, int databaseLen,
                                        int minLen, int maxLen, int *cards,
                                        int cardsLen);

extern void longDatabaseDelete(LongDatabase *longDatabase);

extern void scoreLongDatabaseGpu(int *scores, int type, Chain *query,
                                 LongDatabase *longDatabase, Scorer *scorer,
                                 int *indexes, int indexesLen, int *cards,
                                 int cardsLen, Thread *thread);

extern void scoreLongDatabasesGpu(int *scores, int type, Chain **queries,
                                  int queriesLen, LongDatabase *longDatabase,
                                  Scorer *scorer, int *indexes, int indexesLen,
                                  int *cards, int cardsLen, Thread *thread);

//******************************************************************************

//******************************************************************************
// PRIVATE

// constructor
static LongDatabase *createDatabase(Chain **database, int databaseLen,
                                    int minLen, int maxLen, int *cards,
                                    int cardsLen);

// destructor
static void deleteDatabase(LongDatabase *database);

// scoring
static void scoreDatabase(int *scores, int type, Chain **queries,
                          int queriesLen, LongDatabase *longDatabase,
                          Scorer *scorer, int *indexes, int indexesLen,
                          int *cards, int cardsLen, Thread *thread);

static void *scoreDatabaseThread(void *param);

static void scoreDatabaseMulti(int *scores, ScoringFunction scoringFunction,
                               Chain **queries, int queriesLen,
                               LongDatabase *longDatabase, Scorer *scorer,
                               int *indexes, int indexesLen, int *cards,
                               int cardsLen);

static void scoreDatabaseSingle(int *scores, ScoringFunction scoringFunction,
                                Chain **queries, int queriesLen,
                                LongDatabase *longDatabase, Scorer *scorer,
                                int *indexes, int indexesLen, int *cards,
                                int cardsLen);

// cpu kernels
static void *kernelThread(void *param);

static void *kernelsThread(void *param);

// gpu kernels
__global__ __launch_bounds__(MAX_THREADS) void hwSolve(char *codes, int *starts,
                                                       int *lengths,
                                                       int *indexes,
                                                       int *scores, int2 *hBus);

__global__ __launch_bounds__(MAX_THREADS) void nwSolve(char *codes, int *starts,
                                                       int *lengths,
                                                       int *indexes,
                                                       int *scores, int2 *hBus);

__global__ __launch_bounds__(MAX_THREADS) void ovSolve(char *codes, int *starts,
                                                       int *lengths,
                                                       int *indexes,
                                                       int *scores, int2 *hBus);

__global__ __launch_bounds__(MAX_THREADS) void swSolve(char *codes, int *starts,
                                                       int *lengths,
                                                       int *indexes,
                                                       int *scores, int2 *hBus);

__device__ static int gap(int index);

__device__ void hwSolveSingle(int id, char *codes, int *starts, int *lengths,
                              int *scores, int2 *hBus);

__device__ void nwSolveSingle(int id, char *codes, int *starts, int *lengths,
                              int *scores, int2 *hBus);

__device__ void ovSolveSingle(int id, char *codes, int *starts, int *lengths,
                              int *scores, int2 *hBus);

__device__ void swSolveSingle(int id, char *codes, int *starts, int *lengths,
                              int *scores, int2 *hBus);

// query profile
static QueryProfile *createQueryProfile(Chain *query, Scorer *scorer);

static void deleteQueryProfile(QueryProfile *queryProfile);

static QueryProfileGpu *createQueryProfileGpu(QueryProfile *queryProfile);

static void deleteQueryProfileGpu(QueryProfileGpu *queryProfileGpu);

//******************************************************************************

//******************************************************************************
// PUBLIC

//------------------------------------------------------------------------------
// CONSTRUCTOR, DESTRUCTOR

extern LongDatabase *longDatabaseCreate(Chain **database, int databaseLen,
                                        int minLen, int maxLen, int *cards,
                                        int cardsLen) {
  return createDatabase(database, databaseLen, minLen, maxLen, cards, cardsLen);
}

extern void longDatabaseDelete(LongDatabase *longDatabase) {
  deleteDatabase(longDatabase);
}

extern size_t longDatabaseGpuMemoryConsumption(Chain **database,
                                               int databaseLen, int minLen,
                                               int maxLen) {

  int length = 0;
  long codesLen = 0;

  for (int i = 0; i < databaseLen; ++i) {

    const int n = chainGetLength(database[i]);

    if (n >= minLen && n < maxLen) {
      codesLen += n;
      length++;
    }
  }

  size_t lengthsSize = length * sizeof(int);
  size_t startsSize = length * sizeof(int);
  size_t codesSize = codesLen * sizeof(char);
  size_t indexesSize = length * sizeof(int);
  size_t scoresSize = length * sizeof(int);
  size_t hBusSize = codesLen * sizeof(int2);

  size_t memory = codesSize + startsSize + lengthsSize + indexesSize +
                  scoresSize + hBusSize;

  return memory;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// CPU KERNELS

extern void scoreLongDatabaseGpu(int *scores, int type, Chain *query,
                                 LongDatabase *longDatabase, Scorer *scorer,
                                 int *indexes, int indexesLen, int *cards,
                                 int cardsLen, Thread *thread) {
  scoreDatabase(scores, type, &query, 1, longDatabase, scorer, indexes,
                indexesLen, cards, cardsLen, thread);
}

extern void scoreLongDatabasesGpu(int *scores, int type, Chain **queries,
                                  int queriesLen, LongDatabase *longDatabase,
                                  Scorer *scorer, int *indexes, int indexesLen,
                                  int *cards, int cardsLen, Thread *thread) {
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
                                    int cardsLen) {

  //**************************************************************************
  // FILTER DATABASE AND REMEBER ORDER

  int length = 0;

  for (int i = 0; i < databaseLen; ++i) {

    const int n = chainGetLength(database[i]);

    if (n >= minLen && n < maxLen) {
      length++;
    }
  }

  if (length == 0) {
    return NULL;
  }

  int *order = (int *)malloc(length * sizeof(int));

  for (int i = 0, j = 0; i < databaseLen; ++i) {

    const int n = chainGetLength(database[i]);

    if (n >= minLen && n < maxLen) {
      order[j++] = i;
    }
  }

  LOG("Long database length: %d", length);

  //**************************************************************************

  //**************************************************************************
  // CALCULATE DIMENSIONS

  long codesLen = 0;
  for (int i = 0; i < length; ++i) {
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
  for (int i = 0; i < length; ++i) {

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

  for (int i = 0; i < length; ++i) {
    indexes[i] = i;
  }

  //**************************************************************************

  //**************************************************************************
  // CREATE POSITION ARRAY

  int *positions = (int *)malloc(databaseLen * sizeof(int));

  for (int i = 0; i < databaseLen; ++i) {
    positions[i] = -1;
  }

  for (int i = 0; i < length; ++i) {
    positions[order[i]] = i;
  }

  //**************************************************************************

  //**************************************************************************
  // CREATE GPU DATABASES

  size_t gpuDatabasesSize = cardsLen * sizeof(GpuDatabase);
  GpuDatabase *gpuDatabases = (GpuDatabase *)malloc(gpuDatabasesSize);

  for (int i = 0; i < cardsLen; ++i) {

    int card = cards[i];
    CUDA_SAFE_CALL(cudaSetDevice(card));

    char *codesGpu;
    CUDA_SAFE_CALL(cudaMalloc(&codesGpu, codesSize));
    CUDA_SAFE_CALL(cudaMemcpy(codesGpu, codes, codesSize, TO_GPU));

    int *startsGpu;
    CUDA_SAFE_CALL(cudaMalloc(&startsGpu, startsSize));
    CUDA_SAFE_CALL(cudaMemcpy(startsGpu, starts, startsSize, TO_GPU));

    int *lengthsGpu;
    CUDA_SAFE_CALL(cudaMalloc(&lengthsGpu, lengthsSize));
    CUDA_SAFE_CALL(cudaMemcpy(lengthsGpu, lengths, lengthsSize, TO_GPU));

    int *indexesGpu;
    CUDA_SAFE_CALL(cudaMalloc(&indexesGpu, indexesSize));
    CUDA_SAFE_CALL(cudaMemcpy(indexesGpu, indexes, indexesSize, TO_GPU));

    // additional structures

    size_t scoresSize = length * sizeof(int);
    int *scoresGpu;
    CUDA_SAFE_CALL(cudaMalloc(&scoresGpu, scoresSize));

    int2 *hBusGpu;
    size_t hBusSize = codesLen * sizeof(int2);
    CUDA_SAFE_CALL(cudaMalloc(&hBusGpu, hBusSize));

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

    LOG("Long database using %.2lfMBs on card %d", memory / 1024.0 / 1024.0,
        card);
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

static void deleteDatabase(LongDatabase *database) {

  if (database == NULL) {
    return;
  }

  for (int i = 0; i < database->gpuDatabasesLen; ++i) {

    GpuDatabase *gpuDatabase = &(database->gpuDatabases[i]);

    CUDA_SAFE_CALL(cudaSetDevice(gpuDatabase->card));

    CUDA_SAFE_CALL(cudaFree(gpuDatabase->codes));
    CUDA_SAFE_CALL(cudaFree(gpuDatabase->starts));
    CUDA_SAFE_CALL(cudaFree(gpuDatabase->lengths));
    CUDA_SAFE_CALL(cudaFree(gpuDatabase->indexes));
    CUDA_SAFE_CALL(cudaFree(gpuDatabase->scores));
    CUDA_SAFE_CALL(cudaFree(gpuDatabase->hBus));
  }

  free(database->gpuDatabases);
  free(database->order);
  free(database->positions);
  free(database->indexes);

  free(database);
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// SCORING

static void scoreDatabase(int *scores, int type, Chain **queries,
                          int queriesLen, LongDatabase *longDatabase,
                          Scorer *scorer, int *indexes, int indexesLen,
                          int *cards, int cardsLen, Thread *thread) {

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

  if (thread == NULL) {
    scoreDatabaseThread(param);
  } else {
    threadCreate(thread, scoreDatabaseThread, (void *)param);
  }
}

static void *scoreDatabaseThread(void *param) {

  Context *context = (Context *)param;

  int *scores = context->scores;
  int type = context->type;
  Chain **queries = context->queries;
  int queriesLen = context->queriesLen;
  LongDatabase *longDatabase = context->longDatabase;
  Scorer *scorer = context->scorer;
  int *indexes = context->indexes;
  int indexesLen = context->indexesLen;
  int *cards = context->cards;
  int cardsLen = context->cardsLen;

  if (longDatabase == NULL) {
    free(param);
    return NULL;
  }

  //**************************************************************************
  // CREATE NEW INDEXES ARRAY IF NEEDED

  int *newIndexes = NULL;
  int newIndexesLen = 0;

  int deleteIndexes;

  if (indexes != NULL) {

    // translate and filter indexes

    int databaseLen = longDatabase->databaseLen;
    int *positions = longDatabase->positions;

    newIndexes = (int *)malloc(indexesLen * sizeof(int));
    newIndexesLen = 0;

    for (int i = 0; i < indexesLen; ++i) {

      int idx = indexes[i];
      if (idx < 0 || idx > databaseLen || positions[idx] == -1) {
        continue;
      }

      newIndexes[newIndexesLen++] = positions[idx];
    }

    deleteIndexes = 1;
  } else {
    // load prebuilt defaults
    newIndexes = longDatabase->indexes;
    newIndexesLen = longDatabase->length;
    deleteIndexes = 0;
  }

  //**************************************************************************

  //**************************************************************************
  // CHOOSE SOLVING FUNCTION

  ScoringFunction function;
  switch (type) {
  case SW_ALIGN:
    function = swSolve;
    break;
  case NW_ALIGN:
    function = nwSolve;
    break;
  case HW_ALIGN:
    function = hwSolve;
    break;
  case OV_ALIGN:
    function = ovSolve;
    break;
  default:
    ERROR("Wrong align type");
  }

  //**************************************************************************

  //**************************************************************************
  // SCORE MULTITHREADED

  if (queriesLen < cardsLen) {
    scoreDatabaseMulti(scores, function, queries, queriesLen, longDatabase,
                       scorer, newIndexes, newIndexesLen, cards, cardsLen);
  } else {
    scoreDatabaseSingle(scores, function, queries, queriesLen, longDatabase,
                        scorer, newIndexes, newIndexesLen, cards, cardsLen);
  }

  //**************************************************************************

  //**************************************************************************
  // CLEAN MEMORY

  if (deleteIndexes) {
    free(newIndexes);
  }

  free(param);

  //**************************************************************************

  return NULL;
}

static void scoreDatabaseMulti(int *scores, ScoringFunction scoringFunction,
                               Chain **queries, int queriesLen,
                               LongDatabase *longDatabase, Scorer *scorer,
                               int *indexes, int indexesLen, int *cards,
                               int cardsLen) {

  //**************************************************************************
  // CREATE QUERY PROFILES

  size_t profilesSize = queriesLen * sizeof(QueryProfile *);
  QueryProfile **profiles = (QueryProfile **)malloc(profilesSize);

  for (int i = 0; i < queriesLen; ++i) {
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

  for (int i = 0; i < indexesLen; ++i) {
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

  for (int i = 0, k = 0; i < queriesLen; ++i) {

    int cCardsLen = cardsChunk + (i < cardsAdd);
    int *cCards = cards + cardsOff;
    cardsOff += cCardsLen;

    QueryProfile *queryProfile = profiles[i];

    int chunks = min(cCardsLen, indexesLen);
    if (chunks != idxChunksLen) {
      weightChunkArray(idxChunksOff, idxChunksLens, &idxChunksLen, weights,
                       indexesLen, chunks);
    }

    for (int j = 0; j < idxChunksLen; ++j, ++k) {

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

  for (int i = 0; i < length; ++i) {
    threadJoin(tasks[i]);
  }

  free(tasks);
  free(contexts);

  //**************************************************************************

  //**************************************************************************
  // CLEAN MEMORY

  for (int i = 0; i < queriesLen; ++i) {
    deleteQueryProfile(profiles[i]);
  }

  free(profiles);
  free(weights);
  free(idxChunksOff);
  free(idxChunksLens);

  //**************************************************************************
}

static void scoreDatabaseSingle(int *scores, ScoringFunction scoringFunction,
                                Chain **queries, int queriesLen,
                                LongDatabase *longDatabase, Scorer *scorer,
                                int *indexes, int indexesLen, int *cards,
                                int cardsLen) {

  //**************************************************************************
  // CREATE CONTEXTS

  size_t contextsSize = cardsLen * sizeof(KernelContext);
  KernelContexts *contexts = (KernelContexts *)malloc(contextsSize);

  for (int i = 0; i < cardsLen; ++i) {
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
  for (int i = 0; i < queriesLen; ++i) {

    int minIdx = 0;
    long long minVal = contexts[0].cells;
    for (int j = 1; j < cardsLen; ++j) {
      if (contexts[j].cells < minVal) {
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

  for (int i = 0; i < cardsLen; ++i) {
    threadCreate(&(tasks[i]), kernelsThread, &(contexts[i]));
  }

  for (int i = 0; i < cardsLen; ++i) {
    threadJoin(tasks[i]);
  }
  free(tasks);

  //**************************************************************************

  //**************************************************************************
  // CLEAN MEMORY

  for (int i = 0; i < cardsLen; ++i) {
    free(contexts[i].contexts);
  }
  free(contexts);

  //**************************************************************************
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// CPU KERNELS

static void *kernelsThread(void *param) {

  KernelContexts *context = (KernelContexts *)param;

  KernelContext *contexts = context->contexts;
  int contextsLen = context->contextsLen;

  for (int i = 0; i < contextsLen; ++i) {

    Chain *query = contexts[i].query;
    Scorer *scorer = contexts[i].scorer;
    int card = contexts[i].card;

    int currentCard;
    CUDA_SAFE_CALL(cudaGetDevice(&currentCard));
    if (currentCard != card) {
      CUDA_SAFE_CALL(cudaSetDevice(card));
    }

    contexts[i].queryProfile = createQueryProfile(query, scorer);

    kernelThread(&(contexts[i]));

    deleteQueryProfile(contexts[i].queryProfile);
  }

  return NULL;
}

static void *kernelThread(void *param) {

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

  for (int i = 0; i < gpuDatabasesLen; ++i) {
    if (gpuDatabases[i].card == card) {
      gpuDatabase = &(gpuDatabases[i]);
      break;
    }
  }

  ASSERT(gpuDatabase != NULL, "Long database not available on card %d", card);

  //**************************************************************************

  //**************************************************************************
  // CUDA SETUP

  int currentCard;
  CUDA_SAFE_CALL(cudaGetDevice(&currentCard));
  if (currentCard != card) {
    CUDA_SAFE_CALL(cudaSetDevice(card));
  }

  int threads;
  int blocks;

  maxWorkGroups(card, BLOCKS, THREADS, 0, &blocks, &threads);

  //**************************************************************************

  //**************************************************************************
  // FIX INDEXES

  int deleteIndexes;
  int *indexesGpu;

  if (indexesLen == longDatabase->length) {
    indexes = longDatabase->indexes;
    indexesLen = longDatabase->length;
    indexesGpu = gpuDatabase->indexes;
    deleteIndexes = 0;
  } else {
    size_t indexesSize = indexesLen * sizeof(int);
    CUDA_SAFE_CALL(cudaMalloc(&indexesGpu, indexesSize));
    CUDA_SAFE_CALL(cudaMemcpy(indexesGpu, indexes, indexesSize, TO_GPU));
    deleteIndexes = 1;
  }

  //**************************************************************************

  //**************************************************************************
  // PREPARE GPU

  QueryProfileGpu *queryProfileGpu = createQueryProfileGpu(queryProfile);

  int gapOpen = scorerGetGapOpen(scorer);
  int gapExtend = scorerGetGapExtend(scorer);
  int rows = queryProfile->length;
  int rowsGpu = queryProfile->height * 4;
  int iters = rowsGpu / (threads * 4) + (rowsGpu % (threads * 4) != 0);

  CUDA_SAFE_CALL(cudaMemcpyToSymbol(gapOpen_, &gapOpen, sizeof(int)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(gapExtend_, &gapExtend, sizeof(int)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(rows_, &rows, sizeof(int)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(rowsPadded_, &rowsGpu, sizeof(int)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(length_, &indexesLen, sizeof(int)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(iters_, &iters, sizeof(int)));

  //**************************************************************************

  //**************************************************************************
  // SOLVE

  char *codesGpu = gpuDatabase->codes;
  int *startsGpu = gpuDatabase->starts;
  int *lengthsGpu = gpuDatabase->lengths;
  int *scoresGpu = gpuDatabase->scores;
  int2 *hBusGpu = gpuDatabase->hBus;

  scoringFunction<<<blocks, threads>>>(codesGpu, startsGpu, lengthsGpu,
                                       indexesGpu, scoresGpu, hBusGpu);

  //**************************************************************************

  //**************************************************************************
  // SAVE RESULTS

  int length = longDatabase->length;

  size_t scoresSize = length * sizeof(int);
  int *scoresCpu = (int *)malloc(scoresSize);

  CUDA_SAFE_CALL(cudaMemcpy(scoresCpu, scoresGpu, scoresSize, FROM_GPU));

  int *order = longDatabase->order;

  for (int i = 0; i < indexesLen; ++i) {
    scores[order[indexes[i]]] = scoresCpu[indexes[i]];
  }

  free(scoresCpu);

  //**************************************************************************

  //**************************************************************************
  // CLEAN MEMORY

  deleteQueryProfileGpu(queryProfileGpu);

  if (deleteIndexes) {
    CUDA_SAFE_CALL(cudaFree(indexesGpu));
  }

  //**************************************************************************

  return NULL;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// GPU KERNELS

__global__ __launch_bounds__(MAX_THREADS) void hwSolve(char *codes, int *starts,
                                                       int *lengths,
                                                       int *indexes,
                                                       int *scores,
                                                       int2 *hBus) {

  for (int i = blockIdx.x; i < length_; i += gridDim.x) {
    hwSolveSingle(indexes[i], codes, starts, lengths, scores, hBus);
  }
}

__global__ __launch_bounds__(MAX_THREADS) void nwSolve(char *codes, int *starts,
                                                       int *lengths,
                                                       int *indexes,
                                                       int *scores,
                                                       int2 *hBus) {

  for (int i = blockIdx.x; i < length_; i += gridDim.x) {
    nwSolveSingle(indexes[i], codes, starts, lengths, scores, hBus);
  }
}

__global__ __launch_bounds__(MAX_THREADS) void ovSolve(char *codes, int *starts,
                                                       int *lengths,
                                                       int *indexes,
                                                       int *scores,
                                                       int2 *hBus) {

  for (int i = blockIdx.x; i < length_; i += gridDim.x) {
    ovSolveSingle(indexes[i], codes, starts, lengths, scores, hBus);
  }
}

__global__ __launch_bounds__(MAX_THREADS) void swSolve(char *codes, int *starts,
                                                       int *lengths,
                                                       int *indexes,
                                                       int *scores,
                                                       int2 *hBus) {

  for (int i = blockIdx.x; i < length_; i += gridDim.x) {
    swSolveSingle(indexes[i], codes, starts, lengths, scores, hBus);
  }
}

__device__ static int gap(int index) {
  return (-gapOpen_ - index * gapExtend_) * (index >= 0);
}

__device__ void hwSolveSingle(int id, char *codes, int *starts, int *lengths,
                              int *scores, int2 *hBus) {

  __shared__ int scoresShr[MAX_THREADS];

  __shared__ int hBusScrShr[MAX_THREADS + 1];
  __shared__ int hBusAffShr[MAX_THREADS + 1];

  int off = starts[id];
  int cols = lengths[id];

  int score = SCORE_MIN;

  int width = cols * iters_ + 2 * (blockDim.x - 1);
  int col = -threadIdx.x;
  int row = threadIdx.x * 4;
  int iter = 0;

  Atom atom;
  atom.mch = gap(row - 1);
  atom.lScr = make_int4(gap(row), gap(row + 1), gap(row + 2), gap(row + 3));
  atom.lAff = INT4_SCORE_MIN;

  hBusScrShr[threadIdx.x] = 0;
  hBusAffShr[threadIdx.x] = SCORE_MIN;

  for (int i = 0; i < width; ++i) {

    int del;
    int valid = col >= 0 && row < rowsPadded_;

    if (valid) {

      if (iter != 0 && threadIdx.x == 0) {
        atom.up = hBus[off + col];
      } else {
        atom.up.x = hBusScrShr[threadIdx.x];
        atom.up.y = hBusAffShr[threadIdx.x];
      }

      char code = codes[off + col];
      char4 rowScores = tex2D(qpTexture, code, row >> 2);

      del = max(atom.up.x - gapOpen_, atom.up.y - gapExtend_);
      int ins = max(atom.lScr.x - gapOpen_, atom.lAff.x - gapExtend_);
      int mch = atom.mch + rowScores.x;

      atom.rScr.x = MAX3(mch, del, ins);
      atom.rAff.x = ins;

      del = max(atom.rScr.x - gapOpen_, del - gapExtend_);
      ins = max(atom.lScr.y - gapOpen_, atom.lAff.y - gapExtend_);
      mch = atom.lScr.x + rowScores.y;

      atom.rScr.y = MAX3(mch, del, ins);
      atom.rAff.y = ins;

      del = max(atom.rScr.y - gapOpen_, del - gapExtend_);
      ins = max(atom.lScr.z - gapOpen_, atom.lAff.z - gapExtend_);
      mch = atom.lScr.y + rowScores.z;

      atom.rScr.z = MAX3(mch, del, ins);
      atom.rAff.z = ins;

      del = max(atom.rScr.z - gapOpen_, del - gapExtend_);
      ins = max(atom.lScr.w - gapOpen_, atom.lAff.w - gapExtend_);
      mch = atom.lScr.z + rowScores.w;

      atom.rScr.w = MAX3(mch, del, ins);
      atom.rAff.w = ins;

      if (row + 0 == rows_ - 1)
        score = max(score, atom.rScr.x);
      if (row + 1 == rows_ - 1)
        score = max(score, atom.rScr.y);
      if (row + 2 == rows_ - 1)
        score = max(score, atom.rScr.z);
      if (row + 3 == rows_ - 1)
        score = max(score, atom.rScr.w);

      atom.mch = atom.up.x;
      VEC4_ASSIGN(atom.lScr, atom.rScr);
      VEC4_ASSIGN(atom.lAff, atom.rAff);
    }

    __syncthreads();

    if (valid) {
      if (iter < iters_ - 1 && threadIdx.x == blockDim.x - 1) {
        VEC2_ASSIGN(hBus[off + col], make_int2(atom.rScr.w, del));
      } else {
        hBusScrShr[threadIdx.x + 1] = atom.rScr.w;
        hBusAffShr[threadIdx.x + 1] = del;
      }
    }

    col++;

    if (col == cols) {

      col = 0;
      row += blockDim.x * 4;
      iter++;

      atom.mch = gap(row - 1);
      atom.lScr = make_int4(gap(row), gap(row + 1), gap(row + 2), gap(row + 3));
      ;
      atom.lAff = INT4_SCORE_MIN;
    }

    __syncthreads();
  }

  // write all scores
  scoresShr[threadIdx.x] = score;
  __syncthreads();

  // gather scores
  if (threadIdx.x == 0) {

    for (int i = 1; i < blockDim.x; ++i) {
      score = max(score, scoresShr[i]);
    }

    scores[id] = score;
  }
}

__device__ void nwSolveSingle(int id, char *codes, int *starts, int *lengths,
                              int *scores, int2 *hBus) {

  __shared__ int scoresShr[MAX_THREADS];

  __shared__ int hBusScrShr[MAX_THREADS + 1];
  __shared__ int hBusAffShr[MAX_THREADS + 1];

  int off = starts[id];
  int cols = lengths[id];

  int score = SCORE_MIN;

  int width = cols * iters_ + 2 * (blockDim.x - 1);
  int col = -threadIdx.x;
  int row = threadIdx.x * 4;
  int iter = 0;

  Atom atom;
  atom.mch = gap(row - 1);
  atom.lScr = make_int4(gap(row), gap(row + 1), gap(row + 2), gap(row + 3));
  atom.lAff = INT4_SCORE_MIN;

  for (int i = 0; i < width; ++i) {

    int del;
    int valid = col >= 0 && row < rowsPadded_;

    if (valid) {

      if (threadIdx.x == 0) {
        if (iter == 0) {
          atom.up.x = gap(col);
          atom.up.y = SCORE_MIN;
        } else {
          atom.up = hBus[off + col];
        }
      } else {
        atom.up.x = hBusScrShr[threadIdx.x];
        atom.up.y = hBusAffShr[threadIdx.x];
      }

      char code = codes[off + col];
      char4 rowScores = tex2D(qpTexture, code, row >> 2);

      del = max(atom.up.x - gapOpen_, atom.up.y - gapExtend_);
      int ins = max(atom.lScr.x - gapOpen_, atom.lAff.x - gapExtend_);
      int mch = atom.mch + rowScores.x;

      atom.rScr.x = MAX3(mch, del, ins);
      atom.rAff.x = ins;

      del = max(atom.rScr.x - gapOpen_, del - gapExtend_);
      ins = max(atom.lScr.y - gapOpen_, atom.lAff.y - gapExtend_);
      mch = atom.lScr.x + rowScores.y;

      atom.rScr.y = MAX3(mch, del, ins);
      atom.rAff.y = ins;

      del = max(atom.rScr.y - gapOpen_, del - gapExtend_);
      ins = max(atom.lScr.z - gapOpen_, atom.lAff.z - gapExtend_);
      mch = atom.lScr.y + rowScores.z;

      atom.rScr.z = MAX3(mch, del, ins);
      atom.rAff.z = ins;

      del = max(atom.rScr.z - gapOpen_, del - gapExtend_);
      ins = max(atom.lScr.w - gapOpen_, atom.lAff.w - gapExtend_);
      mch = atom.lScr.z + rowScores.w;

      atom.rScr.w = MAX3(mch, del, ins);
      atom.rAff.w = ins;

      atom.mch = atom.up.x;
      VEC4_ASSIGN(atom.lScr, atom.rScr);
      VEC4_ASSIGN(atom.lAff, atom.rAff);
    }

    __syncthreads();

    if (valid) {
      if (iter < iters_ - 1 && threadIdx.x == blockDim.x - 1) {
        VEC2_ASSIGN(hBus[off + col], make_int2(atom.rScr.w, del));
      } else {
        hBusScrShr[threadIdx.x + 1] = atom.rScr.w;
        hBusAffShr[threadIdx.x + 1] = del;
      }
    }

    col++;

    if (col == cols) {

      if (row + 0 == rows_ - 1)
        score = max(score, atom.lScr.x);
      if (row + 1 == rows_ - 1)
        score = max(score, atom.lScr.y);
      if (row + 2 == rows_ - 1)
        score = max(score, atom.lScr.z);
      if (row + 3 == rows_ - 1)
        score = max(score, atom.lScr.w);

      col = 0;
      row += blockDim.x * 4;
      iter++;

      atom.mch = gap(row - 1);
      atom.lScr = make_int4(gap(row), gap(row + 1), gap(row + 2), gap(row + 3));
      ;
      atom.lAff = INT4_SCORE_MIN;
    }

    __syncthreads();
  }

  // write all scores
  scoresShr[threadIdx.x] = score;
  __syncthreads();

  // gather scores
  if (threadIdx.x == 0) {

    for (int i = 1; i < blockDim.x; ++i) {
      score = max(score, scoresShr[i]);
    }

    scores[id] = score;
  }
}

__device__ void ovSolveSingle(int id, char *codes, int *starts, int *lengths,
                              int *scores, int2 *hBus) {

  __shared__ int scoresShr[MAX_THREADS];

  __shared__ int hBusScrShr[MAX_THREADS + 1];
  __shared__ int hBusAffShr[MAX_THREADS + 1];

  int off = starts[id];
  int cols = lengths[id];

  int score = SCORE_MIN;

  int width = cols * iters_ + 2 * (blockDim.x - 1);
  int col = -threadIdx.x;
  int row = threadIdx.x * 4;
  int iter = 0;

  Atom atom;
  atom.mch = 0;
  atom.lScr = INT4_ZERO;
  atom.lAff = INT4_SCORE_MIN;

  hBusScrShr[threadIdx.x] = 0;
  hBusAffShr[threadIdx.x] = SCORE_MIN;

  for (int i = 0; i < width; ++i) {

    int del;
    int valid = col >= 0 && row < rowsPadded_;

    if (valid) {

      if (iter != 0 && threadIdx.x == 0) {
        atom.up = hBus[off + col];
      } else {
        atom.up.x = hBusScrShr[threadIdx.x];
        atom.up.y = hBusAffShr[threadIdx.x];
      }

      char code = codes[off + col];
      char4 rowScores = tex2D(qpTexture, code, row >> 2);

      del = max(atom.up.x - gapOpen_, atom.up.y - gapExtend_);
      int ins = max(atom.lScr.x - gapOpen_, atom.lAff.x - gapExtend_);
      int mch = atom.mch + rowScores.x;

      atom.rScr.x = MAX3(mch, del, ins);
      atom.rAff.x = ins;

      del = max(atom.rScr.x - gapOpen_, del - gapExtend_);
      ins = max(atom.lScr.y - gapOpen_, atom.lAff.y - gapExtend_);
      mch = atom.lScr.x + rowScores.y;

      atom.rScr.y = MAX3(mch, del, ins);
      atom.rAff.y = ins;

      del = max(atom.rScr.y - gapOpen_, del - gapExtend_);
      ins = max(atom.lScr.z - gapOpen_, atom.lAff.z - gapExtend_);
      mch = atom.lScr.y + rowScores.z;

      atom.rScr.z = MAX3(mch, del, ins);
      atom.rAff.z = ins;

      del = max(atom.rScr.z - gapOpen_, del - gapExtend_);
      ins = max(atom.lScr.w - gapOpen_, atom.lAff.w - gapExtend_);
      mch = atom.lScr.z + rowScores.w;

      atom.rScr.w = MAX3(mch, del, ins);
      atom.rAff.w = ins;

      if (row + 0 == rows_ - 1)
        score = max(score, atom.rScr.x);
      if (row + 1 == rows_ - 1)
        score = max(score, atom.rScr.y);
      if (row + 2 == rows_ - 1)
        score = max(score, atom.rScr.z);
      if (row + 3 == rows_ - 1)
        score = max(score, atom.rScr.w);

      atom.mch = atom.up.x;
      VEC4_ASSIGN(atom.lScr, atom.rScr);
      VEC4_ASSIGN(atom.lAff, atom.rAff);
    }

    __syncthreads();

    if (valid) {
      if (iter < iters_ - 1 && threadIdx.x == blockDim.x - 1) {
        VEC2_ASSIGN(hBus[off + col], make_int2(atom.rScr.w, del));
      } else {
        hBusScrShr[threadIdx.x + 1] = atom.rScr.w;
        hBusAffShr[threadIdx.x + 1] = del;
      }
    }

    col++;

    if (col == cols) {

      if (row < rows_) {
        score = max(score, atom.lScr.x);
        score = max(score, atom.lScr.y);
        score = max(score, atom.lScr.z);
        score = max(score, atom.lScr.w);
      }

      col = 0;
      row += blockDim.x * 4;
      iter++;

      atom.mch = 0;
      atom.lScr = INT4_ZERO;
      atom.lAff = INT4_SCORE_MIN;
    }

    __syncthreads();
  }

  // write all scores
  scoresShr[threadIdx.x] = score;
  __syncthreads();

  // gather scores
  if (threadIdx.x == 0) {

    for (int i = 1; i < blockDim.x; ++i) {
      score = max(score, scoresShr[i]);
    }

    scores[id] = score;
  }
}

__device__ void swSolveSingle(int id, char *codes, int *starts, int *lengths,
                              int *scores, int2 *hBus) {

  __shared__ int scoresShr[MAX_THREADS + 1];

  __shared__ int hBusScrShr[MAX_THREADS + 1];
  __shared__ int hBusAffShr[MAX_THREADS + 1];

  int off = starts[id];
  int cols = lengths[id];

  int score = 0;

  int width = cols * iters_ + 2 * (blockDim.x - 1);
  int col = -threadIdx.x;
  int row = threadIdx.x * 4;
  int iter = 0;

  Atom atom;
  atom.mch = 0;
  atom.lScr = INT4_ZERO;
  atom.lAff = INT4_SCORE_MIN;

  hBusScrShr[threadIdx.x] = 0;
  hBusAffShr[threadIdx.x] = SCORE_MIN;

  for (int i = 0; i < width; ++i) {

    int del;
    int valid = col >= 0 && row < rowsPadded_;

    if (valid) {

      if (iter != 0 && threadIdx.x == 0) {
        atom.up = hBus[off + col];
      } else {
        atom.up.x = hBusScrShr[threadIdx.x];
        atom.up.y = hBusAffShr[threadIdx.x];
      }

      char code = codes[off + col];
      char4 rowScores = tex2D(qpTexture, code, row >> 2);

      del = max(atom.up.x - gapOpen_, atom.up.y - gapExtend_);
      int ins = max(atom.lScr.x - gapOpen_, atom.lAff.x - gapExtend_);
      int mch = atom.mch + rowScores.x;

      atom.rScr.x = MAX4(0, mch, del, ins);
      atom.rAff.x = ins;

      del = max(atom.rScr.x - gapOpen_, del - gapExtend_);
      ins = max(atom.lScr.y - gapOpen_, atom.lAff.y - gapExtend_);
      mch = atom.lScr.x + rowScores.y;

      atom.rScr.y = MAX4(0, mch, del, ins);
      atom.rAff.y = ins;

      del = max(atom.rScr.y - gapOpen_, del - gapExtend_);
      ins = max(atom.lScr.z - gapOpen_, atom.lAff.z - gapExtend_);
      mch = atom.lScr.y + rowScores.z;

      atom.rScr.z = MAX4(0, mch, del, ins);
      atom.rAff.z = ins;

      del = max(atom.rScr.z - gapOpen_, del - gapExtend_);
      ins = max(atom.lScr.w - gapOpen_, atom.lAff.w - gapExtend_);
      mch = atom.lScr.z + rowScores.w;

      atom.rScr.w = MAX4(0, mch, del, ins);
      atom.rAff.w = ins;

      score = max(score, atom.rScr.x);
      score = max(score, atom.rScr.y);
      score = max(score, atom.rScr.z);
      score = max(score, atom.rScr.w);

      atom.mch = atom.up.x;
      VEC4_ASSIGN(atom.lScr, atom.rScr);
      VEC4_ASSIGN(atom.lAff, atom.rAff);
    }

    __syncthreads();

    if (valid) {
      if (iter < iters_ - 1 && threadIdx.x == blockDim.x - 1) {
        VEC2_ASSIGN(hBus[off + col], make_int2(atom.rScr.w, del));
      } else {
        hBusScrShr[threadIdx.x + 1] = atom.rScr.w;
        hBusAffShr[threadIdx.x + 1] = del;
      }
    }

    col++;

    if (col == cols) {

      col = 0;
      row += blockDim.x * 4;
      iter++;

      atom.mch = 0;
      atom.lScr = INT4_ZERO;
      atom.lAff = INT4_SCORE_MIN;
    }

    __syncthreads();
  }

  // write all scores
  scoresShr[threadIdx.x] = score;
  __syncthreads();

  // gather scores
  if (threadIdx.x == 0) {

    for (int i = 1; i < blockDim.x; ++i) {
      score = max(score, scoresShr[i]);
    }

    scores[id] = score;
  }
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// QUERY PROFILE

static QueryProfile *createQueryProfile(Chain *query, Scorer *scorer) {

  int rows = chainGetLength(query);
  int rowsGpu = rows + (8 - rows % 8) % 8;

  int width = scorerGetMaxCode(scorer) + 1;
  int height = rowsGpu / 4;

  char *row = (char *)malloc(rows * sizeof(char));
  chainCopyCodes(query, row);

  size_t size = width * height * sizeof(char4);
  char4 *data = (char4 *)malloc(size);
  memset(data, 0, size);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width - 1; ++j) {
      char4 scr;
      scr.x = i * 4 + 0 >= rows ? 0 : scorerScore(scorer, row[i * 4 + 0], j);
      scr.y = i * 4 + 1 >= rows ? 0 : scorerScore(scorer, row[i * 4 + 1], j);
      scr.z = i * 4 + 2 >= rows ? 0 : scorerScore(scorer, row[i * 4 + 2], j);
      scr.w = i * 4 + 3 >= rows ? 0 : scorerScore(scorer, row[i * 4 + 3], j);
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

static void deleteQueryProfile(QueryProfile *queryProfile) {
  free(queryProfile->data);
  free(queryProfile);
}

static QueryProfileGpu *createQueryProfileGpu(QueryProfile *queryProfile) {

  int width = queryProfile->width;
  int height = queryProfile->height;

  size_t size = queryProfile->size;
  char4 *data = queryProfile->data;
  cudaArray *dataGpu;

  CUDA_SAFE_CALL(
      cudaMallocArray(&dataGpu, &qpTexture.channelDesc, width, height));
  CUDA_SAFE_CALL(cudaMemcpyToArray(dataGpu, 0, 0, data, size, TO_GPU));
  CUDA_SAFE_CALL(cudaBindTextureToArray(qpTexture, dataGpu));
  qpTexture.addressMode[0] = cudaAddressModeClamp;
  qpTexture.addressMode[1] = cudaAddressModeClamp;
  qpTexture.filterMode = cudaFilterModePoint;
  qpTexture.normalized = false;

  size_t queryProfileGpuSize = sizeof(QueryProfileGpu);
  QueryProfileGpu *queryProfileGpu =
      (QueryProfileGpu *)malloc(queryProfileGpuSize);
  queryProfileGpu->data = dataGpu;

  return queryProfileGpu;
}

static void deleteQueryProfileGpu(QueryProfileGpu *queryProfileGpu) {
  CUDA_SAFE_CALL(cudaFreeArray(queryProfileGpu->data));
  CUDA_SAFE_CALL(cudaUnbindTexture(qpTexture));
  free(queryProfileGpu);
}

//------------------------------------------------------------------------------
//******************************************************************************

#endif // __CUDACC__
