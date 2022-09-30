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

#ifdef SYCL_LANGUAGE_VERSION

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
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

#define THREADS   64
#define BLOCKS    240

#define MAX_THREADS THREADS

#define INT4_ZERO sycl::int4(0, 0, 0, 0)
#define INT4_SCORE_MIN sycl::int4(SCORE_MIN, SCORE_MIN, SCORE_MIN, SCORE_MIN)

typedef struct GpuDatabase {
    int card;
    char* codes;
    int* starts;
    int* lengths;
    int* indexes;
    int* scores;
    sycl::int2 *hBus;
} GpuDatabase;

struct LongDatabase {
    Chain** database;
    int databaseLen;
    int length;
    int* order;
    int* positions;
    int* indexes;
    GpuDatabase* gpuDatabases;
    int gpuDatabasesLen;
};

typedef struct Context {
    int* scores; 
    int type;
    Chain** queries;
    int queriesLen;
    LongDatabase* longDatabase;
    Scorer* scorer;
    int* indexes;
    int indexesLen;
    int* cards;
    int cardsLen;
} Context;

typedef struct QueryProfile {
    int height;
    int width;
    int length;
    sycl::char4 *data;
    size_t size;
} QueryProfile;

typedef struct QueryProfileGpu {
    dpct::image_matrix *data;
} QueryProfileGpu;

typedef void (*ScoringFunction)(char *, int *, int *, int *, int *,
                                sycl::int2 *);

typedef struct KernelContext {
    int* scores;
    ScoringFunction scoringFunction;
    QueryProfile* queryProfile;
    Chain* query;
    LongDatabase* longDatabase;
    Scorer* scorer;
    int* indexes;
    int indexesLen;
    int card;
} KernelContext;

typedef struct KernelContexts {
    KernelContext* contexts;
    int contextsLen;
    long long cells;
} KernelContexts;

typedef struct Atom {
    int mch;
    sycl::int2 up;
    sycl::int4 lScr;
    sycl::int4 lAff;
    sycl::int4 rScr;
    sycl::int4 rAff;
} Atom;

static dpct::constant_memory<int, 0> gapOpen_;
static dpct::constant_memory<int, 0> gapExtend_;

static dpct::constant_memory<int, 0> rows_;
static dpct::constant_memory<int, 0> rowsPadded_;
static dpct::constant_memory<int, 0> length_;
static dpct::constant_memory<int, 0> iters_;

dpct::image_wrapper<sycl::char4, 2> qpTexture;

//******************************************************************************
// PUBLIC

extern LongDatabase* longDatabaseCreate(Chain** database, int databaseLen, 
    int minLen, int maxLen, int* cards, int cardsLen);

extern void longDatabaseDelete(LongDatabase* longDatabase);

extern void scoreLongDatabaseGpu(int* scores, int type, Chain* query, 
    LongDatabase* longDatabase, Scorer* scorer, int* indexes, int indexesLen, 
    int* cards, int cardsLen, Thread* thread);

extern void scoreLongDatabasesGpu(int* scores, int type, Chain** queries, 
    int queriesLen, LongDatabase* longDatabase, Scorer* scorer, int* indexes, 
    int indexesLen, int* cards, int cardsLen, Thread* thread);

//******************************************************************************

//******************************************************************************
// PRIVATE

// constructor
static LongDatabase* createDatabase(Chain** database, int databaseLen, 
    int minLen, int maxLen, int* cards, int cardsLen);

// destructor
static void deleteDatabase(LongDatabase* database);

// scoring 
static void scoreDatabase(int* scores, int type, Chain** queries, 
    int queriesLen, LongDatabase* longDatabase, Scorer* scorer, int* indexes, 
    int indexesLen, int* cards, int cardsLen, Thread* thread);

static void* scoreDatabaseThread(void* param);

static void scoreDatabaseMulti(int* scores, ScoringFunction scoringFunction, 
    Chain** queries, int queriesLen, LongDatabase* longDatabase, Scorer* scorer, 
    int* indexes, int indexesLen, int* cards, int cardsLen);

static void scoreDatabaseSingle(int* scores, ScoringFunction scoringFunction, 
    Chain** queries, int queriesLen, LongDatabase* longDatabase, Scorer* scorer, 
    int* indexes, int indexesLen, int* cards, int cardsLen);

// cpu kernels
static void* kernelThread(void* param);

static void* kernelsThread(void* param);

// gpu kernels
void hwSolve(char *codes, int *starts, int *lengths, int *indexes, int *scores,
             sycl::int2 *hBus, sycl::nd_item<3> item_ct1, int gapOpen_,
             int gapExtend_, int rows_, int rowsPadded_, int length_,
             int iters_, int *scoresShr, int *hBusScrShr, int *hBusAffShr,
             dpct::image_accessor_ext<sycl::char4, 2> qpTexture);

void nwSolve(char *codes, int *starts, int *lengths, int *indexes, int *scores,
             sycl::int2 *hBus, sycl::nd_item<3> item_ct1, int gapOpen_,
             int gapExtend_, int rows_, int rowsPadded_, int length_,
             int iters_, int *scoresShr, int *hBusScrShr, int *hBusAffShr,
             dpct::image_accessor_ext<sycl::char4, 2> qpTexture);

void ovSolve(char *codes, int *starts, int *lengths, int *indexes, int *scores,
             sycl::int2 *hBus, sycl::nd_item<3> item_ct1, int gapOpen_,
             int gapExtend_, int rows_, int rowsPadded_, int length_,
             int iters_, int *scoresShr, int *hBusScrShr, int *hBusAffShr,
             dpct::image_accessor_ext<sycl::char4, 2> qpTexture);

void swSolve(char *codes, int *starts, int *lengths, int *indexes, int *scores,
             sycl::int2 *hBus, sycl::nd_item<3> item_ct1, int gapOpen_,
             int gapExtend_, int rowsPadded_, int length_, int iters_,
             int *scoresShr, int *hBusScrShr, int *hBusAffShr,
             dpct::image_accessor_ext<sycl::char4, 2> qpTexture);

static int gap(int index, int gapOpen_, int gapExtend_);

void hwSolveSingle(int id, char *codes, int *starts, int *lengths, int *scores,
                   sycl::int2 *hBus, sycl::nd_item<3> item_ct1, int gapOpen_,
                   int gapExtend_, int rows_, int rowsPadded_, int iters_,
                   int *scoresShr, int *hBusScrShr, int *hBusAffShr,
                   dpct::image_accessor_ext<sycl::char4, 2> qpTexture);

void nwSolveSingle(int id, char *codes, int *starts, int *lengths, int *scores,
                   sycl::int2 *hBus, sycl::nd_item<3> item_ct1, int gapOpen_,
                   int gapExtend_, int rows_, int rowsPadded_, int iters_,
                   int *scoresShr, int *hBusScrShr, int *hBusAffShr,
                   dpct::image_accessor_ext<sycl::char4, 2> qpTexture);

void ovSolveSingle(int id, char *codes, int *starts, int *lengths, int *scores,
                   sycl::int2 *hBus, sycl::nd_item<3> item_ct1, int gapOpen_,
                   int gapExtend_, int rows_, int rowsPadded_, int iters_,
                   int *scoresShr, int *hBusScrShr, int *hBusAffShr,
                   dpct::image_accessor_ext<sycl::char4, 2> qpTexture);

void swSolveSingle(int id, char *codes, int *starts, int *lengths, int *scores,
                   sycl::int2 *hBus, sycl::nd_item<3> item_ct1, int gapOpen_,
                   int gapExtend_, int rowsPadded_, int iters_, int *scoresShr,
                   int *hBusScrShr, int *hBusAffShr,
                   dpct::image_accessor_ext<sycl::char4, 2> qpTexture);

// query profile
static QueryProfile* createQueryProfile(Chain* query, Scorer* scorer);

static void deleteQueryProfile(QueryProfile* queryProfile);

static QueryProfileGpu* createQueryProfileGpu(QueryProfile* queryProfile);

static void deleteQueryProfileGpu(QueryProfileGpu* queryProfileGpu);

//******************************************************************************

//******************************************************************************
// PUBLIC

//------------------------------------------------------------------------------
// CONSTRUCTOR, DESTRUCTOR

extern LongDatabase* longDatabaseCreate(Chain** database, int databaseLen, 
    int minLen, int maxLen, int* cards, int cardsLen) {
    return createDatabase(database, databaseLen, minLen, maxLen, cards, cardsLen);
}

extern void longDatabaseDelete(LongDatabase* longDatabase) {
    deleteDatabase(longDatabase);
}

extern size_t longDatabaseGpuMemoryConsumption(Chain** database, int databaseLen,
    int minLen, int maxLen) {

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
    size_t hBusSize = codesLen * sizeof(sycl::int2);

    size_t memory = codesSize + startsSize + lengthsSize + indexesSize + 
        scoresSize + hBusSize;

    return memory;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// CPU KERNELS

extern void scoreLongDatabaseGpu(int* scores, int type, Chain* query, 
    LongDatabase* longDatabase, Scorer* scorer, int* indexes, int indexesLen, 
    int* cards, int cardsLen, Thread* thread) {
    scoreDatabase(scores, type, &query, 1, longDatabase, scorer, indexes, 
        indexesLen, cards, cardsLen, thread);
}

extern void scoreLongDatabasesGpu(int* scores, int type, Chain** queries, 
    int queriesLen, LongDatabase* longDatabase, Scorer* scorer, int* indexes, 
    int indexesLen, int* cards, int cardsLen, Thread* thread) {
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
                                    int cardsLen) try {

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
    
    int* order = (int*) malloc(length * sizeof(int));

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
    int* lengths = (int*) malloc(lengthsSize);
    
    size_t startsSize = length * sizeof(int);
    int* starts = (int*) malloc(startsSize);
    
    size_t codesSize = codesLen * sizeof(char);
    char* codes = (char*) malloc(codesSize);

    //**************************************************************************
    
    //**************************************************************************
    // CREATE STRUCTURES
    
    long codesOff = 0;
    for (int i = 0; i < length; ++i) {

        Chain* chain = database[order[i]];
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
    int* indexes = (int*) malloc(indexesSize);

    for (int i = 0; i < length; ++i) {
        indexes[i] = i;
    }
     
    //**************************************************************************
    
    //**************************************************************************
    // CREATE POSITION ARRAY
    
    int* positions = (int*) malloc(databaseLen * sizeof(int));

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
    GpuDatabase* gpuDatabases = (GpuDatabase*) malloc(gpuDatabasesSize);

    for (int i = 0; i < cardsLen; ++i) {

        int card = cards[i];
        /*
        DPCT1093:313: The "card" may not be the best XPU device. Adjust the
        selected device if needed.
        */
        /*
        DPCT1003:314: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_SAFE_CALL((dpct::dev_mgr::instance().select_device(card), 0));

        char* codesGpu;
        /*
        DPCT1003:315: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_SAFE_CALL((codesGpu = (char *)sycl::malloc_device(
                            codesSize, dpct::get_default_queue()),
                        0));
        /*
        DPCT1003:316: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_SAFE_CALL((
            dpct::get_default_queue().memcpy(codesGpu, codes, codesSize).wait(),
            0));

        int* startsGpu;
        /*
        DPCT1003:317: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_SAFE_CALL((startsGpu = (int *)sycl::malloc_device(
                            startsSize, dpct::get_default_queue()),
                        0));
        /*
        DPCT1003:318: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_SAFE_CALL((dpct::get_default_queue()
                            .memcpy(startsGpu, starts, startsSize)
                            .wait(),
                        0));

        int* lengthsGpu;
        /*
        DPCT1003:319: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_SAFE_CALL((lengthsGpu = (int *)sycl::malloc_device(
                            lengthsSize, dpct::get_default_queue()),
                        0));
        /*
        DPCT1003:320: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_SAFE_CALL((dpct::get_default_queue()
                            .memcpy(lengthsGpu, lengths, lengthsSize)
                            .wait(),
                        0));

        int* indexesGpu;
        /*
        DPCT1003:321: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_SAFE_CALL((indexesGpu = (int *)sycl::malloc_device(
                            indexesSize, dpct::get_default_queue()),
                        0));
        /*
        DPCT1003:322: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_SAFE_CALL((dpct::get_default_queue()
                            .memcpy(indexesGpu, indexes, indexesSize)
                            .wait(),
                        0));

        // additional structures

        size_t scoresSize = length * sizeof(int);
        int* scoresGpu;
        /*
        DPCT1003:323: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_SAFE_CALL((scoresGpu = (int *)sycl::malloc_device(
                            scoresSize, dpct::get_default_queue()),
                        0));

        sycl::int2 *hBusGpu;
        size_t hBusSize = codesLen * sizeof(sycl::int2);
        /*
        DPCT1003:324: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_SAFE_CALL((hBusGpu = (sycl::int2 *)sycl::malloc_device(
                            hBusSize, dpct::get_default_queue()),
                        0));

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
    LongDatabase* longDatabase = (LongDatabase*) malloc(longDatabaseSize);
    
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
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void deleteDatabase(LongDatabase *database) try {

    if (database == NULL) {
        return;
    }

    for (int i = 0; i < database->gpuDatabasesLen; ++i) {
    
        GpuDatabase* gpuDatabase = &(database->gpuDatabases[i]);

        /*
        DPCT1093:325: The "gpuDatabase->card" may not be the best XPU device.
        Adjust the selected device if needed.
        */
        /*
        DPCT1003:326: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_SAFE_CALL(
            (dpct::dev_mgr::instance().select_device(gpuDatabase->card), 0));

        /*
        DPCT1003:327: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_SAFE_CALL(
            (sycl::free(gpuDatabase->codes, dpct::get_default_queue()), 0));
        /*
        DPCT1003:328: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_SAFE_CALL(
            (sycl::free(gpuDatabase->starts, dpct::get_default_queue()), 0));
        /*
        DPCT1003:329: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_SAFE_CALL(
            (sycl::free(gpuDatabase->lengths, dpct::get_default_queue()), 0));
        /*
        DPCT1003:330: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_SAFE_CALL(
            (sycl::free(gpuDatabase->indexes, dpct::get_default_queue()), 0));
        /*
        DPCT1003:331: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_SAFE_CALL(
            (sycl::free(gpuDatabase->scores, dpct::get_default_queue()), 0));
        /*
        DPCT1003:332: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_SAFE_CALL((sycl::free(gpuDatabase->hBus, dpct::get_default_queue()), 0));
    }

    free(database->gpuDatabases);
    free(database->order);
    free(database->positions);
    free(database->indexes);

    free(database);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// SCORING

static void scoreDatabase(int* scores, int type, Chain** queries, 
    int queriesLen, LongDatabase* longDatabase, Scorer* scorer, int* indexes, 
    int indexesLen, int* cards, int cardsLen, Thread* thread) {
    
    ASSERT(cardsLen > 0, "no GPUs available");
    
    Context* param = (Context*) malloc(sizeof(Context));
    
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
        threadCreate(thread, scoreDatabaseThread, (void*) param);
    }
}

static void* scoreDatabaseThread(void* param) {

    Context* context = (Context*) param;
    
    int* scores = context->scores;
    int type = context->type;
    Chain** queries = context->queries;
    int queriesLen = context->queriesLen;
    LongDatabase* longDatabase = context->longDatabase;
    Scorer* scorer = context->scorer;
    int* indexes = context->indexes;
    int indexesLen = context->indexesLen;
    int* cards = context->cards;
    int cardsLen = context->cardsLen;

    if (longDatabase == NULL) {
        free(param);
        return NULL;
    }

    //**************************************************************************
    // CREATE NEW INDEXES ARRAY IF NEEDED
    
    int* newIndexes = NULL;
    int newIndexesLen = 0;

    int deleteIndexes;

    if (indexes != NULL) {

        // translate and filter indexes

        int databaseLen = longDatabase->databaseLen;
        int* positions = longDatabase->positions;
        
        newIndexes = (int*) malloc(indexesLen * sizeof(int));
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

static void scoreDatabaseMulti(int* scores, ScoringFunction scoringFunction, 
    Chain** queries, int queriesLen, LongDatabase* longDatabase, Scorer* scorer, 
    int* indexes, int indexesLen, int* cards, int cardsLen) {

    //**************************************************************************
    // CREATE QUERY PROFILES
    
    size_t profilesSize = queriesLen * sizeof(QueryProfile*);
    QueryProfile** profiles = (QueryProfile**) malloc(profilesSize);
    
    for (int i = 0; i < queriesLen; ++i) {
        profiles[i] = createQueryProfile(queries[i], scorer);
    }
    
    //**************************************************************************
    
    //**************************************************************************
    // CREATE BALANCING DATA

    Chain** database = longDatabase->database;
    int* order = longDatabase->order;

    size_t weightsSize = indexesLen * sizeof(int);
    int* weights = (int*) malloc(weightsSize);
    memset(weights, 0, weightsSize);

    for (int i = 0; i < indexesLen; ++i) {
        weights[i] += chainGetLength(database[order[indexes[i]]]);
    }

    //**************************************************************************

    //**************************************************************************
    // SCORE MULTICARDED
    
    int contextsLen = cardsLen * queriesLen;
    size_t contextsSize = contextsLen * sizeof(KernelContext);
    KernelContext* contexts = (KernelContext*) malloc(contextsSize);
    
    size_t tasksSize = contextsLen * sizeof(Thread);
    Thread* tasks = (Thread*) malloc(tasksSize);

    int databaseLen = longDatabase->databaseLen;
    
    int cardsChunk = cardsLen / queriesLen;
    int cardsAdd = cardsLen % queriesLen;
    int cardsOff = 0;

    int* idxChunksOff = (int*) malloc(cardsLen * sizeof(int));
    int* idxChunksLens = (int*) malloc(cardsLen * sizeof(int));
    int idxChunksLen = 0;

    int length = 0;

    for (int i = 0, k = 0; i < queriesLen; ++i) {

        int cCardsLen = cardsChunk + (i < cardsAdd);
        int* cCards = cards + cardsOff;
        cardsOff += cCardsLen;
        
        QueryProfile* queryProfile = profiles[i];

        int chunks = std::min(cCardsLen, indexesLen);
        if (chunks != idxChunksLen) {
            weightChunkArray(idxChunksOff, idxChunksLens, &idxChunksLen, 
                weights, indexesLen, chunks);
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

static void scoreDatabaseSingle(int* scores, ScoringFunction scoringFunction, 
    Chain** queries, int queriesLen, LongDatabase* longDatabase, Scorer* scorer, 
    int* indexes, int indexesLen, int* cards, int cardsLen) {

    //**************************************************************************
    // CREATE CONTEXTS
    
    size_t contextsSize = cardsLen * sizeof(KernelContext);
    KernelContexts* contexts = (KernelContexts*) malloc(contextsSize);
    
    for (int i = 0; i < cardsLen; ++i) {
        size_t size = queriesLen * sizeof(KernelContext);
        contexts[i].contexts = (KernelContext*) malloc(size);
        contexts[i].contextsLen = 0;
        contexts[i].cells = 0;
    }
    
    //**************************************************************************    
    
    //**************************************************************************
    // SCORE MULTITHREADED
    
    size_t tasksSize = cardsLen * sizeof(Thread);
    Thread* tasks = (Thread*) malloc(tasksSize);
    
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

static void *kernelsThread(void *param) try {

    KernelContexts* context = (KernelContexts*) param;

    KernelContext* contexts = context->contexts;
    int contextsLen = context->contextsLen;

    for (int i = 0; i < contextsLen; ++i) {
    
        Chain* query = contexts[i].query;
        Scorer* scorer = contexts[i].scorer;
        int card = contexts[i].card;
        
        int currentCard;
        CUDA_SAFE_CALL(currentCard = dpct::dev_mgr::instance().current_device_id());
        if (currentCard != card) {
            /*
            DPCT1093:333: The "card" may not be the best XPU device. Adjust the
            selected device if needed.
            */
            /*
            DPCT1003:334: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            CUDA_SAFE_CALL((dpct::dev_mgr::instance().select_device(card), 0));
        }
    
        contexts[i].queryProfile = createQueryProfile(query, scorer);
        
        kernelThread(&(contexts[i]));
        
        deleteQueryProfile(contexts[i].queryProfile);
    }
    
    return NULL;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void *kernelThread(void *param) try {

    KernelContext* context = (KernelContext*) param;
    
    int* scores = context->scores;
    ScoringFunction scoringFunction = context->scoringFunction;
    QueryProfile* queryProfile = context->queryProfile;
    LongDatabase* longDatabase = context->longDatabase;
    Scorer* scorer = context->scorer;
    int* indexes = context->indexes;
    int indexesLen = context->indexesLen;
    int card = context->card;

    //**************************************************************************
    // FIND DATABASE
    
    GpuDatabase* gpuDatabases = longDatabase->gpuDatabases;
    int gpuDatabasesLen = longDatabase->gpuDatabasesLen;
    
    GpuDatabase* gpuDatabase = NULL;
    
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
    CUDA_SAFE_CALL(currentCard = dpct::dev_mgr::instance().current_device_id());
    if (currentCard != card) {
        /*
        DPCT1093:335: The "card" may not be the best XPU device. Adjust the
        selected device if needed.
        */
        /*
        DPCT1003:336: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_SAFE_CALL((dpct::dev_mgr::instance().select_device(card), 0));
    }
    
    //**************************************************************************

    //**************************************************************************
    // FIX INDEXES
    
    int deleteIndexes;
    int* indexesGpu;
    
    if (indexesLen == longDatabase->length) {
        indexes = longDatabase->indexes;
        indexesLen = longDatabase->length;
        indexesGpu = gpuDatabase->indexes;
        deleteIndexes = 0;
    } else {
        size_t indexesSize = indexesLen * sizeof(int);
        /*
        DPCT1003:337: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_SAFE_CALL((indexesGpu = (int *)sycl::malloc_device(
                            indexesSize, dpct::get_default_queue()),
                        0));
        CUDA_SAFE_CALL((dpct::get_default_queue()
                            .memcpy(indexesGpu, indexes, indexesSize)
                            .wait(),
                        0));
        deleteIndexes = 1;
    }

    //**************************************************************************

    //**************************************************************************
    // PREPARE GPU
    
    QueryProfileGpu* queryProfileGpu = createQueryProfileGpu(queryProfile);

    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);
    int rows = queryProfile->length;
    int rowsGpu = queryProfile->height * 4;
    int iters = rowsGpu / (THREADS * 4) + (rowsGpu % (THREADS * 4) != 0);

    /*
    DPCT1003:338: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(gapOpen_.get_ptr(), &gapOpen, sizeof(int))
                        .wait(),
                    0));
    /*
    DPCT1003:339: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(gapExtend_.get_ptr(), &gapExtend, sizeof(int))
                        .wait(),
                    0));
    /*
    DPCT1003:340: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(rows_.get_ptr(), &rows, sizeof(int))
                        .wait(),
                    0));
    /*
    DPCT1003:341: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(rowsPadded_.get_ptr(), &rowsGpu, sizeof(int))
                        .wait(),
                    0));
    /*
    DPCT1003:342: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(length_.get_ptr(), &indexesLen, sizeof(int))
                        .wait(),
                    0));
    /*
    DPCT1003:343: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(iters_.get_ptr(), &iters, sizeof(int))
                        .wait(),
                    0));

    //**************************************************************************
    
        
    //**************************************************************************
    // SOLVE
    
    char* codesGpu = gpuDatabase->codes;
    int* startsGpu = gpuDatabase->starts;
    int* lengthsGpu = gpuDatabase->lengths;
    int* scoresGpu = gpuDatabase->scores;
    sycl::int2 *hBusGpu = gpuDatabase->hBus;

    dpct::get_default_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, BLOCKS) *
                              sycl::range<3>(1, 1, THREADS),
                          sycl::range<3>(1, 1, THREADS)),
        [=](sycl::nd_item<3> item_ct1) {
            (codesGpu, startsGpu, lengthsGpu, indexesGpu, scoresGpu, hBusGpu);
        });

    //**************************************************************************
    
    //**************************************************************************
    // SAVE RESULTS
    
    int length = longDatabase->length;
    
    size_t scoresSize = length * sizeof(int);
    int* scoresCpu = (int*) malloc(scoresSize);

    /*
    DPCT1003:344: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(scoresCpu, scoresGpu, scoresSize)
                        .wait(),
                    0));

    int* order = longDatabase->order;
    
    for (int i = 0; i < indexesLen; ++i) {
        scores[order[indexes[i]]] = scoresCpu[indexes[i]];
    }
    
    free(scoresCpu);
                
    //**************************************************************************
    
    //**************************************************************************
    // CLEAN MEMORY
    
    deleteQueryProfileGpu(queryProfileGpu);
    
    if (deleteIndexes) {
        /*
        DPCT1003:345: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_SAFE_CALL((sycl::free(indexesGpu, dpct::get_default_queue()), 0));
    }

    //**************************************************************************
    
    return NULL;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// GPU KERNELS

void hwSolve(char *codes, int *starts, int *lengths, int *indexes, int *scores,
             sycl::int2 *hBus, sycl::nd_item<3> item_ct1, int gapOpen_,
             int gapExtend_, int rows_, int rowsPadded_, int length_,
             int iters_, int *scoresShr, int *hBusScrShr, int *hBusAffShr,
             dpct::image_accessor_ext<sycl::char4, 2> qpTexture) {

    for (int i = item_ct1.get_group(2); i < length_;
         i += item_ct1.get_group_range(2)) {
        hwSolveSingle(indexes[i], codes, starts, lengths, scores, hBus,
                      item_ct1, gapOpen_, gapExtend_, rows_, rowsPadded_,
                      iters_, scoresShr, hBusScrShr, hBusAffShr, qpTexture);
    }
}

void nwSolve(char *codes, int *starts, int *lengths, int *indexes, int *scores,
             sycl::int2 *hBus, sycl::nd_item<3> item_ct1, int gapOpen_,
             int gapExtend_, int rows_, int rowsPadded_, int length_,
             int iters_, int *scoresShr, int *hBusScrShr, int *hBusAffShr,
             dpct::image_accessor_ext<sycl::char4, 2> qpTexture) {

    for (int i = item_ct1.get_group(2); i < length_;
         i += item_ct1.get_group_range(2)) {
        nwSolveSingle(indexes[i], codes, starts, lengths, scores, hBus,
                      item_ct1, gapOpen_, gapExtend_, rows_, rowsPadded_,
                      iters_, scoresShr, hBusScrShr, hBusAffShr, qpTexture);
    }
}

void ovSolve(char *codes, int *starts, int *lengths, int *indexes, int *scores,
             sycl::int2 *hBus, sycl::nd_item<3> item_ct1, int gapOpen_,
             int gapExtend_, int rows_, int rowsPadded_, int length_,
             int iters_, int *scoresShr, int *hBusScrShr, int *hBusAffShr,
             dpct::image_accessor_ext<sycl::char4, 2> qpTexture) {

    for (int i = item_ct1.get_group(2); i < length_;
         i += item_ct1.get_group_range(2)) {
        ovSolveSingle(indexes[i], codes, starts, lengths, scores, hBus,
                      item_ct1, gapOpen_, gapExtend_, rows_, rowsPadded_,
                      iters_, scoresShr, hBusScrShr, hBusAffShr, qpTexture);
    }
}

void swSolve(char *codes, int *starts, int *lengths, int *indexes, int *scores,
             sycl::int2 *hBus, sycl::nd_item<3> item_ct1, int gapOpen_,
             int gapExtend_, int rowsPadded_, int length_, int iters_,
             int *scoresShr, int *hBusScrShr, int *hBusAffShr,
             dpct::image_accessor_ext<sycl::char4, 2> qpTexture) {

    for (int i = item_ct1.get_group(2); i < length_;
         i += item_ct1.get_group_range(2)) {
        swSolveSingle(indexes[i], codes, starts, lengths, scores, hBus,
                      item_ct1, gapOpen_, gapExtend_, rowsPadded_, iters_,
                      scoresShr, hBusScrShr, hBusAffShr, qpTexture);
    }
}

static int gap(int index, int gapOpen_, int gapExtend_) {
    return (-gapOpen_ - index * gapExtend_) * (index >= 0);
}

void hwSolveSingle(int id, char *codes, int *starts, int *lengths, int *scores,
                   sycl::int2 *hBus, sycl::nd_item<3> item_ct1, int gapOpen_,
                   int gapExtend_, int rows_, int rowsPadded_, int iters_,
                   int *scoresShr, int *hBusScrShr, int *hBusAffShr,
                   dpct::image_accessor_ext<sycl::char4, 2> qpTexture) {

    int off = starts[id];
    int cols = lengths[id];

    int score = SCORE_MIN;

    int width = cols * iters_ + 2 * (item_ct1.get_local_range(2) - 1);
    int col = -item_ct1.get_local_id(2);
    int row = item_ct1.get_local_id(2) * 4;
    int iter = 0;
    
    Atom atom;
    atom.mch = gap(row - 1, gapOpen_, gapExtend_);
    atom.lScr = sycl::int4(
        gap(row, gapOpen_, gapExtend_), gap(row + 1, gapOpen_, gapExtend_),
        gap(row + 2, gapOpen_, gapExtend_), gap(row + 3, gapOpen_, gapExtend_));
    atom.lAff = INT4_SCORE_MIN;

    hBusScrShr[item_ct1.get_local_id(2)] = 0;
    hBusAffShr[item_ct1.get_local_id(2)] = SCORE_MIN;

    for (int i = 0; i < width; ++i) {
    
        int del;
        int valid = col >= 0 && row < rowsPadded_;
    
        if (valid) {

            if (iter != 0 && item_ct1.get_local_id(2) == 0) {
                atom.up = hBus[off + col];
            } else {
                atom.up.x() = hBusScrShr[item_ct1.get_local_id(2)];
                atom.up.y() = hBusAffShr[item_ct1.get_local_id(2)];
            }
            
            char code = codes[off + col];
            sycl::char4 rowScores = qpTexture.read(code, row >> 2);

            del = sycl::max((int)(atom.up.x() - gapOpen_),
                            (int)(atom.up.y() - gapExtend_));
            int ins = sycl::max((int)(atom.lScr.x() - gapOpen_),
                                (int)(atom.lAff.x() - gapExtend_));
            int mch = atom.mch + rowScores.x();

            atom.rScr.x() = MAX3(mch, del, ins);
            atom.rAff.x() = ins;

            del = sycl::max((int)(atom.rScr.x() - gapOpen_), (int)(del - gapExtend_));
            ins = sycl::max((int)(atom.lScr.y() - gapOpen_),
                            (int)(atom.lAff.y() - gapExtend_));
            mch = atom.lScr.x() + rowScores.y();

            atom.rScr.y() = MAX3(mch, del, ins);
            atom.rAff.y() = ins;

            del = sycl::max((int)(atom.rScr.y() - gapOpen_), (int)(del - gapExtend_));
            ins = sycl::max((int)(atom.lScr.z() - gapOpen_),
                            (int)(atom.lAff.z() - gapExtend_));
            mch = atom.lScr.y() + rowScores.z();

            atom.rScr.z() = MAX3(mch, del, ins);
            atom.rAff.z() = ins;

            del = sycl::max((int)(atom.rScr.z() - gapOpen_), (int)(del - gapExtend_));
            ins = sycl::max((int)(atom.lScr.w() - gapOpen_),
                            (int)(atom.lAff.w() - gapExtend_));
            mch = atom.lScr.z() + rowScores.w();

            atom.rScr.w() = MAX3(mch, del, ins);
            atom.rAff.w() = ins;

            if (row + 0 == rows_ - 1) score = sycl::max(score, atom.rScr.x());
            if (row + 1 == rows_ - 1) score = sycl::max(score, atom.rScr.y());
            if (row + 2 == rows_ - 1) score = sycl::max(score, atom.rScr.z());
            if (row + 3 == rows_ - 1) score = sycl::max(score, atom.rScr.w());

            atom.mch = atom.up.x();
            VEC4_ASSIGN(atom.lScr, atom.rScr);
            VEC4_ASSIGN(atom.lAff, atom.rAff);
        }

        /*
        DPCT1065:347: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        if (valid) {
            if (iter < iters_ - 1 &&
                item_ct1.get_local_id(2) == item_ct1.get_local_range(2) - 1) {
                VEC2_ASSIGN(hBus[off + col], sycl::int2(atom.rScr.w(), del));
            } else {
                hBusScrShr[item_ct1.get_local_id(2) + 1] = atom.rScr.w();
                hBusAffShr[item_ct1.get_local_id(2) + 1] = del;
            }
        }
        
        col++;
        
        if (col == cols) {

            col = 0;
            row += item_ct1.get_local_range(2) * 4;
            iter++;

            atom.mch = gap(row - 1, gapOpen_, gapExtend_);
            atom.lScr = sycl::int4(gap(row, gapOpen_, gapExtend_),
                                   gap(row + 1, gapOpen_, gapExtend_),
                                   gap(row + 2, gapOpen_, gapExtend_),
                                   gap(row + 3, gapOpen_, gapExtend_));
                ;
            atom.lAff = INT4_SCORE_MIN;
        }

        /*
        DPCT1065:348: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    // write all scores
    scoresShr[item_ct1.get_local_id(2)] = score;
    /*
    DPCT1065:346: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // gather scores
    if (item_ct1.get_local_id(2) == 0) {

        for (int i = 1; i < item_ct1.get_local_range(2); ++i) {
            score = sycl::max(score, scoresShr[i]);
        }
    
        scores[id] = score;
    }
}

void nwSolveSingle(int id, char *codes, int *starts, int *lengths, int *scores,
                   sycl::int2 *hBus, sycl::nd_item<3> item_ct1, int gapOpen_,
                   int gapExtend_, int rows_, int rowsPadded_, int iters_,
                   int *scoresShr, int *hBusScrShr, int *hBusAffShr,
                   dpct::image_accessor_ext<sycl::char4, 2> qpTexture) {

    int off = starts[id];
    int cols = lengths[id];

    int score = SCORE_MIN;

    int width = cols * iters_ + 2 * (item_ct1.get_local_range(2) - 1);
    int col = -item_ct1.get_local_id(2);
    int row = item_ct1.get_local_id(2) * 4;
    int iter = 0;
    
    Atom atom;
    atom.mch = gap(row - 1, gapOpen_, gapExtend_);
    atom.lScr = sycl::int4(
        gap(row, gapOpen_, gapExtend_), gap(row + 1, gapOpen_, gapExtend_),
        gap(row + 2, gapOpen_, gapExtend_), gap(row + 3, gapOpen_, gapExtend_));
    atom.lAff = INT4_SCORE_MIN;

    for (int i = 0; i < width; ++i) {
    
        int del;
        int valid = col >= 0 && row < rowsPadded_;
    
        if (valid) {

            if (item_ct1.get_local_id(2) == 0) {
                if (iter == 0) {
                   atom.up.x() = gap(col, gapOpen_, gapExtend_);
                   atom.up.y() = SCORE_MIN;
                } else {
                    atom.up = hBus[off + col];
                }
            } else {
                atom.up.x() = hBusScrShr[item_ct1.get_local_id(2)];
                atom.up.y() = hBusAffShr[item_ct1.get_local_id(2)];
            }
            
            char code = codes[off + col];
            sycl::char4 rowScores = qpTexture.read(code, row >> 2);

            del = sycl::max((int)(atom.up.x() - gapOpen_),
                            (int)(atom.up.y() - gapExtend_));
            int ins = sycl::max((int)(atom.lScr.x() - gapOpen_),
                                (int)(atom.lAff.x() - gapExtend_));
            int mch = atom.mch + rowScores.x();

            atom.rScr.x() = MAX3(mch, del, ins);
            atom.rAff.x() = ins;

            del = sycl::max((int)(atom.rScr.x() - gapOpen_), (int)(del - gapExtend_));
            ins = sycl::max((int)(atom.lScr.y() - gapOpen_),
                            (int)(atom.lAff.y() - gapExtend_));
            mch = atom.lScr.x() + rowScores.y();

            atom.rScr.y() = MAX3(mch, del, ins);
            atom.rAff.y() = ins;

            del = sycl::max((int)(atom.rScr.y() - gapOpen_), (int)(del - gapExtend_));
            ins = sycl::max((int)(atom.lScr.z() - gapOpen_),
                            (int)(atom.lAff.z() - gapExtend_));
            mch = atom.lScr.y() + rowScores.z();

            atom.rScr.z() = MAX3(mch, del, ins);
            atom.rAff.z() = ins;

            del = sycl::max((int)(atom.rScr.z() - gapOpen_), (int)(del - gapExtend_));
            ins = sycl::max((int)(atom.lScr.w() - gapOpen_),
                            (int)(atom.lAff.w() - gapExtend_));
            mch = atom.lScr.z() + rowScores.w();

            atom.rScr.w() = MAX3(mch, del, ins);
            atom.rAff.w() = ins;

            atom.mch = atom.up.x();
            VEC4_ASSIGN(atom.lScr, atom.rScr);
            VEC4_ASSIGN(atom.lAff, atom.rAff);
        }

        /*
        DPCT1065:350: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        if (valid) {
            if (iter < iters_ - 1 &&
                item_ct1.get_local_id(2) == item_ct1.get_local_range(2) - 1) {
                VEC2_ASSIGN(hBus[off + col], sycl::int2(atom.rScr.w(), del));
            } else {
                hBusScrShr[item_ct1.get_local_id(2) + 1] = atom.rScr.w();
                hBusAffShr[item_ct1.get_local_id(2) + 1] = del;
            }
        }
        
        col++;
        
        if (col == cols) {

            if (row + 0 == rows_ - 1) score = sycl::max(score, atom.lScr.x());
            if (row + 1 == rows_ - 1) score = sycl::max(score, atom.lScr.y());
            if (row + 2 == rows_ - 1) score = sycl::max(score, atom.lScr.z());
            if (row + 3 == rows_ - 1) score = sycl::max(score, atom.lScr.w());

            col = 0;
            row += item_ct1.get_local_range(2) * 4;
            iter++;

            atom.mch = gap(row - 1, gapOpen_, gapExtend_);
            atom.lScr = sycl::int4(gap(row, gapOpen_, gapExtend_),
                                   gap(row + 1, gapOpen_, gapExtend_),
                                   gap(row + 2, gapOpen_, gapExtend_),
                                   gap(row + 3, gapOpen_, gapExtend_));
                ;
            atom.lAff = INT4_SCORE_MIN;
        }

        /*
        DPCT1065:351: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    // write all scores
    scoresShr[item_ct1.get_local_id(2)] = score;
    /*
    DPCT1065:349: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // gather scores
    if (item_ct1.get_local_id(2) == 0) {

        for (int i = 1; i < item_ct1.get_local_range(2); ++i) {
            score = sycl::max(score, scoresShr[i]);
        }
    
        scores[id] = score;
    }
}

void ovSolveSingle(int id, char *codes, int *starts, int *lengths, int *scores,
                   sycl::int2 *hBus, sycl::nd_item<3> item_ct1, int gapOpen_,
                   int gapExtend_, int rows_, int rowsPadded_, int iters_,
                   int *scoresShr, int *hBusScrShr, int *hBusAffShr,
                   dpct::image_accessor_ext<sycl::char4, 2> qpTexture) {

    int off = starts[id];
    int cols = lengths[id];

    int score = SCORE_MIN;

    int width = cols * iters_ + 2 * (item_ct1.get_local_range(2) - 1);
    int col = -item_ct1.get_local_id(2);
    int row = item_ct1.get_local_id(2) * 4;
    int iter = 0;
    
    Atom atom;
    atom.mch = 0;
    atom.lScr = INT4_ZERO;
    atom.lAff = INT4_SCORE_MIN;

    hBusScrShr[item_ct1.get_local_id(2)] = 0;
    hBusAffShr[item_ct1.get_local_id(2)] = SCORE_MIN;

    for (int i = 0; i < width; ++i) {
    
        int del;
        int valid = col >= 0 && row < rowsPadded_;
    
        if (valid) {

            if (iter != 0 && item_ct1.get_local_id(2) == 0) {
                atom.up = hBus[off + col];
            } else {
                atom.up.x() = hBusScrShr[item_ct1.get_local_id(2)];
                atom.up.y() = hBusAffShr[item_ct1.get_local_id(2)];
            }
            
            char code = codes[off + col];
            sycl::char4 rowScores = qpTexture.read(code, row >> 2);

            del = sycl::max((int)(atom.up.x() - gapOpen_),
                            (int)(atom.up.y() - gapExtend_));
            int ins = sycl::max((int)(atom.lScr.x() - gapOpen_),
                                (int)(atom.lAff.x() - gapExtend_));
            int mch = atom.mch + rowScores.x();

            atom.rScr.x() = MAX3(mch, del, ins);
            atom.rAff.x() = ins;

            del = sycl::max((int)(atom.rScr.x() - gapOpen_), (int)(del - gapExtend_));
            ins = sycl::max((int)(atom.lScr.y() - gapOpen_),
                            (int)(atom.lAff.y() - gapExtend_));
            mch = atom.lScr.x() + rowScores.y();

            atom.rScr.y() = MAX3(mch, del, ins);
            atom.rAff.y() = ins;

            del = sycl::max((int)(atom.rScr.y() - gapOpen_), (int)(del - gapExtend_));
            ins = sycl::max((int)(atom.lScr.z() - gapOpen_),
                            (int)(atom.lAff.z() - gapExtend_));
            mch = atom.lScr.y() + rowScores.z();

            atom.rScr.z() = MAX3(mch, del, ins);
            atom.rAff.z() = ins;

            del = sycl::max((int)(atom.rScr.z() - gapOpen_), (int)(del - gapExtend_));
            ins = sycl::max((int)(atom.lScr.w() - gapOpen_),
                            (int)(atom.lAff.w() - gapExtend_));
            mch = atom.lScr.z() + rowScores.w();

            atom.rScr.w() = MAX3(mch, del, ins);
            atom.rAff.w() = ins;

            if (row + 0 == rows_ - 1) score = sycl::max(score, atom.rScr.x());
            if (row + 1 == rows_ - 1) score = sycl::max(score, atom.rScr.y());
            if (row + 2 == rows_ - 1) score = sycl::max(score, atom.rScr.z());
            if (row + 3 == rows_ - 1) score = sycl::max(score, atom.rScr.w());

            atom.mch = atom.up.x();
            VEC4_ASSIGN(atom.lScr, atom.rScr);
            VEC4_ASSIGN(atom.lAff, atom.rAff);
        }

        /*
        DPCT1065:353: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        if (valid) {
            if (iter < iters_ - 1 &&
                item_ct1.get_local_id(2) == item_ct1.get_local_range(2) - 1) {
                VEC2_ASSIGN(hBus[off + col], sycl::int2(atom.rScr.w(), del));
            } else {
                hBusScrShr[item_ct1.get_local_id(2) + 1] = atom.rScr.w();
                hBusAffShr[item_ct1.get_local_id(2) + 1] = del;
            }
        }
        
        col++;
        
        if (col == cols) {

            if (row < rows_) {
                score = sycl::max(score, atom.lScr.x());
                score = sycl::max(score, atom.lScr.y());
                score = sycl::max(score, atom.lScr.z());
                score = sycl::max(score, atom.lScr.w());
            }

            col = 0;
            row += item_ct1.get_local_range(2) * 4;
            iter++;
            
            atom.mch = 0;
            atom.lScr = INT4_ZERO;
            atom.lAff = INT4_SCORE_MIN;
        }

        /*
        DPCT1065:354: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    // write all scores
    scoresShr[item_ct1.get_local_id(2)] = score;
    /*
    DPCT1065:352: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // gather scores
    if (item_ct1.get_local_id(2) == 0) {

        for (int i = 1; i < item_ct1.get_local_range(2); ++i) {
            score = sycl::max(score, scoresShr[i]);
        }
    
        scores[id] = score;
    }
}

void swSolveSingle(int id, char *codes, int *starts, int *lengths, int *scores,
                   sycl::int2 *hBus, sycl::nd_item<3> item_ct1, int gapOpen_,
                   int gapExtend_, int rowsPadded_, int iters_, int *scoresShr,
                   int *hBusScrShr, int *hBusAffShr,
                   dpct::image_accessor_ext<sycl::char4, 2> qpTexture) {

    int off = starts[id];
    int cols = lengths[id];

    int score = 0;

    int width = cols * iters_ + 2 * (item_ct1.get_local_range(2) - 1);
    int col = -item_ct1.get_local_id(2);
    int row = item_ct1.get_local_id(2) * 4;
    int iter = 0;
    
    Atom atom;
    atom.mch = 0;
    atom.lScr = INT4_ZERO;
    atom.lAff = INT4_SCORE_MIN;

    hBusScrShr[item_ct1.get_local_id(2)] = 0;
    hBusAffShr[item_ct1.get_local_id(2)] = SCORE_MIN;

    for (int i = 0; i < width; ++i) {
    
        int del;
        int valid = col >= 0 && row < rowsPadded_;
    
        if (valid) {

            if (iter != 0 && item_ct1.get_local_id(2) == 0) {
                atom.up = hBus[off + col];
            } else {
                atom.up.x() = hBusScrShr[item_ct1.get_local_id(2)];
                atom.up.y() = hBusAffShr[item_ct1.get_local_id(2)];
            }
            
            char code = codes[off + col];
            sycl::char4 rowScores = qpTexture.read(code, row >> 2);

            del = sycl::max((int)(atom.up.x() - gapOpen_),
                            (int)(atom.up.y() - gapExtend_));
            int ins = sycl::max((int)(atom.lScr.x() - gapOpen_),
                                (int)(atom.lAff.x() - gapExtend_));
            int mch = atom.mch + rowScores.x();

            atom.rScr.x() = MAX4(0, mch, del, ins);
            atom.rAff.x() = ins;

            del = sycl::max((int)(atom.rScr.x() - gapOpen_), (int)(del - gapExtend_));
            ins = sycl::max((int)(atom.lScr.y() - gapOpen_),
                            (int)(atom.lAff.y() - gapExtend_));
            mch = atom.lScr.x() + rowScores.y();

            atom.rScr.y() = MAX4(0, mch, del, ins);
            atom.rAff.y() = ins;

            del = sycl::max((int)(atom.rScr.y() - gapOpen_), (int)(del - gapExtend_));
            ins = sycl::max((int)(atom.lScr.z() - gapOpen_),
                            (int)(atom.lAff.z() - gapExtend_));
            mch = atom.lScr.y() + rowScores.z();

            atom.rScr.z() = MAX4(0, mch, del, ins);
            atom.rAff.z() = ins;

            del = sycl::max((int)(atom.rScr.z() - gapOpen_), (int)(del - gapExtend_));
            ins = sycl::max((int)(atom.lScr.w() - gapOpen_),
                            (int)(atom.lAff.w() - gapExtend_));
            mch = atom.lScr.z() + rowScores.w();

            atom.rScr.w() = MAX4(0, mch, del, ins);
            atom.rAff.w() = ins;

            score = sycl::max(score, atom.rScr.x());
            score = sycl::max(score, atom.rScr.y());
            score = sycl::max(score, atom.rScr.z());
            score = sycl::max(score, atom.rScr.w());

            atom.mch = atom.up.x();
            VEC4_ASSIGN(atom.lScr, atom.rScr);
            VEC4_ASSIGN(atom.lAff, atom.rAff);
        }

        /*
        DPCT1065:356: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        if (valid) {
            if (iter < iters_ - 1 &&
                item_ct1.get_local_id(2) == item_ct1.get_local_range(2) - 1) {
                VEC2_ASSIGN(hBus[off + col], sycl::int2(atom.rScr.w(), del));
            } else {
                hBusScrShr[item_ct1.get_local_id(2) + 1] = atom.rScr.w();
                hBusAffShr[item_ct1.get_local_id(2) + 1] = del;
            }
        }
        
        col++;
        
        if (col == cols) {

            col = 0;
            row += item_ct1.get_local_range(2) * 4;
            iter++;
                    
            atom.mch = 0;
            atom.lScr = INT4_ZERO;
            atom.lAff = INT4_SCORE_MIN;
        }

        /*
        DPCT1065:357: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    // write all scores
    scoresShr[item_ct1.get_local_id(2)] = score;
    /*
    DPCT1065:355: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // gather scores
    if (item_ct1.get_local_id(2) == 0) {

        for (int i = 1; i < item_ct1.get_local_range(2); ++i) {
            score = sycl::max(score, scoresShr[i]);
        }
    
        scores[id] = score;
    }
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// QUERY PROFILE

static QueryProfile* createQueryProfile(Chain* query, Scorer* scorer) {

    int rows = chainGetLength(query);
    int rowsGpu = rows + (8 - rows % 8) % 8;
    
    int width = scorerGetMaxCode(scorer) + 1;
    int height = rowsGpu / 4;

    char* row = (char*) malloc(rows * sizeof(char));
    chainCopyCodes(query, row);

    size_t size = width * height * sizeof(sycl::char4);
    sycl::char4 *data = (sycl::char4 *)malloc(size);
    memset(data, 0, size);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width - 1; ++j) {
            sycl::char4 scr;
            scr.x() = i * 4 + 0 >= rows ? 0 : scorerScore(scorer, row[i * 4 + 0], j);
            scr.y() = i * 4 + 1 >= rows ? 0 : scorerScore(scorer, row[i * 4 + 1], j);
            scr.z() = i * 4 + 2 >= rows ? 0 : scorerScore(scorer, row[i * 4 + 2], j);
            scr.w() = i * 4 + 3 >= rows ? 0 : scorerScore(scorer, row[i * 4 + 3], j);
            data[i * width + j] = scr;
        }
    }
    
    free(row);
    
    QueryProfile* queryProfile = (QueryProfile*) malloc(sizeof(QueryProfile));
    queryProfile->data = data;
    queryProfile->width = width;
    queryProfile->height = height;
    queryProfile->length = rows;
    queryProfile->size = size;
    
    return queryProfile;
}

static void deleteQueryProfile(QueryProfile* queryProfile) {
    free(queryProfile->data);
    free(queryProfile);
}

static QueryProfileGpu *createQueryProfileGpu(QueryProfile *queryProfile) try {

    int width = queryProfile->width;
    int height = queryProfile->height;
    
    size_t size = queryProfile->size;
    sycl::char4 *data = queryProfile->data;
    dpct::image_matrix *dataGpu;

    /*
    DPCT1003:358: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((dataGpu = new dpct::image_matrix(
                        qpTexture.get_channel(), sycl::range<2>(width, height)),
                    0));
    /*
    DPCT1003:359: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL(
        (dpct::dpct_memcpy(dataGpu->to_pitched_data(), sycl::id<3>(0, 0, 0),
                           dpct::pitched_data(data, size, size, 1),
                           sycl::id<3>(0, 0, 0), sycl::range<3>(size, 1, 1)),
         0));
    /*
    DPCT1003:360: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((qpTexture.attach(dataGpu), 0));
    qpTexture.set(sycl::addressing_mode::clamp_to_edge,
                  sycl::filtering_mode::nearest,
                  sycl::coordinate_normalization_mode::unnormalized);

    size_t queryProfileGpuSize = sizeof(QueryProfileGpu);
    QueryProfileGpu* queryProfileGpu = (QueryProfileGpu*) malloc(queryProfileGpuSize);
    queryProfileGpu->data = dataGpu;
    
    return queryProfileGpu;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void deleteQueryProfileGpu(QueryProfileGpu *queryProfileGpu) try {
    /*
    DPCT1003:361: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((delete queryProfileGpu->data, 0));
    /*
    DPCT1003:362: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((qpTexture.detach(), 0));
    free(queryProfileGpu);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

//------------------------------------------------------------------------------
//******************************************************************************

#endif // __CUDACC__

