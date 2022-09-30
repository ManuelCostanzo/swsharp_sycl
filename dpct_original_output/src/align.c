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

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "alignment.h"
#include "chain.h"
#include "constants.h"
#include "cpu_module.h"
#include "cuda_utils.h"
#include "error.h"
#include "gpu_module.h"
#include "reconstruct.h"
#include "scorer.h"
#include "threadpool.h"
#include "utils.h"

#include "align.h"

#define GPU_MIN_LEN     256
#define GPU_MIN_CELLS   1000000.0

#define HW_DATA             0
#define NW_DATA             1
#define OV_DATA             2
#define SW_DATA_SINGLE      3
#define SW_DATA_DUAL        4

typedef struct AlignData {
    int type;
    void* data;
} AlignData;

typedef struct ContextBest {
    Alignment** alignment;
    int type;
    Chain** queries;
    int queriesLen;
    Chain* target;
    Scorer* scorer;
    int* cards;
    int cardsLen;
} ContextBest;

typedef struct ContextPair {
    Alignment** alignment;
    int type;
    Chain* query;
    Chain* target;
    Scorer* scorer;
    int score;
    int* cards;
    int cardsLen;
} ContextPair;

typedef struct ContextScore {
    int* score;
    AlignData** data;
    int type;
    Chain* query;
    Chain* target;
    Scorer* scorer;
    int* cards;
    int cardsLen;
} ContextScore;

typedef struct HwData {
    int score;
    int queryEnd;
    int targetEnd;
} HwData;

typedef struct NwData {
    int score;
} NwData;

typedef struct OvData {
    int score;
    int queryEnd;
    int targetEnd;
} OvData;

typedef struct SwDataSingle {
    int score;
    int queryEnd;
    int targetEnd;
} SwDataSingle;

typedef struct SwDataDual {
    int score;
    int middleScore;
    int middleScoreUp;
    int middleScoreDown;
    int row;
    int col;
    int gap;
    int upScore;
    int upQueryEnd;
    int upTargetEnd;
    int downScore;
    int downQueryEnd;
    int downTargetEnd;
} SwDataDual;

typedef struct OvFindScoreSpecificContext {
    int* queryStart;
    int* targetStart;
    Chain* query;
    Chain* target;
    Scorer* scorer;
    int score;
} OvFindScoreSpecificContext;

typedef struct NwFindScoreSpecificContext {
    int* queryStart;
    int* targetStart;
    Chain* query;
    int queryFrontGap;
    Chain* target;
    Scorer* scorer;
    int score;
} NwFindScoreSpecificContext;

//******************************************************************************
// PUBLIC

extern void alignPair(Alignment** alignment, int type, Chain* query, 
    Chain* target, Scorer* scorer, int* cards, int cardsLen, Thread* thread);

extern void alignScoredPair(Alignment** alignment, int type, Chain* query, 
    Chain* target, Scorer* scorer, int score, int* cards, int cardsLen, 
    Thread* thread);

extern void alignBest(Alignment** alignment, int type, Chain** queries, 
    int queriesLen, Chain* target, Scorer* scorer, int* cards, int cardsLen, 
    Thread* thread);

extern void scorePair(int* score, int type, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen, Thread* thread);
    
//******************************************************************************

//******************************************************************************
// PRIVATE

static void* alignPairThread(void* param);

static void* alignBestThread(void* param);

static void* scorePairThread(void* param);

static int scorePairGpu(AlignData** data, int type, Chain* query, Chain* target, 
    Scorer* scorer, int score, int* cards, int cardsLen);
    
static void reconstructPairGpu(Alignment** alignment, AlignData* data, int type, 
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen);

// hw
static int hwScorePairGpu(AlignData** data, Chain* query, Chain* target, 
    Scorer* scorer, int score, int* cards, int cardsLen);
    
static void hwReconstructPairGpu(Alignment** alignment, AlignData* data, 
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen);
    
// nw
static int nwScorePairGpu(AlignData** data, Chain* query, Chain* target, 
    Scorer* scorer, int score, int* cards, int cardsLen);

static void nwReconstructPairGpu(Alignment** alignment, AlignData* data, 
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen);
    
static void nwFindScoreSpecific(int* queryStart, int* targetStart, Chain* query, 
    int queryFrontGap, Chain* target, Scorer* scorer, int score, int card,
    Thread* thread);

static void* nwFindScoreSpecificThread(void* param);

// ov
static int ovScorePairGpu(AlignData** data, Chain* query, Chain* target, 
    Scorer* scorer, int score, int* cards, int cardsLen);
    
static void ovReconstructPairGpu(Alignment** alignment, AlignData* data, 
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen);

static void ovFindScoreSpecific(int* queryStart, int* targetStart, Chain* query, 
    Chain* target, Scorer* scorer, int score, int card, Thread* thread);

static void* ovFindScoreSpecificThread(void* param);

// sw
static int swScorePairGpuSingle(AlignData** data, Chain* query, Chain* target, 
    Scorer* scorer, int score, int* cards, int cardsLen);

static void swReconstructPairGpuSingle(Alignment** alignment, AlignData* data,
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen);
    
static int swScorePairGpuDual(AlignData** data, Chain* query, Chain* target, 
    Scorer* scorer, int score, int* cards, int cardsLen);

static void swReconstructPairGpuDual(Alignment** alignment, AlignData* data, 
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen);
    
//******************************************************************************

//******************************************************************************
// PUBLIC

extern void alignPair(Alignment** alignment, int type, Chain* query, 
    Chain* target, Scorer* scorer, int* cards, int cardsLen, Thread* thread) {
    alignScoredPair(alignment, type, query, target, scorer, NO_SCORE, cards, 
        cardsLen, thread);
}

extern void alignScoredPair(Alignment** alignment, int type, Chain* query, 
    Chain* target, Scorer* scorer, int score, int* cards, int cardsLen, 
    Thread* thread) {
   
    ContextPair* param = (ContextPair*) malloc(sizeof(ContextPair));

    param->alignment = alignment;
    param->type = type;
    param->query = query;
    param->target = target;
    param->scorer = scorer;
    param->score = score;
    param->cards = cards;
    param->cardsLen = cardsLen;

    if (thread == NULL) {
        alignPairThread(param);
    } else {
        threadCreate(thread, alignPairThread, (void*) param);
    }
}

extern void alignBest(Alignment** alignment, int type, Chain** queries, 
    int queriesLen, Chain* target, Scorer* scorer, int* cards, int cardsLen, 
    Thread* thread) {
    
    // reduce problem to simple pair align
    if (queriesLen == 1) {
        alignPair(alignment, type, queries[0], target, scorer, cards, 
            cardsLen, thread);
        return;
    }
    
    ContextBest* param = (ContextBest*) malloc(sizeof(ContextBest));

    param->alignment = alignment;
    param->type = type;
    param->queries = queries;
    param->queriesLen = queriesLen;
    param->target = target;
    param->scorer = scorer;
    param->cards = cards;
    param->cardsLen = cardsLen;
    
    if (thread == NULL) {
        alignBestThread(param);
    } else {
        threadCreate(thread, alignBestThread, (void*) param);
    }
}

extern void scorePair(int* score, int type, Chain* query, Chain* target, 
    Scorer* scorer, int* cards, int cardsLen, Thread* thread) {
    
    ContextScore* param = (ContextScore*) malloc(sizeof(ContextScore));

    param->score = score;
    param->data = NULL; // not needed
    param->type = type;
    param->query = query;
    param->target = target;
    param->scorer = scorer;
    param->cards = cards;
    param->cardsLen = cardsLen;

    if (thread == NULL) {
        scorePairThread(param);
    } else {
        threadCreate(thread, scorePairThread, (void*) param);
    }
}

//******************************************************************************
    
//******************************************************************************
// PRIVATE

//------------------------------------------------------------------------------
// ENTRY

static void* alignPairThread(void* param) {

    ContextPair* context = (ContextPair*) param;
    
    Alignment** alignment = context->alignment;
    int type = context->type;
    Chain* query = context->query;
    Chain* target = context->target;
    Scorer* scorer = context->scorer;
    int score = context->score;
    int* cards = context->cards;
    int cardsLen = context->cardsLen;
    
    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    double cells = (double) rows * cols;

    if (cols < GPU_MIN_LEN || cells < GPU_MIN_CELLS || cardsLen == 0) {
        if (score == NO_SCORE) {
            alignPairCpu(alignment, type, query, target, scorer);
        } else {
            alignScoredPairCpu(alignment, type, query, target, scorer, score);
        }
    } else {
    
        AlignData* data;
        scorePairGpu(&data, type, query, target, scorer, score, cards, cardsLen);

        reconstructPairGpu(alignment, data, type, query, target, scorer, 
            cards, cardsLen);

        free(data->data);
        free(data);        
    }
    
    free(param);
    
    return NULL;
}

static void* alignBestThread(void* param) {

    ContextBest* context = (ContextBest*) param;

    Alignment** alignment = context->alignment;
    int type = context->type;
    Chain** queries = context->queries;
    int queriesLen = context->queriesLen;
    Chain* target = context->target;
    Scorer* scorer = context->scorer;
    int* cards = context->cards;
    int cardsLen = context->cardsLen;

    int i, j;

    //**************************************************************************
    // SCORE MULTITHREADED
    
    AlignData** data = (AlignData**) malloc(queriesLen * sizeof(AlignData*));
    int* scores = (int*) malloc(queriesLen * sizeof(int));
    
    size_t contextsSize = queriesLen * sizeof(ContextScore);
    ContextScore** contextsCpu = (ContextScore**) malloc(contextsSize);
    ContextScore** contextsGpu = (ContextScore**) malloc(contextsSize);
    int contextsCpuLen = 0;
    int contextsGpuLen = 0;

    size_t tasksSize = queriesLen * sizeof(ThreadPoolTask*);
    ThreadPoolTask** tasks = (ThreadPoolTask**) malloc(tasksSize);

    int cols = chainGetLength(target);
    for (i = 0; i < queriesLen; ++i) {
    
        int rows = chainGetLength(queries[i]);
        double cells = (double) rows * cols;
    
        ContextScore* context = (ContextScore*) malloc(sizeof(ContextScore));
        context->score = &(scores[i]);
        context->data = &(data[i]);;
        context->type = type;
        context->query = queries[i];
        context->target = target;
        context->scorer = scorer;

        if (cols < GPU_MIN_LEN || cells < GPU_MIN_CELLS || cardsLen == 0) {
            contextsCpu[contextsCpuLen++] = context;
            context->cards = NULL;
            context->cardsLen = 0;
        } else {
            contextsGpu[contextsGpuLen++] = context;
        }
    }

    for (i = 0; i < contextsCpuLen; ++i) {
        tasks[i] = threadPoolSubmit(scorePairThread, (void*) contextsCpu[i]);
    }
    
    if (contextsGpuLen) {

        int buckets = MIN(contextsGpuLen, cardsLen);
        int** cardBuckets;
        int* cardBucketsLens;
        chunkArray(&cardBuckets, &cardBucketsLens, cards, cardsLen, buckets);
    
        i = 0;
        while (i < contextsGpuLen) {
        
            for (j = 0; j < buckets && i + j < contextsGpuLen; ++j) {
                
                ContextScore* context = contextsGpu[i + j];
                context->cards = cardBuckets[j];
                context->cardsLen = cardBucketsLens[j];

                ThreadPoolTask* task = 
                    threadPoolSubmit(scorePairThread, (void*) context);
                    
                tasks[contextsCpuLen + j] = task;
            }
            
            for (j = 0; j < buckets && i < contextsGpuLen; ++j, ++i) {
                threadPoolTaskWait(tasks[contextsCpuLen + j]);
                threadPoolTaskDelete(tasks[contextsCpuLen + j]);
                free(contextsGpu[i]);
            }
        }
        
        free(cardBuckets);
        free(cardBucketsLens);
    }

    // wait for cpu tasks
    for (i = 0; i < contextsCpuLen; ++i) {
        threadPoolTaskWait(tasks[i]);
        threadPoolTaskDelete(tasks[i]);
        free(contextsCpu[i]);
    }
    
    free(contextsCpu);
    free(contextsGpu);
    free(tasks);

    //**************************************************************************

    //**************************************************************************
    // FIND AND ALIGN THE BEST

    int maxScore = scores[0];
    int maxIdx = 0;
    
    for (i = 1; i < queriesLen; ++i) {
        if (scores[i] > maxScore) {
            maxScore = scores[i];
            maxIdx = i;
        }
    }
    
    if (data[maxIdx] == NULL) {
        alignScoredPairCpu(alignment, type, queries[maxIdx], target,
            scorer, maxScore);
    } else {
        reconstructPairGpu(alignment, data[maxIdx], type, queries[maxIdx], 
            target, scorer, cards, cardsLen);
    }

    //**************************************************************************
    
    //**************************************************************************
    // CLEAN MEMORY

    for (i = 0; i < queriesLen; ++i) {
        if (data[i] != NULL) {
            free(data[i]->data);
            free(data[i]);
        }
    }
    
    free(data);
    free(scores);
    
    free(param);

    //**************************************************************************
    
    return NULL;
}

static void* scorePairThread(void* param) {

    ContextScore* context = (ContextScore*) param;

    int* score = context->score;
    AlignData** data = context->data;
    int type = context->type;
    Chain* query = context->query;
    Chain* target = context->target;
    Scorer* scorer = context->scorer;
    int* cards = context->cards;
    int cardsLen = context->cardsLen;

    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    double cells = (double) rows * cols;
    
    if (cols < GPU_MIN_LEN || cells < GPU_MIN_CELLS || cardsLen == 0) {
        *score = scorePairCpu(type, query, target, scorer);
        if (data != NULL) *data = NULL;
    } else {
        *score = scorePairGpu(data, type, query, target, scorer, NO_SCORE, 
            cards, cardsLen);
    }
    
    return NULL;
}

static int scorePairGpu(AlignData** data, int type, Chain* query, Chain* target, 
    Scorer* scorer, int score, int* cards, int cardsLen) {

    int dual = cardsLen >= 2;
    
    int (*function) (AlignData**, Chain*, Chain*, Scorer*, int, int*, int);
    
    switch (type) {
    case HW_ALIGN:
        function = hwScorePairGpu;
        break;
    case NW_ALIGN:
        function = nwScorePairGpu;
        break;
    case OV_ALIGN:
        function = ovScorePairGpu;
        break;
    case SW_ALIGN:
        if (dual) {
            function = swScorePairGpuDual;
        } else {
            function = swScorePairGpuSingle;
        }
        break;
    default:
        ERROR("invalid align type");
    }
    
    return function(data, query, target, scorer, score, cards, cardsLen);
}
    
static void reconstructPairGpu(Alignment** alignment, AlignData* data, int type, 
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen) {

    int dataType = data->type;
    
    void (*function) (Alignment**, AlignData*, Chain*, Chain*, Scorer*, int*, int);
    
    switch (dataType) {
    case HW_DATA:
        function = hwReconstructPairGpu;
        break;
    case NW_DATA:
        function = nwReconstructPairGpu;
        break;
    case OV_DATA:
        function = ovReconstructPairGpu;
        break;
    case SW_DATA_SINGLE:
        function = swReconstructPairGpuSingle;
        break;
    case SW_DATA_DUAL:
        function = swReconstructPairGpuDual;
        break;
    default:
        ERROR("invalid align type");
    }
    
    function(alignment, data, query, target, scorer, cards, cardsLen);
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// HW

static int hwScorePairGpu(AlignData** data_, Chain* query, Chain* target, 
    Scorer* scorer, int score, int* cards, int cardsLen) {
    
    int card = cards[0];
    
    int queryEnd;
    int targetEnd;
    int outScore;

    hwEndDataGpu(&queryEnd, &targetEnd, &outScore, query, target, scorer, 
        score, card, NULL);

    ASSERT(outScore == score || score == NO_SCORE, "invalid alignment input score");
    ASSERT(queryEnd == chainGetLength(query) - 1, "invalid hw alignment");
    
    if (data_ != NULL) {
    
        HwData* data = (HwData*) malloc(sizeof(HwData));
        data->score = outScore;
        data->queryEnd = queryEnd;
        data->targetEnd = targetEnd;
        
        AlignData* alignData = (AlignData*) malloc(sizeof(AlignData));
        alignData->type = HW_DATA;
        alignData->data = data;
        
        *data_ = alignData;
    }

    return outScore;
}
    
static void hwReconstructPairGpu(Alignment** alignment, AlignData* data_, 
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen) {
    
    AlignData* alignData = (AlignData*) data_;
    ASSERT(alignData->type == HW_DATA, "wrong align data type");

    HwData* data = (HwData*) alignData->data;
    
    int score = data->score;
    int queryEnd = data->queryEnd;
    int targetEnd = data->targetEnd;
    
    int card = cards[0];
    
    // find the start      
    Chain* queryFind = chainCreateView(query, 0, queryEnd, 1);
    Chain* targetFind = chainCreateView(target, 0, targetEnd, 1);

    if (chainGetLength(targetFind) < GPU_MIN_LEN) {

        chainDelete(queryFind);
        chainDelete(targetFind);

        Chain* queryCpu = chainCreateView(query, 0, queryEnd, 0);
        Chain* targetCpu = chainCreateView(target, 0, targetEnd, 0);

        alignScoredPairCpu(alignment, HW_ALIGN, queryCpu, targetCpu, scorer, score);

        chainDelete(queryCpu);
        chainDelete(targetCpu);

        return;
    }

    int* scores;
    nwLinearDataGpu(&scores, NULL, queryFind, 0, targetFind, 0, scorer, -1, -1, 
        card, NULL);
    
    int targetFindLen = chainGetLength(targetFind);
    
    chainDelete(queryFind);
    chainDelete(targetFind);
    
    int queryStart = 0;
    int targetStart = -1;
    
    int i;
    for (i = 0; i < targetFindLen; ++i) {
        if (scores[i] == score) {
            targetStart = targetFindLen - 1 - i;
            break;
        }
    }
   
    free(scores);
    
    ASSERT(targetStart != -1, "invalid hybrid find"); 

    // reconstruct
    char* path;
    int pathLen;
    
    Chain* queryRecn = chainCreateView(query, queryStart, queryEnd, 0);
    Chain* targetRecn = chainCreateView(target, targetStart, targetEnd, 0);
    
    nwReconstruct(&path, &pathLen, NULL, queryRecn, 0, 0, targetRecn, 0, 0, 
        scorer, score, cards, cardsLen, NULL);
     
    chainDelete(queryRecn);
    chainDelete(targetRecn);  
         
    *alignment = alignmentCreate(query, queryStart, queryEnd, target, 
        targetStart, targetEnd, score, scorer, path, pathLen);
}
   
//------------------------------------------------------------------------------
 
//------------------------------------------------------------------------------
// NW

static int nwScorePairGpu(AlignData** data_, Chain* query, Chain* target, 
    Scorer* scorer, int score, int* cards, int cardsLen) {
    
    ASSERT(!(score != NO_SCORE && data_ == NULL), "invalid score data");

    int outScore = NO_SCORE;

    if (data_ != NULL) {
    
        // score will be available through reconstruction, no need to score

        NwData* data = (NwData*) malloc(sizeof(NwData));
        data->score = outScore;

        AlignData* alignData = (AlignData*) malloc(sizeof(AlignData));
        alignData->type = NW_DATA;
        alignData->data = data;
    
        *data_ = alignData;

    } else {

        int* scores;
        
        nwLinearDataGpu(&scores, NULL, query, 0, target, 0, scorer, -1, -1, 
            cards[0], NULL);
            
        outScore = scores[chainGetLength(target) - 1];
        free(scores);
    }
    
    return score;
}

static void nwReconstructPairGpu(Alignment** alignment, AlignData* data_,
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen) {
    
    AlignData* alignData = (AlignData*) data_;
    ASSERT(alignData->type == NW_DATA, "wrong align data type");
    
    NwData* data = (NwData*) alignData->data;
    int score = data->score;
    
    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    
    char* path;
    int pathLen;
    
    nwReconstruct(&path, &pathLen, &score, query, 0, 0, target, 0, 0, 
        scorer, score, cards, cardsLen, NULL);
    
    *alignment = alignmentCreate(query, 0, rows - 1, target, 0, cols - 1, 
        score, scorer, path, pathLen);
}

static void nwFindScoreSpecific(int* queryStart, int* targetStart, Chain* query, 
    int queryFrontGap, Chain* target, Scorer* scorer, int score, int card,
    Thread* thread) {
    
    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    
    double cells = (double) rows * cols;
    
    if (cols < GPU_MIN_LEN || cells < GPU_MIN_CELLS) {
        if (thread == NULL) {
            nwFindScoreCpu(queryStart, targetStart, query, queryFrontGap, target,
                scorer, score);
        } else {

            NwFindScoreSpecificContext* context = 
                (NwFindScoreSpecificContext*) malloc(sizeof(NwFindScoreSpecificContext));

            context->queryStart = queryStart;
            context->targetStart = targetStart;
            context->query = query;
            context->queryFrontGap = queryFrontGap;
            context->target = target;
            context->scorer = scorer;
            context->score = score;

            threadCreate(thread, nwFindScoreSpecificThread, (void*) context);
        }
    } else {
        nwFindScoreGpu(queryStart, targetStart, query, queryFrontGap, target, 
            scorer, score, card, thread);
    }
    
    ASSERT(*queryStart != -1, "Score not found %d (%s) (%s)", score,
        chainGetName(query), chainGetName(target));
}

static void* nwFindScoreSpecificThread(void* param) {

    NwFindScoreSpecificContext* context = (NwFindScoreSpecificContext*) param;

    int* queryStart = context->queryStart;
    int* targetStart = context->targetStart;
    Chain* query = context->query;
    int queryFrontGap = context->queryFrontGap;
    Chain* target = context->target;
    Scorer* scorer = context->scorer;
    int score = context->score;

    nwFindScoreCpu(queryStart, targetStart, query, queryFrontGap, target, scorer, score);

    free(param);

    return NULL;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// OV

static int ovScorePairGpu(AlignData** data_, Chain* query, Chain* target, 
    Scorer* scorer, int score, int* cards, int cardsLen) {
    
    int card = cards[0];
    
    int queryEnd;
    int targetEnd;
    int outScore;

    ovEndDataGpu(&queryEnd, &targetEnd, &outScore, query, target, scorer, 
        score, card, NULL);

    int lastRow = queryEnd == chainGetLength(query) - 1;
    int lastCol = targetEnd == chainGetLength(target) - 1;
    
    ASSERT(outScore == score || score == NO_SCORE, "invalid alignment input score %s %s",
            chainGetName(query), chainGetName(target));
    ASSERT(lastRow || lastCol, "invalid ov alignment");
    
    if (data_ != NULL) {
    
        OvData* data = (OvData*) malloc(sizeof(OvData));
        data->score = outScore;
        data->queryEnd = queryEnd;
        data->targetEnd = targetEnd;
        
        AlignData* alignData = (AlignData*) malloc(sizeof(AlignData));
        alignData->type = OV_DATA;
        alignData->data = data;
        
        *data_ = alignData;
    }

    return outScore;
}
    
static void ovReconstructPairGpu(Alignment** alignment, AlignData* data_, 
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen) {
    
    AlignData* alignData = (AlignData*) data_;
    ASSERT(alignData->type == OV_DATA, "wrong align data type");
    
    OvData* data = (OvData*) alignData->data;
    
    int score = data->score;
    int queryEnd = data->queryEnd;
    int targetEnd = data->targetEnd;
    
    // find the start
    int card = cards[0];
    
    Chain* queryFind = chainCreateView(query, 0, queryEnd, 1);
    Chain* targetFind = chainCreateView(target, 0, targetEnd, 1);

    int queryStart;
    int targetStart;

    ovFindScoreSpecific(&queryStart, &targetStart, queryFind, targetFind, 
        scorer, score, card, NULL);

    queryStart = chainGetLength(queryFind) - queryStart - 1;
    targetStart = chainGetLength(targetFind) - targetStart - 1;

    chainDelete(queryFind);
    chainDelete(targetFind);
    
    Chain* queryRecn = chainCreateView(query, queryStart, queryEnd, 0);
    Chain* targetRecn = chainCreateView(target, targetStart, targetEnd, 0);
    
    int pathLen;
    char* path;
    
    nwReconstruct(&path, &pathLen, NULL, queryRecn, 0, 0, targetRecn, 0, 0, 
        scorer, score, cards, cardsLen, NULL);
    
    chainDelete(queryRecn);
    chainDelete(targetRecn);

    ASSERT(queryStart == 0 || targetStart == 0, "invalid ov alignment");

    *alignment = alignmentCreate(query, queryStart, queryEnd, target, 
        targetStart, targetEnd, score, scorer, path, pathLen);
}

static void ovFindScoreSpecific(int* queryStart, int* targetStart, Chain* query, 
    Chain* target, Scorer* scorer, int score, int card, Thread* thread) {
    
    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    
    double cells = (double) rows * cols;
    
    if (cols < GPU_MIN_LEN || cells < GPU_MIN_CELLS) {
        if (thread == NULL) {
            ovFindScoreCpu(queryStart, targetStart, query, target, scorer, score);
        } else {

            OvFindScoreSpecificContext* context = 
                (OvFindScoreSpecificContext*) malloc(sizeof(OvFindScoreSpecificContext));

            context->queryStart = queryStart;
            context->targetStart = targetStart;
            context->query = query;
            context->target = target;
            context->scorer = scorer;
            context->score = score;

            threadCreate(thread, ovFindScoreSpecificThread, (void*) context);
        }
    } else {
        ovFindScoreGpu(queryStart, targetStart, query, target, scorer, score, 
            card, thread);
    }
    
    ASSERT(*queryStart != -1, "Score not found %d %s %s", score, 
        chainGetName(query), chainGetName(target));
}

static void* ovFindScoreSpecificThread(void* param) {

    OvFindScoreSpecificContext* context = (OvFindScoreSpecificContext*) param;

    int* queryStart = context->queryStart;
    int* targetStart = context->targetStart;
    Chain* query = context->query;
    Chain* target = context->target;
    Scorer* scorer = context->scorer;
    int score = context->score;

    ovFindScoreCpu(queryStart, targetStart, query, target, scorer, score);

    free(param);

    return NULL;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// SW

static int swScorePairGpuSingle(AlignData** data_, Chain* query, Chain* target, 
    Scorer* scorer, int score, int* cards, int cardsLen) {
    
    int card = cards[0];
    
    int queryEnd;
    int targetEnd;
    int outScore;

    swEndDataGpu(&queryEnd, &targetEnd, &outScore, NULL, NULL, query, target, 
        scorer, score, card, NULL);

    ASSERT(outScore == score || score == NO_SCORE,
        "invalid alignment input score %d %d | %s %s",
        outScore, score, chainGetName(query), chainGetName(target));

    if (data_ != NULL) {
    
        SwDataSingle* data = (SwDataSingle*) malloc(sizeof(SwDataSingle));
        data->score = outScore;
        data->queryEnd = queryEnd;
        data->targetEnd = targetEnd;

        AlignData* alignData = (AlignData*) malloc(sizeof(AlignData));
        alignData->type = SW_DATA_SINGLE;
        alignData->data = data;
    
        *data_ = alignData;
    }
    
    return outScore;
}

static void swReconstructPairGpuSingle(Alignment** alignment, AlignData* data_,
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen) {
  
    AlignData* alignData = (AlignData*) data_;
    ASSERT(alignData->type == SW_DATA_SINGLE, "wrong align data type");
    
    SwDataSingle* data = (SwDataSingle*) alignData->data;
    int score = data->score;
    int queryEnd = data->queryEnd;
    int targetEnd = data->targetEnd;
    
    if (score == 0) {
        *alignment = alignmentCreate(query, 0, 0, target, 0, 0, 0, scorer, NULL, 0);
        return;
    }

    int card = cards[0];
    
    Chain* queryFind = chainCreateView(query, 0, queryEnd, 1);
    Chain* targetFind = chainCreateView(target, 0, targetEnd, 1);

    int queryStart;
    int targetStart;

    nwFindScoreSpecific(&queryStart, &targetStart, queryFind, 0, targetFind, 
        scorer, score, card, NULL);

    queryStart = chainGetLength(queryFind) - queryStart - 1;
    targetStart = chainGetLength(targetFind) - targetStart - 1;

    chainDelete(queryFind);
    chainDelete(targetFind);
    
    Chain* queryRecn = chainCreateView(query, queryStart, queryEnd, 0);
    Chain* targetRecn = chainCreateView(target, targetStart, targetEnd, 0);
    
    int pathLen;
    char* path;
    
    nwReconstruct(&path, &pathLen, NULL, queryRecn, 0, 0, targetRecn, 0, 0, 
        scorer, score, cards, cardsLen, NULL);
    
    chainDelete(queryRecn);
    chainDelete(targetRecn);
    
    *alignment = alignmentCreate(query, queryStart, queryEnd, target, 
        targetStart, targetEnd, score, scorer, path, pathLen);
}

static int swScorePairGpuDual(AlignData** data_, Chain* query, Chain* target, 
    Scorer* scorer, int score, int* cards, int cardsLen) {

    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    
    int row = rows / 2;

    int* upScores; 
    int* upAffines;
    int upQueryEnd;
    int upTargetEnd;
    int upScore;
    Chain* upRow = chainCreateView(query, 0, row, 0);
    Chain* upCol = chainCreateView(target, 0, cols - 1, 0);

    int* downScores; 
    int* downAffines;
    int downQueryEnd;
    int downTargetEnd;
    int downScore;
    Chain* downRow = chainCreateView(query, row + 1, rows - 1, 1);
    Chain* downCol = chainCreateView(target, 0, cols - 1, 1);

    if (cardsLen == 1) {
    
        swEndDataGpu(&upQueryEnd, &upTargetEnd, &upScore, &upScores, &upAffines,
            upRow, upCol, scorer, NO_SCORE, cards[0], NULL);
            
        swEndDataGpu(&downQueryEnd, &downTargetEnd, &downScore, &downScores, 
            &downAffines, downRow, downCol, scorer, NO_SCORE, cards[0], NULL);
            
    } else {
    
        Thread thread;
        
        swEndDataGpu(&upQueryEnd, &upTargetEnd, &upScore, &upScores, &upAffines,
            upRow, upCol, scorer, NO_SCORE, cards[1], &thread);
            
        swEndDataGpu(&downQueryEnd, &downTargetEnd, &downScore, &downScores, 
            &downAffines, downRow, downCol, scorer, NO_SCORE, cards[0], NULL);
            
        threadJoin(thread); 
    }
    
    chainDelete(upRow);
    chainDelete(upCol);
    chainDelete(downCol);
    chainDelete(downRow);
    
    if (upScore == 0 || downScore == 0) {
        *data_ = NULL;
        return 0;
    }
    
    int middleScore = INT_MIN;
    int middleScoreUp = 0;
    int middleScoreDown = 0;
    int gap = 0; // boolean
    int col = -1;

    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);
    int gapDiff = gapOpen - gapExtend;

    int up, down;
    for(up = 0, down = cols - 2; up < cols - 1; ++up, --down) {
    
        int scr = upScores[up] + downScores[down];
        int aff = upAffines[up] + downAffines[down] + gapDiff;

        if (scr > middleScore) {
            middleScoreUp = upScores[up];
            middleScoreDown = downScores[down];
            middleScore = scr;
            gap = 0;
            col = up;   
        }

        if (aff >= middleScore) {
            middleScoreUp = upAffines[up] + gapDiff;
            middleScoreDown = downAffines[down] + gapDiff;
            middleScore = aff;
            gap = 1;
            col = up;   
        }
    }
    
    free(upScores);
    free(upAffines);
    free(downScores);
    free(downAffines);

    int outScore = MAX(middleScore, MAX(upScore, downScore));
    
    ASSERT(outScore == score || score == NO_SCORE, "invalid alignment input score");
    
    LOG("Scores | up: %d | down: %d | mid: %d", upScore, downScore, middleScore);
    
    if (data_ != NULL) {
    
        SwDataDual* data = (SwDataDual*) malloc(sizeof(SwDataDual));
        data->score = outScore;
        data->middleScore = middleScore;
        data->middleScoreUp = middleScoreUp;
        data->middleScoreDown = middleScoreDown;
        data->row = row;
        data->col = col;
        data->gap = gap;
        data->upScore = upScore;
        data->upQueryEnd = upQueryEnd;
        data->upTargetEnd = upTargetEnd;
        data->downScore = downScore;
        data->downQueryEnd = downQueryEnd;
        data->downTargetEnd = downTargetEnd;

        AlignData* alignData = (AlignData*) malloc(sizeof(AlignData));
        alignData->type = SW_DATA_DUAL;
        alignData->data = data;
        
        *data_ = alignData;
    }
    
    return outScore;
}

static void swReconstructPairGpuDual(Alignment** alignment, AlignData* data_,
    Chain* query, Chain* target, Scorer* scorer, int* cards, int cardsLen) {

    if (data_ == NULL) {
        *alignment = alignmentCreate(query, 0, 0, target, 0, 0, 0, scorer, NULL, 0);
        return;
    }

    AlignData* alignData = (AlignData*) data_;
    ASSERT(alignData->type == SW_DATA_DUAL, "wrong align data type");
    
    // extract data
    SwDataDual* data = (SwDataDual*) alignData->data;
    int score = data->score;
    int middleScore = data->middleScore;
    int middleScoreUp = data->middleScoreUp;
    int middleScoreDown = data->middleScoreDown;
    int row = data->row;
    int col = data->col;
    int gap = data->gap;
    int upScore = data->upScore;
    int upQueryEnd = data->upQueryEnd;
    int upTargetEnd = data->upTargetEnd;
    int downScore = data->downScore;
    int downQueryEnd = data->downQueryEnd;
    int downTargetEnd = data->downTargetEnd;
    
    Thread thread;
        
    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    
    int queryEnd;
    int targetEnd;
    
    int queryStart;
    int targetStart;
    
    int pathLen;
    char* path;
    
    if (score == upScore) {
    
        queryEnd = upQueryEnd;
        targetEnd = upTargetEnd;
        
        Chain* queryFind = chainCreateView(query, 0, queryEnd, 1);
        Chain* targetFind = chainCreateView(target, 0, targetEnd, 1);
        
        nwFindScoreSpecific(&queryStart, &targetStart, queryFind, 0, targetFind,
            scorer, score, cards[0], NULL);
            
        queryStart = chainGetLength(queryFind) - queryStart - 1;
        targetStart = chainGetLength(targetFind) - targetStart - 1;
        
        chainDelete(queryFind);
        chainDelete(targetFind);
        
        Chain* queryRecn = chainCreateView(query, queryStart, queryEnd, 0);
        Chain* targetRecn = chainCreateView(target, targetStart, targetEnd, 0);
    
        nwReconstruct(&path, &pathLen, NULL, queryRecn, 0, 0, targetRecn, 0, 0,
            scorer, score, cards, cardsLen, NULL);
            
        chainDelete(queryRecn);
        chainDelete(targetRecn);
        
    } else if (score == downScore) {
    
        queryStart = chainGetLength(query) - downQueryEnd - 1;
        targetStart = chainGetLength(target) - downTargetEnd - 1;
        
        Chain* queryFind = chainCreateView(query, queryStart, rows - 1, 0);
        Chain* targetFind = chainCreateView(target, targetStart, cols - 1, 0);
        
        nwFindScoreSpecific(&queryEnd, &targetEnd, queryFind, 0, targetFind,
            scorer, score, cards[0], NULL);
            
        queryEnd = queryStart + queryEnd;
        targetEnd = targetStart + targetEnd;
        
        chainDelete(queryFind);
        chainDelete(targetFind);
        
        Chain* queryRecn = chainCreateView(query, queryStart, queryEnd, 0);
        Chain* targetRecn = chainCreateView(target, targetStart, targetEnd, 0);
    
        nwReconstruct(&path, &pathLen, NULL, queryRecn, 0, 0, targetRecn, 0, 0,
            scorer, score, cards, cardsLen, NULL);
            
        chainDelete(queryRecn);
        chainDelete(targetRecn);

    } else if (score == middleScore) {
    
        Chain* upQueryFind = chainCreateView(query, 0, row, 1);
        Chain* upTargetFind = chainCreateView(target, 0, col, 1);
       
        Chain* downQueryFind = chainCreateView(query, row + 1, rows - 1, 0);
        Chain* downTargetFind = chainCreateView(target, col + 1, cols - 1, 0);
        
        if (cardsLen == 1) {
        
            nwFindScoreSpecific(&queryStart, &targetStart, upQueryFind, gap,
                upTargetFind, scorer, middleScoreUp, cards[0], NULL);
                
            nwFindScoreSpecific(&queryEnd, &targetEnd, downQueryFind, gap,
                downTargetFind, scorer, middleScoreDown, cards[0], NULL);
        
        } else {
        
            nwFindScoreSpecific(&queryStart, &targetStart, upQueryFind, gap,
                upTargetFind, scorer, middleScoreUp, cards[1], &thread);
                
            nwFindScoreSpecific(&queryEnd, &targetEnd, downQueryFind, gap,
                downTargetFind, scorer, middleScoreDown, cards[0], NULL);
                
            threadJoin(thread);
        }
        
        chainDelete(upQueryFind);
        chainDelete(upTargetFind);
        chainDelete(downQueryFind);
        chainDelete(downTargetFind);
        
        queryStart = row - queryStart;
        targetStart = col - targetStart;
        queryEnd += row + 1;
        targetEnd += col + 1;
        
        char* upPath;
        int upPathLen;
        Chain* upQueryRecn = chainCreateView(query, queryStart, row, 0);
        Chain* upTargetRecn = chainCreateView(target, targetStart, col, 0);
        
        char* downPath;
        int downPathLen;
        Chain* downQueryRecn = chainCreateView(query, row + 1, queryEnd, 0);
        Chain* downTargetRecn = chainCreateView(target, col + 1, targetEnd, 0);
        
        if (cardsLen == 1) {
        
            nwReconstruct(&upPath, &upPathLen, NULL, upQueryRecn, 0, gap, 
                upTargetRecn, 0, 0, scorer, middleScoreUp, cards, cardsLen, NULL);
                
            nwReconstruct(&downPath, &downPathLen, NULL, downQueryRecn, gap, 0,
                downTargetRecn, 0, 0, scorer, middleScoreDown,
                cards, cardsLen, NULL);
        
        } else {
        
            int half = cardsLen / 2;
            
            nwReconstruct(&upPath, &upPathLen, NULL, upQueryRecn, 0, gap,
                upTargetRecn, 0, 0, scorer, middleScoreUp, cards, half, &thread);
                
            nwReconstruct(&downPath, &downPathLen, NULL, downQueryRecn, gap, 0,
                downTargetRecn, 0, 0, scorer, middleScoreDown, cards + half,
                cardsLen - half, NULL);
                
            threadJoin(thread);
        }
        
        chainDelete(upQueryRecn);
        chainDelete(upTargetRecn);
        chainDelete(downQueryRecn);
        chainDelete(downTargetRecn);

        pathLen = upPathLen + downPathLen;
        path = (char*) malloc(pathLen * sizeof(char));

        memcpy(path, upPath, upPathLen * sizeof(char));
        memcpy(path + upPathLen, downPath, downPathLen * sizeof(char));
                
        free(upPath);
        free(downPath);
        
    } else {
        ERROR("invalid dual data score");
    }
    
    *alignment = alignmentCreate(query, queryStart, queryEnd, target, 
        targetStart, targetEnd, score, scorer, path, pathLen);
}
    
//------------------------------------------------------------------------------
//******************************************************************************
