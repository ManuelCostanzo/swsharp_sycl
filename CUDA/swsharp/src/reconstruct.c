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

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "chain.h"
#include "cpu_module.h"
#include "constants.h"
#include "error.h"
#include "gpu_module.h"
#include "scorer.h"
#include "thread.h"
#include "threadpool.h"
#include "utils.h"

#include "reconstruct.h"

#define MIN_DUAL_LEN 20000
#define MIN_BLOCK_SIZE 256 // > MINIMAL THREADS IN LINEAR_DATA * 2 !!!!
#define MAX_BLOCK_CELLS 5000000.0

typedef struct Context
{
    char **path;
    int *pathLen;
    int *outScore;
    Chain *query;
    int queryFrontGap; // boolean
    int queryBackGap;  // boolean
    Chain *target;
    int targetFrontGap; // boolean
    int targetBackGap;  // boolean
    Scorer *scorer;
    int score;
    int *cards;
    int cardsLen;
} Context;

typedef struct Block
{
    int outScore;
    int score;
    int queryFrontGap;  // boolean
    int queryBackGap;   // boolean
    int targetFrontGap; // boolean
    int targetBackGap;  // boolean
    int startRow;
    int startCol;
    int endRow;
    int endCol;
    char *path;
    int pathLen;
} Block;

typedef struct BlockContext
{
    Block *block;
    Chain *query;
    Chain *target;
    Scorer *scorer;
} BlockContext;

typedef struct BlocksData
{
    Semaphore mutex;
    Block **blocks;
    BlockContext **contexts;
    ThreadPoolTask **tasks;
    int length;
} BlocksData;

typedef struct HirschbergContext
{
    Block *block;
    BlocksData *blocksData;
    Chain *rowChain;
    Chain *colChain;
    Scorer *scorer;
    int *cards;
    int cardsLen;
} HirschbergContext;

//******************************************************************************
// PUBLIC

extern void nwReconstruct(char **path, int *pathLen, int *outScore,
                          Chain *query, int queryFrontGap, int queryBackGap, Chain *target,
                          int targetFrontGap, int targetBackGap, Scorer *scorer, int score,
                          int *cards, int cardsLen, Thread *thread);

//******************************************************************************

//******************************************************************************
// PRIVATE

static void *nwReconstructThread(void *param);
static void *hirschberg(void *param);

static void *blockReconstruct(void *params);
static int blockCmp(const void *a_, const void *b_);

//******************************************************************************

extern void nwReconstruct(char **path, int *pathLen, int *outScore,
                          Chain *query, int queryFrontGap, int queryBackGap, Chain *target,
                          int targetFrontGap, int targetBackGap, Scorer *scorer, int score,
                          int *cards, int cardsLen, Thread *thread)
{

    Context *param = (Context *)malloc(sizeof(Context));

    param->path = path;
    param->pathLen = pathLen;
    param->outScore = outScore;
    param->query = query;
    param->queryFrontGap = queryFrontGap;
    param->queryBackGap = queryBackGap;
    param->target = target;
    param->targetFrontGap = targetFrontGap;
    param->targetBackGap = targetBackGap;
    param->scorer = scorer;
    param->score = score;
    param->cards = cards;
    param->cardsLen = cardsLen;

    if (thread == NULL)
    {
        nwReconstructThread(param);
    }
    else
    {
        threadCreate(thread, nwReconstructThread, (void *)param);
    }
}

//******************************************************************************

//******************************************************************************
// PRIVATE

//------------------------------------------------------------------------------
// RECONSTRUCT

static void *nwReconstructThread(void *param)
{

    Context *context = (Context *)param;

    char **path = context->path;
    int *pathLen = context->pathLen;
    int *outScore = context->outScore;
    Chain *query = context->query;
    int queryFrontGap = context->queryFrontGap;
    int queryBackGap = context->queryBackGap;
    Chain *target = context->target;
    int targetFrontGap = context->targetFrontGap;
    int targetBackGap = context->targetBackGap;
    Scorer *scorer = context->scorer;
    int score = context->score;
    int *cards = context->cards;
    int cardsLen = context->cardsLen;

    int rows = chainGetLength(query);
    int cols = chainGetLength(target);

    TIMER_START("Reconstruction");

    int blockIdx;

    //**************************************************************************
    // PARALLEL CPU/GPU RECONSTRUCTION

    int blocksMaxLen = 1 + (MAX(rows, cols) * 2) / MIN_BLOCK_SIZE;

    BlocksData blocksData;
    semaphoreCreate(&(blocksData.mutex), 1);
    blocksData.blocks = (Block **)malloc(blocksMaxLen * sizeof(Block *));
    blocksData.contexts = (BlockContext **)malloc(blocksMaxLen * sizeof(BlockContext *));
    blocksData.tasks = (ThreadPoolTask **)malloc(blocksMaxLen * sizeof(ThreadPoolTask *));
    blocksData.length = 0;

    Block *topBlock = (Block *)malloc(sizeof(Block));
    topBlock->score = score;
    topBlock->queryFrontGap = queryFrontGap;
    topBlock->queryBackGap = queryBackGap;
    topBlock->targetFrontGap = targetFrontGap;
    topBlock->targetBackGap = targetBackGap;
    topBlock->startRow = 0;
    topBlock->startCol = 0;
    topBlock->endRow = rows - 1;
    topBlock->endCol = cols - 1;

    HirschbergContext hirschbergContext = {topBlock, &blocksData, query, target,
                                           scorer, cards, cardsLen};

    // WARNING : topBlock is deleted in this function
    hirschberg(&hirschbergContext);

    int blocksLen = blocksData.length;
    for (blockIdx = 0; blockIdx < blocksLen; ++blockIdx)
    {
        threadPoolTaskWait(blocksData.tasks[blockIdx]);
    }

    //**************************************************************************

    //**************************************************************************
    // CONCATENATE THE RESULT

    Block **blocks = blocksData.blocks;

    // becouse of multithreading blocks may not be in order
    qsort(blocks, blocksLen, sizeof(Block *), blockCmp);

    *pathLen = 0;
    for (blockIdx = 0; blockIdx < blocksLen; ++blockIdx)
    {
        *pathLen += blocks[blockIdx]->pathLen;
    }

    *path = (char *)malloc(*pathLen * sizeof(char));
    char *pathPtr = *path;

    for (blockIdx = 0; blockIdx < blocksLen; ++blockIdx)
    {
        size_t size = blocks[blockIdx]->pathLen * sizeof(char);
        memcpy(pathPtr, blocks[blockIdx]->path, size);
        pathPtr += blocks[blockIdx]->pathLen;
    }

    //**************************************************************************

    //**************************************************************************
    // CALCULATE OUT SCORE

    if (outScore != NULL)
    {

        int gapDiff = scorerGetGapOpen(scorer) - scorerGetGapExtend(scorer);

        *outScore = 0;

        for (blockIdx = 0; blockIdx < blocksLen; ++blockIdx)
        {

            Block *block = blocks[blockIdx];

            *outScore += block->outScore;

            // if two consecutive blocks are connected with a gap compensate the
            // double gap opening penalty, use only the back gaps since there
            // needs to be only one compensation per gap
            *outScore += block->queryFrontGap * gapDiff;
            *outScore += block->targetBackGap * gapDiff;
        }
    }

    // TIMER_STOP;

    //**************************************************************************

    //**************************************************************************
    // CLEAN MEMORY

    for (blockIdx = 0; blockIdx < blocksLen; ++blockIdx)
    {
        free(blocksData.blocks[blockIdx]->path);
        free(blocksData.blocks[blockIdx]);
        free(blocksData.contexts[blockIdx]);
        threadPoolTaskDelete(blocksData.tasks[blockIdx]);
    }

    semaphoreDelete(&(blocksData.mutex));
    free(blocksData.blocks);
    free(blocksData.contexts);
    free(blocksData.tasks);

    free(param);

    //**************************************************************************

    TIMER_STOP;

    return NULL;
}

static void *hirschberg(void *param)
{

    HirschbergContext *context = (HirschbergContext *)param;

    Block *block = context->block;
    BlocksData *blocksData = context->blocksData;
    Chain *rowChain = context->rowChain;
    Chain *colChain = context->colChain;
    Scorer *scorer = context->scorer;
    int *cards = context->cards;
    int cardsLen = context->cardsLen;

    Chain *rowSubchain = chainCreateView(rowChain, block->startRow, block->endRow, 0);
    Chain *colSubchain = chainCreateView(colChain, block->startCol, block->endCol, 0);

    int rows = chainGetLength(rowSubchain);
    int cols = chainGetLength(colSubchain);

    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);
    int gapDiff = gapOpen - gapExtend;

    int maxScore = scorerGetMaxScore(scorer);
    int minMatch = maxScore ? block->score / maxScore : 0;
    int t = MAX(rows, cols) - minMatch;
    int p = (t - abs(rows - cols)) / 2;

    double cells = (double)(2 * p + abs(rows - cols) + 1) * cols;

    if (rows < MIN_BLOCK_SIZE || cols < MIN_BLOCK_SIZE ||
        cells < MAX_BLOCK_CELLS || cardsLen == 0)
    {

        chainDelete(rowSubchain);
        chainDelete(colSubchain);

        // mm algorithm compensates and finds subblocks which often do not need
        // to have the optimal aligment, therefore it compensates the non
        // optimal aligment by reducing gap penalties, these have to be
        // compensated back
        block->score -= block->queryFrontGap * gapDiff;
        block->score -= block->queryBackGap * gapDiff;
        block->score -= block->targetFrontGap * gapDiff;
        block->score -= block->targetBackGap * gapDiff;

        BlockContext *context = (BlockContext *)malloc(sizeof(BlockContext));
        context->block = block;
        context->query = rowChain;
        context->target = colChain;
        context->scorer = scorer;

        semaphoreWait(&(blocksData->mutex));
        int last = blocksData->length;
        blocksData->length++;
        semaphorePost(&(blocksData->mutex));

        ThreadPoolTask *task = threadPoolSubmitToFront(blockReconstruct, (void *)context);

        blocksData->blocks[last] = block;
        blocksData->contexts[last] = context;
        blocksData->tasks[last] = task;

        return NULL;
    }

    int swapped = 0;
    if (rows < cols)
    {
        SWAP(rowSubchain, colSubchain);
        SWAP(rows, cols);
        SWAP(block->targetFrontGap, block->queryFrontGap);
        SWAP(block->targetBackGap, block->queryBackGap);
        swapped = 1;
    }

    int row = rows / 2;

    // inclusive, don't assume rows > cols
    int pLeft = rows > cols ? p + rows - cols + 1 : p + 1;
    int pRight = rows > cols ? p + 1 : p + cols - rows + 1;

    Chain *uRow = chainCreateView(rowSubchain, 0, row, 0);
    Chain *dRow = chainCreateView(rowSubchain, row + 1, rows - 1, 1);

    Chain *uCol = chainCreateView(colSubchain, 0, cols - 1, 0);
    Chain *dCol = chainCreateView(colSubchain, 0, cols - 1, 1);

    int *uScr;
    int *uAff;
    int *dScr;
    int *dAff;

    if (cardsLen == 1 || rows / 2 < MIN_DUAL_LEN || cols < MIN_DUAL_LEN)
    {

        nwLinearDataGpu(&uScr, &uAff, uRow, block->queryFrontGap, uCol,
                        block->targetFrontGap, scorer, pLeft, pRight, cards[0], NULL);

        nwLinearDataGpu(&dScr, &dAff, dRow, block->queryBackGap, dCol,
                        block->targetBackGap, scorer, pLeft, pRight, cards[0], NULL);
    }
    else
    {

        Thread thread;

        nwLinearDataGpu(&uScr, &uAff, uRow, block->queryFrontGap, uCol,
                        block->targetFrontGap, scorer, pLeft, pRight, cards[1], &thread);

        nwLinearDataGpu(&dScr, &dAff, dRow, block->queryBackGap, dCol,
                        block->targetBackGap, scorer, pLeft, pRight, cards[0], NULL);

        threadJoin(thread);
    }

    chainDelete(uRow);
    chainDelete(dRow);
    chainDelete(uCol);
    chainDelete(dCol);

    int uEmpty = -gapOpen - row * gapExtend + block->queryFrontGap * gapDiff;
    int dEmpty = -gapOpen - (rows - row - 2) * gapExtend + block->queryBackGap * gapDiff;

    int maxScr = INT_MIN;
    int gap = 0; // boolean
    int col = -1;

    int uMaxScore = 0;
    int dMaxScore = 0;

    int up, down;
    for (up = -1, down = cols - 1; up < cols; ++up, --down)
    {

        int uScore = up == -1 ? uEmpty : uScr[up];
        int uAffine = up == -1 ? uEmpty : uAff[up];

        int dScore = down == -1 ? dEmpty : dScr[down];
        int dAffine = down == -1 ? dEmpty : dAff[down];

        int scr = uScore + dScore;
        int aff = uAffine + dAffine + gapDiff;

        int isScrAff = (uScore == uAffine) && (dScore == dAffine);

        if (scr > maxScr || (scr == maxScr && !isScrAff))
        {
            maxScr = scr;
            gap = 0;
            col = up;
            uMaxScore = uScore;
            dMaxScore = dScore;
        }

        if (aff >= maxScr)
        {
            maxScr = aff;
            gap = 1;
            col = up;
            uMaxScore = uAffine + gapDiff;
            dMaxScore = dAffine + gapDiff;
        }
    }

    free(uScr);
    free(uAff);
    free(dScr);
    free(dAff);

    if (block->score != NO_SCORE)
    {
        ASSERT(maxScr == block->score, "score: %d, found: %d", block->score, maxScr);
    }

    if (swapped)
    {
        SWAP(rows, cols);
        SWAP(row, col);
        SWAP(block->targetFrontGap, block->queryFrontGap);
        SWAP(block->targetBackGap, block->queryBackGap);
    }

    chainDelete(rowSubchain);
    chainDelete(colSubchain);

    Block *upBlock = (Block *)malloc(sizeof(Block));
    upBlock->score = uMaxScore;
    upBlock->queryFrontGap = block->queryFrontGap;
    upBlock->queryBackGap = swapped ? 0 : gap;
    upBlock->targetFrontGap = block->targetFrontGap;
    upBlock->targetBackGap = swapped ? gap : 0;
    upBlock->startRow = block->startRow;
    upBlock->startCol = block->startCol;
    upBlock->endRow = block->startRow + row;
    upBlock->endCol = block->startCol + col;

    Block *downBlock = (Block *)malloc(sizeof(Block));
    downBlock->score = dMaxScore;
    downBlock->queryFrontGap = swapped ? 0 : gap;
    downBlock->queryBackGap = block->queryBackGap;
    downBlock->targetFrontGap = swapped ? gap : 0;
    downBlock->targetBackGap = block->targetBackGap;
    downBlock->startRow = block->startRow + row + 1;
    downBlock->startCol = block->startCol + col + 1;
    downBlock->endRow = block->startRow + rows - 1;
    downBlock->endCol = block->startCol + cols - 1;

    free(block);

    HirschbergContext *upContext = (HirschbergContext *)malloc(sizeof(HirschbergContext));
    upContext->block = upBlock;
    upContext->blocksData = blocksData;
    upContext->rowChain = rowChain;
    upContext->colChain = colChain;
    upContext->scorer = scorer;

    HirschbergContext *downContext = (HirschbergContext *)malloc(sizeof(HirschbergContext));
    downContext->block = downBlock;
    downContext->blocksData = blocksData;
    downContext->rowChain = rowChain;
    downContext->colChain = colChain;
    downContext->scorer = scorer;

    if (cardsLen >= 2)
    {

        int half = cardsLen / 2;

        upContext->cards = cards;
        upContext->cardsLen = half;

        downContext->cards = cards + half;
        downContext->cardsLen = cardsLen - half;

        Thread thread;

        threadCreate(&thread, hirschberg, (void *)(downContext));
        hirschberg(upContext); // master thread will work !!!!

        threadJoin(thread);
    }
    else
    {

        upContext->cards = cards;
        upContext->cardsLen = cardsLen;

        downContext->cards = cards;
        downContext->cardsLen = cardsLen;

        hirschberg(upContext);
        hirschberg(downContext);
    }

    free(upContext);
    free(downContext);

    return NULL;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// BLOCK

static void *blockReconstruct(void *params)
{

    BlockContext *context = (BlockContext *)params;
    Chain *query = context->query;
    Chain *target = context->target;
    Scorer *scorer = context->scorer;
    Block *block = context->block;

    Chain *subRow = chainCreateView(query, block->startRow, block->endRow, 0);
    Chain *subCol = chainCreateView(target, block->startCol, block->endCol, 0);

    nwReconstructCpu(&(block->path), &(block->pathLen), &(block->outScore),
                     subRow, block->queryFrontGap, block->queryBackGap,
                     subCol, block->targetFrontGap, block->targetBackGap,
                     scorer, block->score);

    chainDelete(subRow);
    chainDelete(subCol);

    return NULL;
}

static int blockCmp(const void *a_, const void *b_)
{

    Block *a = *((Block **)a_);
    Block *b = *((Block **)b_);

    int cmp1 = a->startRow - b->startRow;
    int cmp2 = a->startCol - b->startCol;

    return cmp1 == 0 ? cmp2 : cmp1;
}
//------------------------------------------------------------------------------
//******************************************************************************
