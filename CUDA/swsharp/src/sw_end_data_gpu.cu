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

#ifdef __CUDACC__

#include <stdlib.h>
#include <stdio.h>

#include "chain.h"
#include "constants.h"
#include "cuda_utils.h"
#include "error.h"
#include "scorer.h"
#include "thread.h"
#include "utils.h"

#include "gpu_module.h"

#define MAX_THREADS 1024

#define THREADS_SM1 64
#define BLOCKS_SM1 240

#define THREADS_SM2 128
#define BLOCKS_SM2 480

#define INT4_ZERO make_int4(0, 0, 0, 0)

typedef struct Atom
{
    int mch;
    int2 up;
    int4 lScr;
    int4 lAff;
    int4 rScr;
    int4 rAff;
} Atom;

typedef struct VBus
{
    int *mch;
    int4 *scr;
    int4 *aff;
} VBus;

typedef struct Context
{
    int **scores;
    int **affines;
    int *queryEnd;
    int *targetEnd;
    int *outScore;
    Chain *query;
    Chain *target;
    Scorer *scorer;
    int score;
    int card;
} Context;

static __constant__ int gapOpen_;
static __constant__ int gapExtend_;

static __constant__ int rows_;
static __constant__ int cols_;

static __constant__ int cellWidth_;

static __constant__ int pruneLow_;
static __constant__ int pruneHigh_;

static __constant__ int scorerLen_;
static __constant__ int subLen_;

static __constant__ int match_;
static __constant__ int mismatch_;

texture<char4> rowTexture;
texture<char> colTexture;
texture<int2> hBusTexture;
texture<int> subTexture;

//******************************************************************************
// PUBLIC

extern void swEndDataGpu(int *queryEnd, int *targetEnd, int *outScore,
                         int **scores, int **affines, Chain *query, Chain *target, Scorer *scorer,
                         int score, int card, Thread *thread);

//******************************************************************************

//******************************************************************************
// PRIVATE

// With visual c++ compiler and prototypes declared cuda global memory variables
// do not work. No questions asked.
#ifndef _WIN32

template <class Sub>
__device__ static void solveShortDelegated(int d, VBus vBus, int2 *hBus,
                                           int3 *results, Sub sub);

template <class Sub>
__device__ static void solveShortNormal(int d, VBus vBus, int2 *hBus,
                                        int3 *results, Sub sub);

template <class Sub>
__global__ static void solveShort(int d, VBus vBus, int2 *hBus, int3 *results,
                                  Sub sub);

template <class Sub>
__global__ static void solveLong(int d, VBus vBus, int2 *hBus, int *bBus,
                                 int3 *results, Sub sub);

#endif

static void *kernel(void *params);

//******************************************************************************

//******************************************************************************
// PUBLIC

extern void swEndDataGpu(int *queryEnd, int *targetEnd, int *outScore,
                         int **scores, int **affines, Chain *query, Chain *target, Scorer *scorer,
                         int score, int card, Thread *thread)
{

    Context *param = (Context *)malloc(sizeof(Context));

    param->scores = scores;
    param->affines = affines;
    param->queryEnd = queryEnd;
    param->targetEnd = targetEnd;
    param->outScore = outScore;
    param->query = query;
    param->target = target;
    param->scorer = scorer;
    param->score = score;
    param->card = card;

    if (thread == NULL)
    {
        kernel(param);
    }
    else
    {
        threadCreate(thread, kernel, (void *)param);
    }
}

//******************************************************************************

//******************************************************************************
// PRIVATE

//------------------------------------------------------------------------------
// FUNCTORS

class SubScalar
{
public:
    __device__ int operator()(char a, char b)
    {
        return a == b ? match_ : mismatch_;
    }
};

class SubScalarRev
{
public:
    __device__ int operator()(char a, char b)
    {
        return (a == b ? match_ : mismatch_) * (a < scorerLen_ && b < scorerLen_);
    }
};

class SubVector
{
public:
    __device__ int operator()(char a, char b)
    {
        return tex1Dfetch(subTexture, (a * subLen_) + b);
    }
};

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// GPU KERNELS

template <class Sub>
__device__ static void solveShortDelegated(int d, VBus vBus, int2 *hBus,
                                           int3 *results, Sub sub)
{

    __shared__ int hBusScrShr[MAX_THREADS];
    __shared__ int hBusAffShr[MAX_THREADS];

    if (pruneLow_ >= 0 && pruneHigh_ < gridDim.x)
    {
        return;
    }

    int row = (d + blockIdx.x - gridDim.x + 1) * (blockDim.x * 4) + threadIdx.x * 4;
    int col = cellWidth_ * (gridDim.x - blockIdx.x - 1) - threadIdx.x;

    if (row < 0)
        return;

    row -= (col < 0) * (gridDim.x * blockDim.x * 4);
    col += (col < 0) * cols_;

    Atom atom;

    if (0 <= row && row < rows_ && col > 0)
    {
        atom.mch = vBus.mch[(row >> 2) % (gridDim.x * blockDim.x)];
        VEC4_ASSIGN(atom.lScr, vBus.scr[(row >> 2) % (gridDim.x * blockDim.x)]);
        VEC4_ASSIGN(atom.lAff, vBus.aff[(row >> 2) % (gridDim.x * blockDim.x)]);
    }
    else
    {
        atom.mch = 0;
        VEC4_ASSIGN(atom.lScr, INT4_ZERO);
        VEC4_ASSIGN(atom.lAff, INT4_ZERO);
    }

    hBusScrShr[threadIdx.x] = tex1Dfetch(hBusTexture, col).x;
    hBusAffShr[threadIdx.x] = tex1Dfetch(hBusTexture, col).y;

    char4 rowCodes = tex1Dfetch(rowTexture, row >> 2);
    int3 res = {0, 0, 0};

    int del;

    for (int i = 0; i < blockDim.x; ++i)
    {

        if (0 <= row && row < rows_)
        {

            char columnCode = tex1Dfetch(colTexture, col);

            if (threadIdx.x == 0)
            {
                atom.up = tex1Dfetch(hBusTexture, col);
            }
            else
            {
                atom.up.x = hBusScrShr[threadIdx.x];
                atom.up.y = hBusAffShr[threadIdx.x];
            }

            del = max(atom.up.x - gapOpen_, atom.up.y - gapExtend_);
            int ins = max(atom.lScr.x - gapOpen_, atom.lAff.x - gapExtend_);
            int mch = atom.mch + sub(columnCode, rowCodes.x);

            atom.rScr.x = MAX4(0, mch, del, ins);
            atom.rAff.x = ins;

            del = max(atom.rScr.x - gapOpen_, del - gapExtend_);
            ins = max(atom.lScr.y - gapOpen_, atom.lAff.y - gapExtend_);
            mch = atom.lScr.x + sub(columnCode, rowCodes.y);

            atom.rScr.y = MAX4(0, mch, del, ins);
            atom.rAff.y = ins;

            del = max(atom.rScr.y - gapOpen_, del - gapExtend_);
            ins = max(atom.lScr.z - gapOpen_, atom.lAff.z - gapExtend_);
            mch = atom.lScr.y + sub(columnCode, rowCodes.z);

            atom.rScr.z = MAX4(0, mch, del, ins);
            atom.rAff.z = ins;

            del = max(atom.rScr.z - gapOpen_, del - gapExtend_);
            ins = max(atom.lScr.w - gapOpen_, atom.lAff.w - gapExtend_);
            mch = atom.lScr.z + sub(columnCode, rowCodes.w);

            atom.rScr.w = MAX4(0, mch, del, ins);
            atom.rAff.w = ins;

            if (atom.rScr.x > res.x)
            {
                res.x = atom.rScr.x;
                res.y = row;
                res.z = col;
            }
            if (atom.rScr.y > res.x)
            {
                res.x = atom.rScr.y;
                res.y = row + 1;
                res.z = col;
            }
            if (atom.rScr.z > res.x)
            {
                res.x = atom.rScr.z;
                res.y = row + 2;
                res.z = col;
            }
            if (atom.rScr.w > res.x)
            {
                res.x = atom.rScr.w;
                res.y = row + 3;
                res.z = col;
            }

            atom.mch = atom.up.x;
            VEC4_ASSIGN(atom.lScr, atom.rScr);
            VEC4_ASSIGN(atom.lAff, atom.rAff);
        }

        __syncthreads();

        if (0 <= row && row < rows_)
        {

            if (threadIdx.x == blockDim.x - 1 || i == blockDim.x - 1)
            {
                VEC2_ASSIGN(hBus[col], make_int2(atom.rScr.w, del));
            }
            else
            {
                hBusScrShr[threadIdx.x + 1] = atom.rScr.w;
                hBusAffShr[threadIdx.x + 1] = del;
            }
        }

        ++col;

        if (col == cols_)
        {

            col = 0;
            row = row + gridDim.x * blockDim.x * 4;

            atom.mch = 0;
            VEC4_ASSIGN(atom.lScr, INT4_ZERO);
            atom.lAff = atom.lScr;

            rowCodes = tex1Dfetch(rowTexture, row >> 2);
        }

        __syncthreads();
    }

    if (res.x > results[blockIdx.x * blockDim.x + threadIdx.x].x)
    {
        VEC3_ASSIGN(results[blockIdx.x * blockDim.x + threadIdx.x], res);
    }

    if (row < 0 || row >= rows_)
        return;

    vBus.mch[(row >> 2) % (gridDim.x * blockDim.x)] = atom.up.x;
    VEC4_ASSIGN(vBus.scr[(row >> 2) % (gridDim.x * blockDim.x)], atom.lScr);
    VEC4_ASSIGN(vBus.aff[(row >> 2) % (gridDim.x * blockDim.x)], atom.lAff);
}

template <class Sub>
__device__ static void solveShortNormal(int d, VBus vBus, int2 *hBus,
                                        int3 *results, Sub sub)
{

    __shared__ int hBusScrShr[MAX_THREADS];
    __shared__ int hBusAffShr[MAX_THREADS];

    if ((int)blockIdx.x <= pruneLow_ || blockIdx.x >= pruneHigh_)
    {
        return;
    }

    int row = (d + blockIdx.x - gridDim.x + 1) * (blockDim.x * 4) + threadIdx.x * 4;
    int col = cellWidth_ * (gridDim.x - blockIdx.x - 1) - threadIdx.x;

    if (row < 0 || row >= rows_)
        return;

    Atom atom;
    atom.mch = vBus.mch[(row >> 2) % (gridDim.x * blockDim.x)];
    VEC4_ASSIGN(atom.lScr, vBus.scr[(row >> 2) % (gridDim.x * blockDim.x)]);
    VEC4_ASSIGN(atom.lAff, vBus.aff[(row >> 2) % (gridDim.x * blockDim.x)]);

    hBusScrShr[threadIdx.x] = tex1Dfetch(hBusTexture, col).x;
    hBusAffShr[threadIdx.x] = tex1Dfetch(hBusTexture, col).y;

    const char4 rowCodes = tex1Dfetch(rowTexture, row >> 2);
    int3 res = {0, 0, 0};

    int del;

    for (int i = 0; i < blockDim.x; ++i, ++col)
    {

        char columnCode = tex1Dfetch(colTexture, col);

        if (threadIdx.x == 0)
        {
            atom.up = tex1Dfetch(hBusTexture, col);
        }
        else
        {
            atom.up = make_int2(hBusScrShr[threadIdx.x], hBusAffShr[threadIdx.x]);
        }

        del = max(atom.up.x - gapOpen_, atom.up.y - gapExtend_);
        int ins = max(atom.lScr.x - gapOpen_, atom.lAff.x - gapExtend_);
        int mch = atom.mch + sub(columnCode, rowCodes.x);

        atom.rScr.x = MAX4(0, mch, del, ins);
        atom.rAff.x = ins;

        del = max(atom.rScr.x - gapOpen_, del - gapExtend_);
        ins = max(atom.lScr.y - gapOpen_, atom.lAff.y - gapExtend_);
        mch = atom.lScr.x + sub(columnCode, rowCodes.y);

        atom.rScr.y = MAX4(0, mch, del, ins);
        atom.rAff.y = ins;

        del = max(atom.rScr.y - gapOpen_, del - gapExtend_);
        ins = max(atom.lScr.z - gapOpen_, atom.lAff.z - gapExtend_);
        mch = atom.lScr.y + sub(columnCode, rowCodes.z);

        atom.rScr.z = MAX4(0, mch, del, ins);
        atom.rAff.z = ins;

        del = max(atom.rScr.z - gapOpen_, del - gapExtend_);
        ins = max(atom.lScr.w - gapOpen_, atom.lAff.w - gapExtend_);
        mch = atom.lScr.z + sub(columnCode, rowCodes.w);

        atom.rScr.w = MAX4(0, mch, del, ins);
        atom.rAff.w = ins;

        if (atom.rScr.x > res.x)
        {
            res.x = atom.rScr.x;
            res.y = row;
            res.z = col;
        }
        if (atom.rScr.y > res.x)
        {
            res.x = atom.rScr.y;
            res.y = row + 1;
            res.z = col;
        }
        if (atom.rScr.z > res.x)
        {
            res.x = atom.rScr.z;
            res.y = row + 2;
            res.z = col;
        }
        if (atom.rScr.w > res.x)
        {
            res.x = atom.rScr.w;
            res.y = row + 3;
            res.z = col;
        }

        atom.mch = atom.up.x;
        VEC4_ASSIGN(atom.lScr, atom.rScr);
        VEC4_ASSIGN(atom.lAff, atom.rAff);

        __syncthreads();

        if (threadIdx.x == blockDim.x - 1)
        {
            VEC2_ASSIGN(hBus[col], make_int2(atom.rScr.w, del));
        }
        else
        {
            hBusScrShr[threadIdx.x + 1] = atom.rScr.w;
            hBusAffShr[threadIdx.x + 1] = del;
        }

        __syncthreads();
    }

    const int vBusIdx = (row >> 2) % (gridDim.x * blockDim.x);
    vBus.mch[vBusIdx] = atom.up.x;
    VEC4_ASSIGN(vBus.scr[vBusIdx], atom.lScr);
    VEC4_ASSIGN(vBus.aff[vBusIdx], atom.lAff);

    VEC2_ASSIGN(hBus[col - 1], make_int2(atom.rScr.w, del));

    if (res.x > results[blockIdx.x * blockDim.x + threadIdx.x].x)
    {
        VEC3_ASSIGN(results[blockIdx.x * blockDim.x + threadIdx.x], res);
    }
}

template <class Sub>
__global__ static void solveShort(int d, VBus vBus, int2 *hBus, int3 *results, Sub sub)
{

    if (blockIdx.x == (gridDim.x - 1))
    {
        solveShortDelegated(d, vBus, hBus, results, sub);
    }
    else
    {
        solveShortNormal(d, vBus, hBus, results, sub);
    }
}

template <class Sub>
__global__ static void solveLong(int d, VBus vBus, int2 *hBus, int *bBus,
                                 int3 *results, Sub sub)
{

    __shared__ int hBusScrShr[MAX_THREADS];
    __shared__ int hBusAffShr[MAX_THREADS];

    hBusScrShr[threadIdx.x] = 0;

    if ((int)blockIdx.x <= pruneLow_ || blockIdx.x > pruneHigh_)
    {
        return;
    }

    int row = (d + blockIdx.x - gridDim.x + 1) * (blockDim.x * 4) + threadIdx.x * 4;
    int col = cellWidth_ * (gridDim.x - blockIdx.x - 1) - threadIdx.x + blockDim.x;

    if (row < 0 || row >= rows_)
        return;

    if (blockIdx.x == pruneHigh_)
    {

        // clear only the last steepness
        vBus.mch[(row >> 2) % (gridDim.x * blockDim.x)] = 0;
        vBus.scr[(row >> 2) % (gridDim.x * blockDim.x)] = INT4_ZERO;
        vBus.aff[(row >> 2) % (gridDim.x * blockDim.x)] = INT4_ZERO;

        VEC2_ASSIGN(hBus[col + cellWidth_ - blockDim.x - 1], make_int2(0, 0));

        return;
    }

    Atom atom;
    atom.mch = vBus.mch[(row >> 2) % (gridDim.x * blockDim.x)];
    VEC4_ASSIGN(atom.lScr, vBus.scr[(row >> 2) % (gridDim.x * blockDim.x)]);
    VEC4_ASSIGN(atom.lAff, vBus.aff[(row >> 2) % (gridDim.x * blockDim.x)]);

    hBusScrShr[threadIdx.x] = tex1Dfetch(hBusTexture, col).x;
    hBusAffShr[threadIdx.x] = tex1Dfetch(hBusTexture, col).y;

    const char4 rowCodes = tex1Dfetch(rowTexture, row >> 2);
    int3 res = {0, 0, 0};

    int del;

    for (int i = 0; i < cellWidth_ - blockDim.x; ++i, ++col)
    {

        char columnCode = tex1Dfetch(colTexture, col);

        if (threadIdx.x == 0)
        {
            atom.up = tex1Dfetch(hBusTexture, col);
        }
        else
        {
            atom.up = make_int2(hBusScrShr[threadIdx.x], hBusAffShr[threadIdx.x]);
        }

        del = max(atom.up.x - gapOpen_, atom.up.y - gapExtend_);
        int ins = max(atom.lScr.x - gapOpen_, atom.lAff.x - gapExtend_);
        int mch = atom.mch + sub(columnCode, rowCodes.x);

        atom.rScr.x = MAX4(0, mch, del, ins);
        atom.rAff.x = ins;

        del = max(atom.rScr.x - gapOpen_, del - gapExtend_);
        ins = max(atom.lScr.y - gapOpen_, atom.lAff.y - gapExtend_);
        mch = atom.lScr.x + sub(columnCode, rowCodes.y);

        atom.rScr.y = MAX4(0, mch, del, ins);
        atom.rAff.y = ins;

        del = max(atom.rScr.y - gapOpen_, del - gapExtend_);
        ins = max(atom.lScr.z - gapOpen_, atom.lAff.z - gapExtend_);
        mch = atom.lScr.y + sub(columnCode, rowCodes.z);

        atom.rScr.z = MAX4(0, mch, del, ins);
        atom.rAff.z = ins;

        del = max(atom.rScr.z - gapOpen_, del - gapExtend_);
        ins = max(atom.lScr.w - gapOpen_, atom.lAff.w - gapExtend_);
        mch = atom.lScr.z + sub(columnCode, rowCodes.w);

        atom.rScr.w = MAX4(0, mch, del, ins);
        atom.rAff.w = ins;

        if (atom.rScr.x > res.x)
        {
            res.x = atom.rScr.x;
            res.y = row;
            res.z = col;
        }
        if (atom.rScr.y > res.x)
        {
            res.x = atom.rScr.y;
            res.y = row + 1;
            res.z = col;
        }
        if (atom.rScr.z > res.x)
        {
            res.x = atom.rScr.z;
            res.y = row + 2;
            res.z = col;
        }
        if (atom.rScr.w > res.x)
        {
            res.x = atom.rScr.w;
            res.y = row + 3;
            res.z = col;
        }

        atom.mch = atom.up.x;
        VEC4_ASSIGN(atom.lScr, atom.rScr);
        VEC4_ASSIGN(atom.lAff, atom.rAff);

        __syncthreads();

        if (threadIdx.x == blockDim.x - 1)
        {
            VEC2_ASSIGN(hBus[col], make_int2(atom.rScr.w, del));
        }
        else
        {
            hBusScrShr[threadIdx.x + 1] = atom.rScr.w;
            hBusAffShr[threadIdx.x + 1] = del;
        }

        __syncthreads();
    }

    const int vBusIdx = (row >> 2) % (gridDim.x * blockDim.x);
    vBus.mch[vBusIdx] = atom.up.x;
    VEC4_ASSIGN(vBus.scr[vBusIdx], atom.lScr);
    VEC4_ASSIGN(vBus.aff[vBusIdx], atom.lAff);

    VEC2_ASSIGN(hBus[col - 1], make_int2(atom.rScr.w, del));

    if (res.x > results[blockIdx.x * blockDim.x + threadIdx.x].x)
    {
        VEC3_ASSIGN(results[blockIdx.x * blockDim.x + threadIdx.x], res);
    }

    // reuse
    hBusScrShr[threadIdx.x] = res.x;
    __syncthreads();

    int score = 0;
    int idx = 0;

    for (int i = 0; i < blockDim.x; ++i)
    {

        int shr = hBusScrShr[i];

        if (shr > score)
        {
            score = shr;
            idx = i;
        }
    }

    if (threadIdx.x == idx)
        bBus[blockIdx.x] = score;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// CPU KERNELS

static void *kernel(void *params)
{

    Context *context = (Context *)params;

    int **scores = context->scores;
    int **affines = context->affines;
    int *queryEnd = context->queryEnd;
    int *targetEnd = context->targetEnd;
    int *outScore = context->outScore;
    Chain *query = context->query;
    Chain *target = context->target;
    Scorer *scorer = context->scorer;
    int score = context->score;
    int card = context->card;

    // if negative matrix, no need for SW, score will not be found
    if (scorerGetMaxScore(scorer) <= 0)
    {
        *outScore = NO_SCORE;
        *queryEnd = 0;
        *targetEnd = 0;
        if (scores != NULL)
            *scores = NULL;
        if (affines != NULL)
            *affines = NULL;
        free(params);
        return NULL;
    }

    int currentCard;
    CUDA_SAFE_CALL(cudaGetDevice(&currentCard));
    if (currentCard != card)
    {
        // CUDA_SAFE_CALL(cudaThreadExit());
        CUDA_SAFE_CALL(cudaSetDevice(card));
    }

    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);
    int scorerLen = scorerGetMaxCode(scorer);
    int subLen = scorerLen + 1;
    int scalar = scorerIsScalar(scorer);

    TIMER_START("Sw end data %d %d", rows, cols);

    cudaDeviceProp properties;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&properties, card));

    int threads;
    int blocks;

    maxWorkGroups(card, BLOCKS_SM2, THREADS_SM2, cols, &blocks, &threads);

    int cellHeight = 4 * threads;
    int rowsGpu = rows + (cellHeight - rows % cellHeight) % cellHeight;

    int colsGpu = cols + (blocks - cols % blocks) % blocks;
    int cellWidth = colsGpu / blocks;

    int diagonals = blocks + (rowsGpu / cellHeight);

    int pruneLow = -1;
    int pruneHigh = blocks;
    int pruneFactor = scorerGetMaxScore(scorer);

    int memoryUsedGpu = 0;
    int memoryUsedCpu = 0;

    /*
    LOG("Rows cpu: %d, gpu: %d", rows, rowsGpu);
    LOG("Columns cpu: %d, gpu: %d", cols, colsGpu);
    LOG("Cell h: %d, w: %d", cellHeight, cellWidth);
    LOG("Diagonals: %d", diagonals);
    */

    //**************************************************************************
    // PADD CHAINS
    char *rowCpu = (char *)malloc(rowsGpu * sizeof(char));
    memset(rowCpu, scorerLen, (rowsGpu - rows) * sizeof(char));
    chainCopyCodes(query, rowCpu + (rowsGpu - rows));
    memoryUsedCpu += rowsGpu * sizeof(char);

    char *colCpu = (char *)malloc(colsGpu * sizeof(char));
    memset(colCpu + cols, scorerLen + scalar, (colsGpu - cols) * sizeof(char));
    chainCopyCodes(target, colCpu);
    memoryUsedCpu += colsGpu * sizeof(char);
    //**************************************************************************

    //**************************************************************************
    // INIT GPU
    size_t rowSize = rowsGpu * sizeof(char);
    char4 *rowGpu;
    CUDA_SAFE_CALL(cudaMalloc(&rowGpu, rowSize));
    CUDA_SAFE_CALL(cudaMemcpy(rowGpu, rowCpu, rowSize, TO_GPU));
    CUDA_SAFE_CALL(cudaBindTexture(NULL, rowTexture, rowGpu, rowSize));
    memoryUsedGpu += rowSize;

    size_t colSize = colsGpu * sizeof(char);
    char *colGpu;
    CUDA_SAFE_CALL(cudaMalloc(&colGpu, colSize));
    CUDA_SAFE_CALL(cudaMemcpy(colGpu, colCpu, colSize, TO_GPU));
    CUDA_SAFE_CALL(cudaBindTexture(NULL, colTexture, colGpu, colSize));
    memoryUsedGpu += colSize;

    size_t hBusSize = colsGpu * sizeof(int2);
    int2 *hBusCpu;
    int2 *hBusGpu;
    CUDA_SAFE_CALL(cudaMallocHost(&hBusCpu, hBusSize));
    CUDA_SAFE_CALL(cudaMalloc(&hBusGpu, hBusSize));
    CUDA_SAFE_CALL(cudaMemset(hBusGpu, 0, hBusSize));
    CUDA_SAFE_CALL(cudaBindTexture(NULL, hBusTexture, hBusGpu, hBusSize));
    memoryUsedCpu += hBusSize;
    memoryUsedGpu += hBusSize;

    VBus vBusGpu;
    CUDA_SAFE_CALL(cudaMalloc(&vBusGpu.mch, blocks * threads * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&vBusGpu.scr, blocks * threads * sizeof(int4)));
    CUDA_SAFE_CALL(cudaMalloc(&vBusGpu.aff, blocks * threads * sizeof(int4)));
    memoryUsedGpu += blocks * threads * sizeof(int);
    memoryUsedGpu += blocks * threads * sizeof(int4);
    memoryUsedGpu += blocks * threads * sizeof(int4);

    size_t resultsSize = blocks * threads * sizeof(int3);
    int3 *resultsCpu = (int3 *)malloc(resultsSize);
    int3 *resultsGpu;
    CUDA_SAFE_CALL(cudaMalloc(&resultsGpu, resultsSize));
    CUDA_SAFE_CALL(cudaMemset(resultsGpu, 0, resultsSize));
    memoryUsedCpu += resultsSize;
    memoryUsedGpu += resultsSize;

    size_t bSize = blocks * sizeof(int);
    int *bCpu;
    int *bGpu;
    CUDA_SAFE_CALL(cudaMallocHost(&bCpu, bSize));
    CUDA_SAFE_CALL(cudaMalloc(&bGpu, bSize));
    CUDA_SAFE_CALL(cudaMemset(bGpu, 0, bSize));
    memoryUsedCpu += bSize;
    memoryUsedGpu += bSize;

    size_t subSize = subLen * subLen * sizeof(int);
    int *subCpu = (int *)malloc(subSize);
    int *subGpu;
    for (int i = 0; i < subLen; ++i)
    {
        for (int j = 0; j < subLen; ++j)
        {
            if (i < scorerLen && j < scorerLen)
            {
                subCpu[i * subLen + j] = scorerScore(scorer, i, j);
            }
            else
            {
                subCpu[i * subLen + j] = 0;
            }
        }
    }
    CUDA_SAFE_CALL(cudaMalloc(&subGpu, subSize));
    CUDA_SAFE_CALL(cudaMemcpy(subGpu, subCpu, subSize, TO_GPU));
    CUDA_SAFE_CALL(cudaBindTexture(NULL, subTexture, subGpu, subSize));
    memoryUsedCpu += subSize;
    memoryUsedGpu += subSize;

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(match_, &(subCpu[0]), sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(mismatch_, &(subCpu[1]), sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(gapOpen_, &gapOpen, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(gapExtend_, &gapExtend, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(scorerLen_, &scorerLen, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(subLen_, &subLen, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(rows_, &rowsGpu, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(cols_, &colsGpu, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(cellWidth_, &cellWidth, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(pruneLow_, &pruneLow, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(pruneHigh_, &pruneHigh, sizeof(int)));

    // LOG("Memory used CPU: %fMB", memoryUsedCpu / 1024. / 1024.);
    LOG("Memory used GPU: %fMB", memoryUsedGpu / 1024. / 1024.);

    //**************************************************************************

    //**************************************************************************
    // KERNEL RUN

    int best = MAX(0, score);
    int pruning = 1;
    int pruned = 0;
    int pruneHighOld = pruneHigh;
    int halfPruning = scores != NULL || affines != NULL;

    // TIMER_START("Kernel");

    for (int diagonal = 0; diagonal < diagonals; ++diagonal)
    {

        if (scalar)
        {
            if (subCpu[0] >= subCpu[1])
            {
                solveShort<<<blocks, threads>>>(diagonal, vBusGpu, hBusGpu, resultsGpu, SubScalar());
                solveLong<<<blocks, threads>>>(diagonal, vBusGpu, hBusGpu, bGpu, resultsGpu, SubScalar());
            }
            else
            {
                // cannot use mismatch negative trick
                solveShort<<<blocks, threads>>>(diagonal, vBusGpu, hBusGpu, resultsGpu, SubScalarRev());
                solveLong<<<blocks, threads>>>(diagonal, vBusGpu, hBusGpu, bGpu, resultsGpu, SubScalarRev());
            }
        }
        else
        {
            solveShort<<<blocks, threads>>>(diagonal, vBusGpu, hBusGpu, resultsGpu, SubVector());
            solveLong<<<blocks, threads>>>(diagonal, vBusGpu, hBusGpu, bGpu, resultsGpu, SubVector());
        }

        if (pruning)
        {

            size_t bSize = pruneHigh * sizeof(int);
            CUDA_SAFE_CALL(cudaMemcpy(bCpu, bGpu, bSize, FROM_GPU));

            if (score == NO_SCORE)
            {
                for (int i = 0; i < pruneHigh; ++i)
                {
                    best = max(best, bCpu[i]);
                }
            }

            // delta j pruning
            pruneLow = -1;
            for (int i = 0; i < blocks; ++i)
            {
                int row = (diagonal + 1 + i - blocks + 1) * (threads * 4);
                int col = cellWidth * (blocks - i - 1) - threads;
                if (row >= rowsGpu)
                    break;
                if (rowsGpu * (halfPruning ? 2 : 1) - row < cols - col)
                    break;
                int d = cols - col;
                int scr = i == blocks - 1 ? bCpu[i] : max(bCpu[i], bCpu[i + 1]);
                if ((scr + d * pruneFactor) < best)
                    pruneLow = i;
                else
                    break;
            }

            // delta i pruning
            if (!halfPruning)
            {
                pruneHighOld = pruneHigh;
                for (int i = pruneHighOld - 1; i >= 0; --i)
                {
                    int row = (diagonal + 1 + i - blocks + 1) * (threads * 4);
                    int col = cellWidth * (blocks - i - 1) - threads;
                    if (row < rowsGpu / 2)
                        break;
                    if (row >= rowsGpu)
                        continue;
                    if (rowsGpu - row > cols - col)
                        break;
                    int d = rowsGpu - row;
                    int scr1 = d * pruneFactor + (i == blocks - 1 ? 0 : bCpu[i + 1]);
                    int scr2 = (d + threads * 2) * pruneFactor + bCpu[i];
                    if (scr1 < best && scr2 < best)
                        pruneHigh = i;
                    else
                        break;
                }
            }

            pruned += blocks - (pruneHigh - pruneLow - 1);

            if (pruneLow >= pruneHigh)
            {
                break;
            }

            CUDA_SAFE_CALL(cudaMemcpyToSymbol(pruneLow_, &pruneLow, sizeof(int)));
            CUDA_SAFE_CALL(cudaMemcpyToSymbol(pruneHigh_, &pruneHigh, sizeof(int)));

            if (pruneLow >= 0)
            {
                int offset = (blocks - pruneLow - 1) * cellWidth - threads;
                size_t size = (colsGpu - offset) * sizeof(int2);
                CUDA_SAFE_CALL(cudaMemset(hBusGpu + offset, 0, size));
            }
        }
    }

    // TIMER_STOP;

    LOG("Pruned percentage %.2f%%", 100.0 * pruned / (diagonals * blocks));

    //**************************************************************************

    //**************************************************************************
    // SAVE RESULTS

    // save only if needed
    if (scores != NULL && affines != NULL)
    {

        CUDA_SAFE_CALL(cudaMemcpy(hBusCpu, hBusGpu, hBusSize, FROM_GPU));

        *scores = (int *)malloc(cols * sizeof(int));
        *affines = (int *)malloc(cols * sizeof(int));

        for (int i = 0; i < cols; ++i)
        {
            (*scores)[i] = hBusCpu[i].x;
            (*affines)[i] = hBusCpu[i].y;
        }
    }

    CUDA_SAFE_CALL(cudaMemcpy(resultsCpu, resultsGpu, resultsSize, FROM_GPU));

    int3 res = resultsCpu[0];
    for (int i = 1; i < blocks * threads; ++i)
    {
        if (resultsCpu[i].x > res.x)
        {
            res = resultsCpu[i];
        }
    }

    res.y -= (rowsGpu - rows); // restore padding

    // check if the result updated in the padded part
    if (res.y >= rows)
    {
        res.z += rows - res.y - 1;
        res.y += rows - res.y - 1;
    }

    if (res.z >= cols)
    {
        res.y += cols - res.z - 1;
        res.z += cols - res.z - 1;
    }

    *outScore = res.x;
    *queryEnd = res.y;
    *targetEnd = res.z;

    LOG("Score: %d, (%d, %d)", *outScore, *queryEnd, *targetEnd);

    //**************************************************************************

    //**************************************************************************
    // CLEAN MEMORY

    free(subCpu);
    free(rowCpu);
    free(colCpu);
    free(resultsCpu);

    CUDA_SAFE_CALL(cudaFreeHost(bCpu));
    CUDA_SAFE_CALL(cudaFreeHost(hBusCpu));

    CUDA_SAFE_CALL(cudaFree(subGpu));
    CUDA_SAFE_CALL(cudaFree(rowGpu));
    CUDA_SAFE_CALL(cudaFree(colGpu));
    CUDA_SAFE_CALL(cudaFree(vBusGpu.mch));
    CUDA_SAFE_CALL(cudaFree(vBusGpu.scr));
    CUDA_SAFE_CALL(cudaFree(vBusGpu.aff));
    CUDA_SAFE_CALL(cudaFree(hBusGpu));
    CUDA_SAFE_CALL(cudaFree(resultsGpu));
    CUDA_SAFE_CALL(cudaFree(bGpu));

    CUDA_SAFE_CALL(cudaUnbindTexture(rowTexture));
    CUDA_SAFE_CALL(cudaUnbindTexture(colTexture));
    CUDA_SAFE_CALL(cudaUnbindTexture(hBusTexture));
    CUDA_SAFE_CALL(cudaUnbindTexture(subTexture));

    free(params);

    //**************************************************************************

    TIMER_STOP;

    return NULL;
}

//------------------------------------------------------------------------------
//******************************************************************************

#endif // __CUDACC__
