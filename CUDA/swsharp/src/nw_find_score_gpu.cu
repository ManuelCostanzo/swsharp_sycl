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

#define SCORE4_MIN make_int4(SCORE_MIN, SCORE_MIN, SCORE_MIN, SCORE_MIN)

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
    int *queryStart;
    int *targetStart;
    int score;
    Chain *query;
    int queryFrontGap;
    Chain *target;
    Scorer *scorer;
    int card;
} Context;

static __constant__ int queryFrontGap_;

static __constant__ int gapOpen_;
static __constant__ int gapExtend_;
static __constant__ int gapDiff_;

static __constant__ int rows_;
static __constant__ int cols_;
static __constant__ int cellWidth_;

static __constant__ int score_;
static __constant__ int pLeft_;
static __constant__ int pRight_;

static __constant__ int scorerLen_;
static __constant__ int subLen_;

static __constant__ int match_;
static __constant__ int mismatch_;

texture<char4> rowTexture;
texture<char> colTexture;
texture<int2> hBusTexture;
texture<int> subTexture;

static __device__ int2 res_;

//******************************************************************************
// PUBLIC

extern void nwFindScoreGpu(int *queryStart, int *targetStart, Chain *query,
                           int queryFrontGap, Chain *target, Scorer *scorer, int score, int card,
                           Thread *thread);

//******************************************************************************

//******************************************************************************
// PRIVATE

// With visual c++ compiler and prototypes declared cuda global memory variables
// do not work. No questions asked.
#ifndef _WIN32
__device__ static int gap(int idx);

template <class Sub>
__device__ static void solveShortDelegated(int d, VBus vBus, int2 *hBus, Sub sub);

template <class Sub>
__device__ static void solveShortNormal(int d, VBus vBus, int2 *hBus, Sub sub);

template <class Sub>
__global__ static void solveShort(int d, VBus vBus, int2 *hBus, Sub sub);

template <class Sub>
__global__ static void solveLong(int d, VBus vBus, int2 *hBus, Sub sub);

#endif

static void *kernel(void *params);

//******************************************************************************

//******************************************************************************
// PUBLIC

extern void nwFindScoreGpu(int *queryStart, int *targetStart, Chain *query,
                           int queryFrontGap, Chain *target, Scorer *scorer, int score, int card,
                           Thread *thread)
{

    Context *param = (Context *)malloc(sizeof(Context));

    param->queryStart = queryStart;
    param->targetStart = targetStart;
    param->score = score;
    param->query = query;
    param->queryFrontGap = queryFrontGap;
    param->target = target;
    param->scorer = scorer;
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

__device__ static int gap(int idx)
{
    return -gapOpen_ - gapExtend_ * idx + queryFrontGap_ * gapDiff_;
}

template <class Sub>
__device__ static void solveShortDelegated(int d, VBus vBus, int2 *hBus, Sub sub)
{

    __shared__ int hBusScrShr[MAX_THREADS];
    __shared__ int hBusAffShr[MAX_THREADS];

    int row = (d + blockIdx.x - gridDim.x + 1) * (blockDim.x * 4) + threadIdx.x * 4;
    int col = cellWidth_ * (gridDim.x - blockIdx.x - 1) - threadIdx.x;

    if (row < 0)
        return;

    row -= (col < 0) * (gridDim.x * blockDim.x * 4);
    col += (col < 0) * cols_;

    int x1 = cellWidth_ * (gridDim.x - blockIdx.x - 1) + blockDim.x;
    int y1 = (d + blockIdx.x - gridDim.x + 1) * (blockDim.x * 4);

    int x2 = cellWidth_ * (gridDim.x - blockIdx.x - 1) - blockDim.x;
    int y2 = (d + blockIdx.x - gridDim.x + 2) * (blockDim.x * 4);

    y2 -= (x2 < 0) * (gridDim.x * blockDim.x * 4);
    x2 += (x2 < 0) * cols_;

    if (y1 - x1 > pLeft_ && (x2 - y2 > pRight_ || y2 < 0))
    {

        row += (col != 0) * gridDim.x * blockDim.x * 4;
        vBus.mch[(row >> 2) % (gridDim.x * blockDim.x)] = SCORE_MIN;
        vBus.scr[(row >> 2) % (gridDim.x * blockDim.x)] = SCORE4_MIN;
        vBus.aff[(row >> 2) % (gridDim.x * blockDim.x)] = SCORE4_MIN;

        hBus[col] = make_int2(SCORE_MIN, SCORE_MIN);

        return;
    }

    Atom atom;

    if (0 <= row && row < rows_ && col > 0)
    {
        atom.mch = vBus.mch[(row >> 2) % (gridDim.x * blockDim.x)];
        VEC4_ASSIGN(atom.lScr, vBus.scr[(row >> 2) % (gridDim.x * blockDim.x)]);
        VEC4_ASSIGN(atom.lAff, vBus.aff[(row >> 2) % (gridDim.x * blockDim.x)]);
    }
    else
    {
        atom.mch = gap(row - 1) * (row != 0);
        atom.lScr = make_int4(gap(row), gap(row + 1), gap(row + 2), gap(row + 3));
        atom.lAff = SCORE4_MIN;
    }

    hBusScrShr[threadIdx.x] = tex1Dfetch(hBusTexture, col).x;
    hBusAffShr[threadIdx.x] = tex1Dfetch(hBusTexture, col).y;

    char4 rowCodes = tex1Dfetch(rowTexture, row >> 2);

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

            atom.rScr.x = MAX3(mch, del, ins);
            atom.rAff.x = ins;

            del = max(atom.rScr.x - gapOpen_, del - gapExtend_);
            ins = max(atom.lScr.y - gapOpen_, atom.lAff.y - gapExtend_);
            mch = atom.lScr.x + sub(columnCode, rowCodes.y);

            atom.rScr.y = MAX3(mch, del, ins);
            atom.rAff.y = ins;

            del = max(atom.rScr.y - gapOpen_, del - gapExtend_);
            ins = max(atom.lScr.z - gapOpen_, atom.lAff.z - gapExtend_);
            mch = atom.lScr.y + sub(columnCode, rowCodes.z);

            atom.rScr.z = MAX3(mch, del, ins);
            atom.rAff.z = ins;

            del = max(atom.rScr.z - gapOpen_, del - gapExtend_);
            ins = max(atom.lScr.w - gapOpen_, atom.lAff.w - gapExtend_);
            mch = atom.lScr.z + sub(columnCode, rowCodes.w);

            atom.rScr.w = MAX3(mch, del, ins);
            atom.rAff.w = ins;

            if (atom.rScr.x == score_)
            {
                res_.x = row;
                res_.y = col;
            }
            else if (atom.rScr.y == score_)
            {
                res_.x = row + 1;
                res_.y = col;
            }
            else if (atom.rScr.z == score_)
            {
                res_.x = row + 2;
                res_.y = col;
            }
            else if (atom.rScr.w == score_)
            {
                res_.x = row + 3;
                res_.y = col;
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

            atom.mch = gap(row - 1) * (row != 0);
            atom.lScr = make_int4(gap(row), gap(row + 1), gap(row + 2), gap(row + 3));
            atom.lAff = SCORE4_MIN;

            rowCodes = tex1Dfetch(rowTexture, row >> 2);
        }

        __syncthreads();
    }

    if (row < 0 || row >= rows_)
        return;

    vBus.mch[(row >> 2) % (gridDim.x * blockDim.x)] = atom.up.x;
    VEC4_ASSIGN(vBus.scr[(row >> 2) % (gridDim.x * blockDim.x)], atom.lScr);
    VEC4_ASSIGN(vBus.aff[(row >> 2) % (gridDim.x * blockDim.x)], atom.lAff);
}

template <class Sub>
__device__ static void solveShortNormal(int d, VBus vBus, int2 *hBus, Sub sub)
{

    __shared__ int hBusScrShr[MAX_THREADS];
    __shared__ int hBusAffShr[MAX_THREADS];

    int row = (d + blockIdx.x - gridDim.x + 1) * (blockDim.x * 4) + threadIdx.x * 4;
    int col = cellWidth_ * (gridDim.x - blockIdx.x - 1) - threadIdx.x;

    if (row < 0 || row >= rows_)
        return;

    int x1 = cellWidth_ * (gridDim.x - blockIdx.x - 1) + blockDim.x;
    int y1 = (d + blockIdx.x - gridDim.x + 1) * (blockDim.x * 4);

    if (y1 - x1 > pLeft_)
    {

        // only clear right, down is irelevant
        vBus.mch[(row >> 2) % (gridDim.x * blockDim.x)] = SCORE_MIN;
        vBus.scr[(row >> 2) % (gridDim.x * blockDim.x)] = SCORE4_MIN;
        vBus.aff[(row >> 2) % (gridDim.x * blockDim.x)] = SCORE4_MIN;

        return;
    }

    int x2 = cellWidth_ * (gridDim.x - blockIdx.x - 1) - blockDim.x;
    int y2 = (d + blockIdx.x - gridDim.x + 2) * (blockDim.x * 4);

    if (x2 - y2 > pRight_)
    {

        // clear right
        vBus.mch[(row >> 2) % (gridDim.x * blockDim.x)] = SCORE_MIN;
        vBus.scr[(row >> 2) % (gridDim.x * blockDim.x)] = SCORE4_MIN;
        vBus.aff[(row >> 2) % (gridDim.x * blockDim.x)] = SCORE4_MIN;

        // clear down
        hBus[col] = make_int2(SCORE_MIN, SCORE_MIN);
        hBus[col + blockDim.x] = make_int2(SCORE_MIN, SCORE_MIN);

        return;
    }

    Atom atom;
    atom.mch = vBus.mch[(row >> 2) % (gridDim.x * blockDim.x)];
    VEC4_ASSIGN(atom.lScr, vBus.scr[(row >> 2) % (gridDim.x * blockDim.x)]);
    VEC4_ASSIGN(atom.lAff, vBus.aff[(row >> 2) % (gridDim.x * blockDim.x)]);

    hBusScrShr[threadIdx.x] = tex1Dfetch(hBusTexture, col).x;
    hBusAffShr[threadIdx.x] = tex1Dfetch(hBusTexture, col).y;

    const char4 rowCodes = tex1Dfetch(rowTexture, row >> 2);

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
            atom.up.x = hBusScrShr[threadIdx.x];
            atom.up.y = hBusAffShr[threadIdx.x];
        }

        del = max(atom.up.x - gapOpen_, atom.up.y - gapExtend_);
        int ins = max(atom.lScr.x - gapOpen_, atom.lAff.x - gapExtend_);
        int mch = atom.mch + sub(columnCode, rowCodes.x);

        atom.rScr.x = MAX3(mch, del, ins);
        atom.rAff.x = ins;

        del = max(atom.rScr.x - gapOpen_, del - gapExtend_);
        ins = max(atom.lScr.y - gapOpen_, atom.lAff.y - gapExtend_);
        mch = atom.lScr.x + sub(columnCode, rowCodes.y);

        atom.rScr.y = MAX3(mch, del, ins);
        atom.rAff.y = ins;

        del = max(atom.rScr.y - gapOpen_, del - gapExtend_);
        ins = max(atom.lScr.z - gapOpen_, atom.lAff.z - gapExtend_);
        mch = atom.lScr.y + sub(columnCode, rowCodes.z);

        atom.rScr.z = MAX3(mch, del, ins);
        atom.rAff.z = ins;

        del = max(atom.rScr.z - gapOpen_, del - gapExtend_);
        ins = max(atom.lScr.w - gapOpen_, atom.lAff.w - gapExtend_);
        mch = atom.lScr.z + sub(columnCode, rowCodes.w);

        atom.rScr.w = MAX3(mch, del, ins);
        atom.rAff.w = ins;

        if (atom.rScr.x == score_)
        {
            res_.x = row;
            res_.y = col;
        }
        else if (atom.rScr.y == score_)
        {
            res_.x = row + 1;
            res_.y = col;
        }
        else if (atom.rScr.z == score_)
        {
            res_.x = row + 2;
            res_.y = col;
        }
        else if (atom.rScr.w == score_)
        {
            res_.x = row + 3;
            res_.y = col;
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

    VEC2_ASSIGN(hBus[col - 1], make_int2(atom.rScr.w, del));

    const int vBusIdx = (row >> 2) % (gridDim.x * blockDim.x);
    vBus.mch[vBusIdx] = atom.up.x;
    VEC4_ASSIGN(vBus.scr[vBusIdx], atom.lScr);
    VEC4_ASSIGN(vBus.aff[vBusIdx], atom.lAff);
}

template <class Sub>
__global__ static void solveShort(int d, VBus vBus, int2 *hBus, Sub sub)
{

    if (blockIdx.x == (gridDim.x - 1))
    {
        solveShortDelegated(d, vBus, hBus, sub);
    }
    else
    {
        solveShortNormal(d, vBus, hBus, sub);
    }
}

template <class Sub>
__global__ static void solveLong(int d, VBus vBus, int2 *hBus, Sub sub)
{

    __shared__ int hBusScrShr[MAX_THREADS];
    __shared__ int hBusAffShr[MAX_THREADS];

    int row = (d + blockIdx.x - gridDim.x + 1) * (blockDim.x * 4) + threadIdx.x * 4;
    int col = cellWidth_ * (gridDim.x - blockIdx.x - 1) - threadIdx.x + blockDim.x;

    if (row < 0 || row >= rows_)
        return;

    int x1 = cellWidth_ * (gridDim.x - blockIdx.x - 1) + cellWidth_;
    int y1 = (d + blockIdx.x - gridDim.x + 1) * (blockDim.x * 4);

    if (y1 - x1 > pLeft_)
    {

        vBus.mch[(row >> 2) % (gridDim.x * blockDim.x)] = SCORE_MIN;
        vBus.scr[(row >> 2) % (gridDim.x * blockDim.x)] = SCORE4_MIN;
        vBus.aff[(row >> 2) % (gridDim.x * blockDim.x)] = SCORE4_MIN;

        return;
    }

    int x2 = cellWidth_ * (gridDim.x - blockIdx.x - 1);
    int y2 = (d + blockIdx.x - gridDim.x + 2) * (blockDim.x * 4);

    if (x2 - y2 > pRight_)
    {

        vBus.mch[(row >> 2) % (gridDim.x * blockDim.x)] = SCORE_MIN;
        vBus.scr[(row >> 2) % (gridDim.x * blockDim.x)] = SCORE4_MIN;
        vBus.aff[(row >> 2) % (gridDim.x * blockDim.x)] = SCORE4_MIN;

        for (int i = 0; i < cellWidth_ - blockDim.x; i += blockDim.x)
        {
            hBus[col + i] = make_int2(SCORE_MIN, SCORE_MIN);
        }

        return;
    }

    Atom atom;
    atom.mch = vBus.mch[(row >> 2) % (gridDim.x * blockDim.x)];
    VEC4_ASSIGN(atom.lScr, vBus.scr[(row >> 2) % (gridDim.x * blockDim.x)]);
    VEC4_ASSIGN(atom.lAff, vBus.aff[(row >> 2) % (gridDim.x * blockDim.x)]);

    hBusScrShr[threadIdx.x] = tex1Dfetch(hBusTexture, col).x;
    hBusAffShr[threadIdx.x] = tex1Dfetch(hBusTexture, col).y;

    const char4 rowCodes = tex1Dfetch(rowTexture, row >> 2);

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
            atom.up.x = hBusScrShr[threadIdx.x];
            atom.up.y = hBusAffShr[threadIdx.x];
        }

        del = max(atom.up.x - gapOpen_, atom.up.y - gapExtend_);
        int ins = max(atom.lScr.x - gapOpen_, atom.lAff.x - gapExtend_);
        int mch = atom.mch + sub(columnCode, rowCodes.x);

        atom.rScr.x = MAX3(mch, del, ins);
        atom.rAff.x = ins;

        del = max(atom.rScr.x - gapOpen_, del - gapExtend_);
        ins = max(atom.lScr.y - gapOpen_, atom.lAff.y - gapExtend_);
        mch = atom.lScr.x + sub(columnCode, rowCodes.y);

        atom.rScr.y = MAX3(mch, del, ins);
        atom.rAff.y = ins;

        del = max(atom.rScr.y - gapOpen_, del - gapExtend_);
        ins = max(atom.lScr.z - gapOpen_, atom.lAff.z - gapExtend_);
        mch = atom.lScr.y + sub(columnCode, rowCodes.z);

        atom.rScr.z = MAX3(mch, del, ins);
        atom.rAff.z = ins;

        del = max(atom.rScr.z - gapOpen_, del - gapExtend_);
        ins = max(atom.lScr.w - gapOpen_, atom.lAff.w - gapExtend_);
        mch = atom.lScr.z + sub(columnCode, rowCodes.w);

        atom.rScr.w = MAX3(mch, del, ins);
        atom.rAff.w = ins;

        if (atom.rScr.x == score_)
        {
            res_.x = row;
            res_.y = col;
        }
        else if (atom.rScr.y == score_)
        {
            res_.x = row + 1;
            res_.y = col;
        }
        else if (atom.rScr.z == score_)
        {
            res_.x = row + 2;
            res_.y = col;
        }
        else if (atom.rScr.w == score_)
        {
            res_.x = row + 3;
            res_.y = col;
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

    VEC2_ASSIGN(hBus[col - 1], make_int2(atom.rScr.w, del));

    const int vBusIdx = (row >> 2) % (gridDim.x * blockDim.x);
    vBus.mch[vBusIdx] = atom.up.x;
    VEC4_ASSIGN(vBus.scr[vBusIdx], atom.lScr);
    VEC4_ASSIGN(vBus.aff[vBusIdx], atom.lAff);
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// CPU KERNELS

static void *kernel(void *params)
{

    Context *context = (Context *)params;

    int *queryStart = context->queryStart;
    int *targetStart = context->targetStart;
    int score = context->score;
    Chain *query = context->query;
    int queryFrontGap = context->queryFrontGap;
    Chain *target = context->target;
    Scorer *scorer = context->scorer;
    int card = context->card;

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
    int gapDiff = gapOpen - gapExtend;
    int scorerLen = scorerGetMaxCode(scorer);
    int subLen = scorerLen + 1;
    int scalar = scorerIsScalar(scorer);

    TIMER_START("Sw find start %d %d", rows, cols);

    cudaDeviceProp properties;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&properties, card));

    int threads;
    int blocks;

    maxWorkGroups(card, BLOCKS_SM2, THREADS_SM2, cols, &blocks, &threads);

    int cellHeight = 4 * threads;
    int rowsGpu = rows + (4 - rows % 4) % 4;
    int dRow = (rowsGpu - rows);

    int colsGpu = cols + (blocks - cols % blocks) % blocks;
    int cellWidth = colsGpu / blocks;

    int diagonals = blocks + (int)ceil((float)rowsGpu / cellHeight);

    int maxScore = scorerGetMaxScore(scorer);
    int minMatch = maxScore ? score / maxScore : 0;
    int pLeft = rows - minMatch;
    int pRight = cols - minMatch;

    int memoryUsedGpu = 0;
    int memoryUsedCpu = 0;

    /*
    LOG("Rows cpu: %d, gpu: %d", rows, rowsGpu);
    LOG("Columns cpu: %d, gpu: %d", cols, colsGpu);
    LOG("Cell h: %d, w: %d", cellHeight, cellWidth);
    LOG("Diagonals: %d", diagonals);
    LOG("Deformation: %d %d", pRight, pLeft);
    */

    //**************************************************************************
    // PADD CHAINS

    char *rowCpu = (char *)malloc(rowsGpu * sizeof(char));
    memset(rowCpu + rows, scorerLen, dRow * sizeof(char));
    chainCopyCodes(query, rowCpu);
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
    int2 *hBusCpu = (int2 *)malloc(hBusSize);
    int2 *hBusGpu;
    for (int i = 0; i < colsGpu; ++i)
    {
        hBusCpu[i] = make_int2(-gapOpen - gapExtend * i, SCORE_MIN);
    }
    CUDA_SAFE_CALL(cudaMalloc(&hBusGpu, hBusSize));
    CUDA_SAFE_CALL(cudaMemcpy(hBusGpu, hBusCpu, hBusSize, TO_GPU));
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
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(gapDiff_, &gapDiff, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(queryFrontGap_, &queryFrontGap, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(scorerLen_, &scorerLen, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(subLen_, &subLen, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(rows_, &rowsGpu, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(cols_, &colsGpu, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(cellWidth_, &cellWidth, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(pLeft_, &pLeft, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(pRight_, &pRight, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(score_, &score, sizeof(int)));

    /*
    LOG("Memory used CPU: %fMB", memoryUsedCpu / 1024. / 1024.);
    LOG("Memory used GPU: %fMB", memoryUsedGpu / 1024. / 1024.);
    */

    //**************************************************************************

    //**************************************************************************
    // KERNEL RUN

    // TIMER_START("Kernel");

    int2 res = {-1, -1};
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(res_, &res, sizeof(int2)));

    for (int diagonal = 0; diagonal < diagonals; ++diagonal)
    {

        if (scalar)
        {
            if (subCpu[0] >= subCpu[1])
            {
                solveShort<<<blocks, threads>>>(diagonal, vBusGpu, hBusGpu, SubScalar());
            }
            else
            {
                solveShort<<<blocks, threads>>>(diagonal, vBusGpu, hBusGpu, SubScalarRev());
            }
        }
        else
        {
            solveShort<<<blocks, threads>>>(diagonal, vBusGpu, hBusGpu, SubVector());
        }

        CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&res, res_, sizeof(int2), 0, FROM_GPU));
        if (res.x != -1)
            break;

        if (scalar)
        {
            if (subCpu[0] >= subCpu[1])
            {
                solveLong<<<blocks, threads>>>(diagonal, vBusGpu, hBusGpu, SubScalar());
            }
            else
            {
                solveLong<<<blocks, threads>>>(diagonal, vBusGpu, hBusGpu, SubScalarRev());
            }
        }
        else
        {
            solveLong<<<blocks, threads>>>(diagonal, vBusGpu, hBusGpu, SubVector());
        }

        CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&res, res_, sizeof(int2), 0, FROM_GPU));
        if (res.x != -1)
            break;
    }

    // TIMER_STOP;

    //**************************************************************************

    //**************************************************************************
    // SAVE RESULTS

    if (res.x >= rows)
    {
        res.y += rows - res.x - 1;
        res.x += rows - res.x - 1;
    }

    if (res.y >= cols)
    {
        res.x += cols - res.y - 1;
        res.y += cols - res.y - 1;
    }

    if (res.x == -1)
    {
        LOG("Not found: %d", score);
    }
    else
    {
        LOG("Found: %d (%d, %d) (%d, %d)", score, res.x, res.y, rows, cols);
    }

    *queryStart = res.x;
    *targetStart = res.y;
    //**************************************************************************

    //**************************************************************************
    // CLEAN MEMORY

    free(subCpu);
    free(rowCpu);
    free(colCpu);
    free(hBusCpu);

    CUDA_SAFE_CALL(cudaFree(subGpu));
    CUDA_SAFE_CALL(cudaFree(rowGpu));
    CUDA_SAFE_CALL(cudaFree(colGpu));
    CUDA_SAFE_CALL(cudaFree(vBusGpu.mch));
    CUDA_SAFE_CALL(cudaFree(vBusGpu.scr));
    CUDA_SAFE_CALL(cudaFree(vBusGpu.aff));
    CUDA_SAFE_CALL(cudaFree(hBusGpu));

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
