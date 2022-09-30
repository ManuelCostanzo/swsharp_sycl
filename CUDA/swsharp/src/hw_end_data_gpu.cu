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

#include <stdio.h>
#include <stdlib.h>

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
#define BLOCKS_SM2 240

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

static __constant__ int dRow_;
static __constant__ int rows_;
static __constant__ int cols_;

static __constant__ int cellWidth_;

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

extern void hwEndDataGpu(int *queryEnd, int *targetEnd, int *outScore,
                         Chain *query, Chain *target, Scorer *scorer, int score, int card,
                         Thread *thread);

//******************************************************************************

//******************************************************************************
// PRIVATE

// With visual c++ compiler and prototypes declared cuda global memory variables
// do not work. No questions asked.
#ifndef _WIN32

__device__ static int gap(int idx);
__device__ static int aff(int idx);

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

extern void hwEndDataGpu(int *queryEnd, int *targetEnd, int *outScore,
                         Chain *query, Chain *target, Scorer *scorer, int score, int card,
                         Thread *thread)
{

    Context *param = (Context *)malloc(sizeof(Context));

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
    if (idx == dRow_ - 1)
        return 0;
    if (idx < dRow_ - 1)
        return 0;
    return -gapOpen_ - gapExtend_ * (idx - dRow_);
}

__device__ static int aff(int idx)
{
    return SCORE_MIN;
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

    Atom atom;

    if (0 <= row && row < rows_ && col > 0)
    {
        atom.mch = vBus.mch[(row >> 2) % (gridDim.x * blockDim.x)];
        VEC4_ASSIGN(atom.lScr, vBus.scr[(row >> 2) % (gridDim.x * blockDim.x)]);
        VEC4_ASSIGN(atom.lAff, vBus.aff[(row >> 2) % (gridDim.x * blockDim.x)]);
    }
    else
    {
        atom.mch = gap(row - 1);
        atom.lScr = make_int4(gap(row), gap(row + 1), gap(row + 2), gap(row + 3));
        atom.lAff = make_int4(aff(row), aff(row + 1), aff(row + 2), aff(row + 3));
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

            atom.mch = atom.up.x;
            VEC4_ASSIGN(atom.lScr, atom.rScr);
            VEC4_ASSIGN(atom.lAff, atom.rAff);
        }

        __syncthreads();

        if (0 <= row && row < rows_)
        {

            if (threadIdx.x == blockDim.x - 1 || i == blockDim.x - 1 || row == rows_ - 4)
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

            atom.mch = gap(row - 1);
            atom.lScr = make_int4(gap(row), gap(row + 1), gap(row + 2), gap(row + 3));
            atom.lAff = make_int4(aff(row), aff(row + 1), aff(row + 2), aff(row + 3));

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

        atom.mch = atom.up.x;
        VEC4_ASSIGN(atom.lScr, atom.rScr);
        VEC4_ASSIGN(atom.lAff, atom.rAff);

        __syncthreads();

        if (threadIdx.x == blockDim.x - 1 || row == rows_ - 4)
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

    vBus.mch[(row >> 2) % (gridDim.x * blockDim.x)] = atom.up.x;
    VEC4_ASSIGN(vBus.scr[(row >> 2) % (gridDim.x * blockDim.x)], atom.lScr);
    VEC4_ASSIGN(vBus.aff[(row >> 2) % (gridDim.x * blockDim.x)], atom.lAff);
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

        atom.mch = atom.up.x;
        VEC4_ASSIGN(atom.lScr, atom.rScr);
        VEC4_ASSIGN(atom.lAff, atom.rAff);

        __syncthreads();

        if (threadIdx.x == blockDim.x - 1 || row == rows_ - 4)
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

    vBus.mch[(row >> 2) % (gridDim.x * blockDim.x)] = atom.up.x;
    VEC4_ASSIGN(vBus.scr[(row >> 2) % (gridDim.x * blockDim.x)], atom.lScr);
    VEC4_ASSIGN(vBus.aff[(row >> 2) % (gridDim.x * blockDim.x)], atom.lAff);
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// CPU KERNELS

static void *kernel(void *params)
{

    Context *context = (Context *)params;

    int *queryEnd = context->queryEnd;
    int *targetEnd = context->targetEnd;
    int *outScore = context->outScore;
    Chain *query = context->query;
    Chain *target = context->target;
    Scorer *scorer = context->scorer;
    // int score = context->score;
    int card = context->card;

    int currentCard;
    CUDA_SAFE_CALL(cudaGetDevice(&currentCard));
    if (currentCard != card)
    {
        // CUDA_SAFE_CALL(cudaThreadExit());
        CUDA_SAFE_CALL(cudaSetDevice(card));
    }

    TIMER_START("Hw align %d %d", chainGetLength(query), chainGetLength(target));

    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);
    int scorerLen = scorerGetMaxCode(scorer);
    int subLen = scorerLen + 1;
    int scalar = scorerIsScalar(scorer);

    cudaDeviceProp properties;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&properties, card));

    int threads;
    int blocks;

    maxWorkGroups(card, BLOCKS_SM2, THREADS_SM2, cols, &blocks, &threads);

    int cellHeight = 4 * threads;
    int rowsGpu = rows + (4 - rows % 4) % 4;
    int dRow = rowsGpu - rows;

    int colsGpu = (cols + 4) + (blocks - (cols + 4) % blocks) % blocks;
    int cellWidth = colsGpu / blocks;

    int diagonals = blocks + (int)ceil((float)rowsGpu / cellHeight);

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
    memset(rowCpu, scorerLen, dRow * sizeof(char));
    chainCopyCodes(query, rowCpu + dRow);
    memoryUsedCpu += rowsGpu * sizeof(char);

    char *colCpu = (char *)malloc(colsGpu * sizeof(char));
    memset(colCpu + cols, scorerLen + scalar, (colsGpu - cols) * sizeof(char));
    chainCopyCodes(target, colCpu);
    memoryUsedCpu += colsGpu * sizeof(char);

    //**************************************************************************

    //**************************************************************************
    // INIT GPU

    int rowSize = rowsGpu * sizeof(char);
    char *rowGpu;
    CUDA_SAFE_CALL(cudaMalloc(&rowGpu, rowSize));
    CUDA_SAFE_CALL(cudaMemcpy(rowGpu, rowCpu, rowSize, TO_GPU));
    CUDA_SAFE_CALL(cudaBindTexture(NULL, rowTexture, rowGpu, rowSize));
    memoryUsedGpu += rowSize;

    int colSize = colsGpu * sizeof(char);
    char *colGpu;
    CUDA_SAFE_CALL(cudaMalloc(&colGpu, colSize));
    CUDA_SAFE_CALL(cudaMemcpy(colGpu, colCpu, colSize, TO_GPU));
    CUDA_SAFE_CALL(cudaBindTexture(NULL, colTexture, colGpu, colSize));
    memoryUsedGpu += colSize;

    int hBusSize = colsGpu * sizeof(int2);
    int2 *hBusCpu = (int2 *)malloc(hBusSize);
    int2 *hBusGpu;
    for (int i = 0; i < colsGpu; ++i)
    {
        hBusCpu[i] = make_int2(0, SCORE_MIN);
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
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(scorerLen_, &scorerLen, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(subLen_, &subLen, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(rows_, &rowsGpu, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(cols_, &colsGpu, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(cellWidth_, &cellWidth, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(dRow_, &dRow, sizeof(int)));

    /*
    LOG("Memory used CPU: %fMB", memoryUsedCpu / 1024. / 1024.);
    LOG("Memory used GPU: %fMB", memoryUsedGpu / 1024. / 1024.);
    */

    //**************************************************************************

    //**************************************************************************
    // KERNEL RUN

    // TIMER_START("Kernel");

    for (int diagonal = 0; diagonal < diagonals; ++diagonal)
    {
        if (scalar)
        {
            solveShort<<<blocks, threads>>>(diagonal, vBusGpu, hBusGpu, SubScalarRev());
            solveLong<<<blocks, threads>>>(diagonal, vBusGpu, hBusGpu, SubScalarRev());
        }
        else
        {
            solveShort<<<blocks, threads>>>(diagonal, vBusGpu, hBusGpu, SubVector());
            solveLong<<<blocks, threads>>>(diagonal, vBusGpu, hBusGpu, SubVector());
        }
    }

    // TIMER_STOP;
    //**************************************************************************

    //**************************************************************************
    // SAVE RESULTS

    CUDA_SAFE_CALL(cudaMemcpy(hBusCpu, hBusGpu, hBusSize, FROM_GPU));

    *queryEnd = rows - 1;

    *outScore = hBusCpu[0].x;
    *targetEnd = 0;

    for (int i = 1; i < cols; ++i)
    {
        if (hBusCpu[i].x > *outScore)
        {
            *outScore = hBusCpu[i].x;
            *targetEnd = i;
        }
    }

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
