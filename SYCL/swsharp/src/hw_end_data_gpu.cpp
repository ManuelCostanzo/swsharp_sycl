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

#include <CL/sycl.hpp>
#ifdef HIP
namespace sycl = cl::sycl;
#endif
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
#include <cmath>

#define MAX_THREADS MAX(THREADS_SM1, THREADS_SM2)

#define THREADS_SM1 64
#define BLOCKS_SM1 240

#define THREADS_SM2 128
#define BLOCKS_SM2 240

#define SCORE4_MIN make_int4(SCORE_MIN, SCORE_MIN, SCORE_MIN, SCORE_MIN)

typedef struct Atom {
  int mch;
  sycl::int2 up;
  sycl::int4 lScr;
  sycl::int4 lAff;
  sycl::int4 rScr;
  sycl::int4 rAff;
} Atom;

typedef struct VBus {
  int *mch;
  sycl::int4 *scr;
  sycl::int4 *aff;
} VBus;

typedef struct Context {
  int *queryEnd;
  int *targetEnd;
  int *outScore;
  Chain *query;
  Chain *target;
  Scorer *scorer;
  int score;
  int card;
} Context;

//******************************************************************************
// PUBLIC

extern void hwEndDataGpu(int *queryEnd, int *targetEnd, int *outScore,
                         Chain *query, Chain *target, Scorer *scorer, int score,
                         int card, Thread *thread);

//******************************************************************************

//******************************************************************************
// PRIVATE

static int gap(int idx, int gapOpen_, int gapExtend_, int dRow_);
static int aff(int idx);

template <class Sub>
static void solveShortDelegated(int d, VBus vBus, sycl::int2 *hBus, Sub sub,
                                sycl::nd_item<1> item_ct1, int gapOpen_,
                                int gapExtend_, int dRow_, int rows_, int cols_,
                                int cellWidth_, int scorerLen_, int subLen_,
                                int match_, int mismatch_, int *hBusScrShr,
                                int *hBusAffShr, sycl::char4 *rowGpu,
                                char *colGpu, int *subGpu);

template <class Sub>
static void solveShortNormal(int d, VBus vBus, sycl::int2 *hBus, Sub sub,
                             sycl::nd_item<1> item_ct1, int gapOpen_,
                             int gapExtend_, int dRow_, int rows_, int cols_,
                             int cellWidth_, int scorerLen_, int subLen_,
                             int match_, int mismatch_, int *hBusScrShr,
                             int *hBusAffShr, sycl::char4 *rowGpu, char *colGpu,
                             int *subGpu);

template <class Sub>

static void solveShort(int d, VBus vBus, sycl::int2 *hBus, Sub sub,
                       sycl::nd_item<1> item_ct1, int gapOpen_, int gapExtend_,
                       int dRow_, int rows_, int cols_, int cellWidth_,
                       int scorerLen_, int subLen_, int match_, int mismatch_,
                       int *hBusScrShr, int *hBusAffShr, sycl::char4 *rowGpu,
                       char *colGpu, int *subGpu);

template <class Sub>
static void solveLong(int d, VBus vBus, sycl::int2 *hBus, Sub sub,
                      sycl::nd_item<1> item_ct1, int gapOpen_, int gapExtend_,
                      int dRow_, int rows_, int cols_, int cellWidth_,
                      int scorerLen_, int subLen_, int match_, int mismatch_,
                      int *hBusScrShr, int *hBusAffShr, sycl::char4 *rowGpu,
                      char *colGpu, int *subGpu);

static void *hwEndDataGpuKernel(void *params);

//******************************************************************************

//******************************************************************************
// PUBLIC

extern void hwEndDataGpu(int *queryEnd, int *targetEnd, int *outScore,
                         Chain *query, Chain *target, Scorer *scorer, int score,
                         int card, Thread *thread) {

  Context *param = (Context *)malloc(sizeof(Context));

  param->queryEnd = queryEnd;
  param->targetEnd = targetEnd;
  param->outScore = outScore;
  param->query = query;
  param->target = target;
  param->scorer = scorer;
  param->score = score;
  param->card = card;

  if (thread == NULL) {
    hwEndDataGpuKernel(param);
  } else {
    threadCreate(thread, hwEndDataGpuKernel, (void *)param);
  }
}

//******************************************************************************

//******************************************************************************
// PRIVATE

//------------------------------------------------------------------------------
// FUNCTORS

class SubScalarRev {
public:
  int operator()(char a, char b, int scorerLen_, int match_, int mismatch_,
                 int subLen_, int *subGpu) {
    return (a == b ? match_ : mismatch_) * (a < scorerLen_ && b < scorerLen_);
  }
};

class SubVector {
public:
  int operator()(char a, char b, int scorerLen_, int match_, int mismatch_,
                 int subLen_, int *subGpu) {
    return subGpu[(a * subLen_) + b];
  }
};

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// GPU KERNELS

static int gap(int idx, int gapOpen_, int gapExtend_, int dRow_) {
  if (idx == dRow_ - 1)
    return 0;
  if (idx < dRow_ - 1)
    return 0;
  return -gapOpen_ - gapExtend_ * (idx - dRow_);
}

static int aff(int idx) { return SCORE_MIN; }

template <class Sub>
static void solveShortDelegated(int d, VBus vBus, sycl::int2 *hBus, Sub sub,
                                sycl::nd_item<1> item_ct1, int gapOpen_,
                                int gapExtend_, int dRow_, int rows_, int cols_,
                                int cellWidth_, int scorerLen_, int subLen_,
                                int match_, int mismatch_, int *hBusScrShr,
                                int *hBusAffShr, sycl::char4 *rowGpu,
                                char *colGpu, int *subGpu) {

  const int groupRangeId = item_ct1.get_group_range(0);
  const int groupId = item_ct1.get_group(0);
  const int localId = item_ct1.get_local_id(0);
  const int localRangeId = item_ct1.get_local_range(0);

  int row = (d + groupId - groupRangeId + 1) * (localRangeId * 4) + localId * 4;
  int col = cellWidth_ * (groupRangeId - groupId - 1) - localId;

  if (row < 0)
    return;

  row -= (col < 0) * (groupRangeId * localRangeId * 4);
  col += (col < 0) * cols_;

  Atom atom;

  if (0 <= row && row < rows_ && col > 0) {
    atom.mch = vBus.mch[(row >> 2) % (groupRangeId * localRangeId)];
    atom.lScr = vBus.scr[(row >> 2) % (groupRangeId * localRangeId)];
    atom.lAff = vBus.aff[(row >> 2) % (groupRangeId * localRangeId)];
  } else {
    atom.mch = gap(row - 1, gapOpen_, gapExtend_, dRow_);
    atom.lScr = sycl::int4(gap(row, gapOpen_, gapExtend_, dRow_),
                           gap(row + 1, gapOpen_, gapExtend_, dRow_),
                           gap(row + 2, gapOpen_, gapExtend_, dRow_),
                           gap(row + 3, gapOpen_, gapExtend_, dRow_));
    atom.lAff = sycl::int4(aff(row), aff(row + 1), aff(row + 2), aff(row + 3));
  }

  hBusScrShr[localId] = hBus[col].x();
  hBusAffShr[localId] = hBus[col].y();

  sycl::char4 rowCodes;

  if (0 <= row && row < rows_)
    rowCodes = rowGpu[row >> 2];

  int del;

  for (int i = 0; i < localRangeId; ++i) {

    if (0 <= row && row < rows_) {

      char columnCode = colGpu[col];

      if (localId == 0) {
        atom.up = hBus[col];
      } else {
        atom.up.x() = hBusScrShr[localId];
        atom.up.y() = hBusAffShr[localId];
      }

      del = sycl::max((atom.up.x() - gapOpen_), (atom.up.y() - gapExtend_));
      int ins =
          sycl::max((atom.lScr.x() - gapOpen_), (atom.lAff.x() - gapExtend_));

      int mch = atom.mch + sub(columnCode, rowCodes.x(), scorerLen_, match_,
                               mismatch_, subLen_, subGpu);

      atom.rScr.x() = MAX3(mch, del, ins);
      atom.rAff.x() = ins;

      del = sycl::max((atom.rScr.x() - gapOpen_), (del - gapExtend_));
      ins = sycl::max((atom.lScr.y() - gapOpen_), (atom.lAff.y() - gapExtend_));

      mch = atom.lScr.x() + sub(columnCode, rowCodes.y(), scorerLen_, match_,
                                mismatch_, subLen_, subGpu);

      atom.rScr.y() = MAX3(mch, del, ins);
      atom.rAff.y() = ins;

      del = sycl::max((atom.rScr.y() - gapOpen_), (del - gapExtend_));
      ins = sycl::max((atom.lScr.z() - gapOpen_), (atom.lAff.z() - gapExtend_));

      mch = atom.lScr.y() + sub(columnCode, rowCodes.z(), scorerLen_, match_,
                                mismatch_, subLen_, subGpu);

      atom.rScr.z() = MAX3(mch, del, ins);
      atom.rAff.z() = ins;

      del = sycl::max((atom.rScr.z() - gapOpen_), (del - gapExtend_));
      ins = sycl::max((atom.lScr.w() - gapOpen_), (atom.lAff.w() - gapExtend_));

      mch = atom.lScr.z() + sub(columnCode, rowCodes.w(), scorerLen_, match_,
                                mismatch_, subLen_, subGpu);

      atom.rScr.w() = MAX3(mch, del, ins);
      atom.rAff.w() = ins;

      atom.mch = atom.up.x();
      atom.lScr = atom.rScr;
      atom.lAff = atom.rAff;
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (0 <= row && row < rows_) {

      if (localId == localRangeId - 1 || i == localRangeId - 1 ||
          row == rows_ - 4) {
        hBus[col] = sycl::int2(atom.rScr.w(), del);
      } else {
        hBusScrShr[localId + 1] = atom.rScr.w();
        hBusAffShr[localId + 1] = del;
      }
    }

    ++col;

    if (col == cols_) {

      col = 0;
      row = row + groupRangeId * localRangeId * 4;

      atom.mch = gap(row - 1, gapOpen_, gapExtend_, dRow_);
      atom.lScr = sycl::int4(gap(row, gapOpen_, gapExtend_, dRow_),
                             gap(row + 1, gapOpen_, gapExtend_, dRow_),
                             gap(row + 2, gapOpen_, gapExtend_, dRow_),
                             gap(row + 3, gapOpen_, gapExtend_, dRow_));
      atom.lAff =
          sycl::int4(aff(row), aff(row + 1), aff(row + 2), aff(row + 3));

      rowCodes = rowGpu[row >> 2];
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);
  }

  if (row < 0 || row >= rows_)
    return;

  vBus.mch[(row >> 2) % (groupRangeId * localRangeId)] = atom.up.x();
  vBus.scr[(row >> 2) % (groupRangeId * localRangeId)] = atom.lScr;
  vBus.aff[(row >> 2) % (groupRangeId * localRangeId)] = atom.lAff;
}

template <class Sub>
static void solveShortNormal(int d, VBus vBus, sycl::int2 *hBus, Sub sub,
                             sycl::nd_item<1> item_ct1, int gapOpen_,
                             int gapExtend_, int dRow_, int rows_, int cols_,
                             int cellWidth_, int scorerLen_, int subLen_,
                             int match_, int mismatch_, int *hBusScrShr,
                             int *hBusAffShr, sycl::char4 *rowGpu, char *colGpu,
                             int *subGpu) {
  const int groupRangeId = item_ct1.get_group_range(0);
  const int groupId = item_ct1.get_group(0);
  const int localId = item_ct1.get_local_id(0);
  const int localRangeId = item_ct1.get_local_range(0);

  int row = (d + groupId - groupRangeId + 1) * (localRangeId * 4) + localId * 4;
  int col = cellWidth_ * (groupRangeId - groupId - 1) - localId;

  if (row < 0 || row >= rows_)
    return;

  Atom atom;

  atom.mch = vBus.mch[(row >> 2) % (groupRangeId * localRangeId)];
  atom.lScr = vBus.scr[(row >> 2) % (groupRangeId * localRangeId)];
  atom.lAff = vBus.aff[(row >> 2) % (groupRangeId * localRangeId)];

  hBusScrShr[localId] = hBus[col].x();
  hBusAffShr[localId] = hBus[col].y();

  sycl::char4 rowCodes;
  VEC4_ASSIGN(rowCodes, rowGpu[row >> 2]);

  int del;

  for (int i = 0; i < localRangeId; ++i, ++col) {
    char columnCode = colGpu[col];

    if (localId == 0) {
      atom.up = hBus[col];
    } else {
      atom.up.x() = hBusScrShr[localId];
      atom.up.y() = hBusAffShr[localId];
    }

    del = sycl::max((atom.up.x() - gapOpen_), (atom.up.y() - gapExtend_));
    int ins =
        sycl::max((atom.lScr.x() - gapOpen_), (atom.lAff.x() - gapExtend_));

    int mch = atom.mch + sub(columnCode, rowCodes.x(), scorerLen_, match_,
                             mismatch_, subLen_, subGpu);

    atom.rScr.x() = MAX3(mch, del, ins);
    atom.rAff.x() = ins;

    del = sycl::max((atom.rScr.x() - gapOpen_), (del - gapExtend_));
    ins = sycl::max((atom.lScr.y() - gapOpen_), (atom.lAff.y() - gapExtend_));

    mch = atom.lScr.x() + sub(columnCode, rowCodes.y(), scorerLen_, match_,
                              mismatch_, subLen_, subGpu);

    atom.rScr.y() = MAX3(mch, del, ins);
    atom.rAff.y() = ins;

    del = sycl::max((atom.rScr.y() - gapOpen_), (del - gapExtend_));
    ins = sycl::max((atom.lScr.z() - gapOpen_), (atom.lAff.z() - gapExtend_));

    mch = atom.lScr.y() + sub(columnCode, rowCodes.z(), scorerLen_, match_,
                              mismatch_, subLen_, subGpu);

    atom.rScr.z() = MAX3(mch, del, ins);
    atom.rAff.z() = ins;

    del = sycl::max((atom.rScr.z() - gapOpen_), (del - gapExtend_));
    ins = sycl::max((atom.lScr.w() - gapOpen_), (atom.lAff.w() - gapExtend_));

    mch = atom.lScr.z() + sub(columnCode, rowCodes.w(), scorerLen_, match_,
                              mismatch_, subLen_, subGpu);

    atom.rScr.w() = MAX3(mch, del, ins);
    atom.rAff.w() = ins;

    atom.mch = atom.up.x();
    atom.lScr = atom.rScr;
    atom.lAff = atom.rAff;

    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (localId == localRangeId - 1 || row == rows_ - 4) {
      hBus[col] = sycl::int2(atom.rScr.w(), del);
    } else {
      hBusScrShr[localId + 1] = atom.rScr.w();
      hBusAffShr[localId + 1] = del;
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);
  }

  hBus[col - 1] = sycl::int2(atom.rScr.w(), del);

  vBus.mch[(row >> 2) % (groupRangeId * localRangeId)] = atom.up.x();
  vBus.scr[(row >> 2) % (groupRangeId * localRangeId)] = atom.lScr;
  vBus.aff[(row >> 2) % (groupRangeId * localRangeId)] = atom.lAff;
}

template <class Sub>
static void solveShort(int d, VBus vBus, sycl::int2 *hBus, Sub sub,
                       sycl::nd_item<1> item_ct1, int gapOpen_, int gapExtend_,
                       int dRow_, int rows_, int cols_, int cellWidth_,
                       int scorerLen_, int subLen_, int match_, int mismatch_,
                       int *hBusScrShr, int *hBusAffShr, sycl::char4 *rowGpu,
                       char *colGpu, int *subGpu) {
  const int groupRangeId = item_ct1.get_group_range(0);
  const int groupId = item_ct1.get_group(0);

  if (groupId == (groupRangeId - 1)) {
    solveShortDelegated(d, vBus, hBus, sub, item_ct1, gapOpen_, gapExtend_,
                        dRow_, rows_, cols_, cellWidth_, scorerLen_, subLen_,
                        match_, mismatch_, hBusScrShr, hBusAffShr, rowGpu,
                        colGpu, subGpu);
  } else {
    solveShortNormal(d, vBus, hBus, sub, item_ct1, gapOpen_, gapExtend_, dRow_,
                     rows_, cols_, cellWidth_, scorerLen_, subLen_, match_,
                     mismatch_, hBusScrShr, hBusAffShr, rowGpu, colGpu, subGpu);
  }
}

template <class Sub>
static void solveLong(int d, VBus vBus, sycl::int2 *hBus, Sub sub,
                      sycl::nd_item<1> item_ct1, int gapOpen_, int gapExtend_,
                      int dRow_, int rows_, int cols_, int cellWidth_,
                      int scorerLen_, int subLen_, int match_, int mismatch_,
                      int *hBusScrShr, int *hBusAffShr, sycl::char4 *rowGpu,
                      char *colGpu, int *subGpu) {
  const int groupRangeId = item_ct1.get_group_range(0);
  const int groupId = item_ct1.get_group(0);
  const int localId = item_ct1.get_local_id(0);
  const int localRangeId = item_ct1.get_local_range(0);

  int row = (d + groupId - groupRangeId + 1) * (localRangeId * 4) + localId * 4;
  int col = cellWidth_ * (groupRangeId - groupId - 1) - localId + localRangeId;

  if (row < 0 || row >= rows_)
    return;

  Atom atom;

  atom.mch = vBus.mch[(row >> 2) % (groupRangeId * localRangeId)];
  atom.lScr = vBus.scr[(row >> 2) % (groupRangeId * localRangeId)];
  atom.lAff = vBus.aff[(row >> 2) % (groupRangeId * localRangeId)];

  hBusScrShr[localId] = hBus[col].x();
  hBusAffShr[localId] = hBus[col].y();

  sycl::char4 rowCodes;
  VEC4_ASSIGN(rowCodes, rowGpu[row >> 2]);

  int del;

  for (int i = 0; i < cellWidth_ - localRangeId; ++i, ++col) {

    char columnCode = colGpu[col];

    if (localId == 0) {
      atom.up = hBus[col];
    } else {
      atom.up.x() = hBusScrShr[localId];
      atom.up.y() = hBusAffShr[localId];
    }

    del = sycl::max((atom.up.x() - gapOpen_), (atom.up.y() - gapExtend_));
    int ins =
        sycl::max((atom.lScr.x() - gapOpen_), (atom.lAff.x() - gapExtend_));

    int mch = atom.mch + sub(columnCode, rowCodes.x(), scorerLen_, match_,
                             mismatch_, subLen_, subGpu);

    atom.rScr.x() = MAX3(mch, del, ins);
    atom.rAff.x() = ins;

    del = sycl::max((atom.rScr.x() - gapOpen_), (del - gapExtend_));
    ins = sycl::max((atom.lScr.y() - gapOpen_), (atom.lAff.y() - gapExtend_));

    mch = atom.lScr.x() + sub(columnCode, rowCodes.y(), scorerLen_, match_,
                              mismatch_, subLen_, subGpu);

    atom.rScr.y() = MAX3(mch, del, ins);
    atom.rAff.y() = ins;

    del = sycl::max((atom.rScr.y() - gapOpen_), (del - gapExtend_));
    ins = sycl::max((atom.lScr.z() - gapOpen_), (atom.lAff.z() - gapExtend_));

    mch = atom.lScr.y() + sub(columnCode, rowCodes.z(), scorerLen_, match_,
                              mismatch_, subLen_, subGpu);

    atom.rScr.z() = MAX3(mch, del, ins);
    atom.rAff.z() = ins;

    del = sycl::max((atom.rScr.z() - gapOpen_), (del - gapExtend_));
    ins = sycl::max((atom.lScr.w() - gapOpen_), (atom.lAff.w() - gapExtend_));

    mch = atom.lScr.z() + sub(columnCode, rowCodes.w(), scorerLen_, match_,
                              mismatch_, subLen_, subGpu);

    atom.rScr.w() = MAX3(mch, del, ins);
    atom.rAff.w() = ins;

    atom.mch = atom.up.x();
    atom.lScr = atom.rScr;
    atom.lAff = atom.rAff;

    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (localId == localRangeId - 1 || row == rows_ - 4) {
      hBus[col] = sycl::int2(atom.rScr.w(), del);
    } else {
      hBusScrShr[localId + 1] = atom.rScr.w();
      hBusAffShr[localId + 1] = del;
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);
  }

  hBus[col - 1] = sycl::int2(atom.rScr.w(), del);

  vBus.mch[(row >> 2) % (groupRangeId * localRangeId)] = atom.up.x();
  vBus.scr[(row >> 2) % (groupRangeId * localRangeId)] = atom.lScr;
  vBus.aff[(row >> 2) % (groupRangeId * localRangeId)] = atom.lAff;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// CPU KERNELS

static void *hwEndDataGpuKernel(void *params) try {

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

  sycl::queue dev_q = queues[card];

  int rows = chainGetLength(query);
  int cols = chainGetLength(target);
  int gapOpen = scorerGetGapOpen(scorer);
  int gapExtend = scorerGetGapExtend(scorer);
  int scorerLen = scorerGetMaxCode(scorer);
  int subLen = scorerLen + 1;
  int scalar = scorerIsScalar(scorer);

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
  sycl::char4 *rowGpu = sycl::malloc_device<sycl::char4>(rowsGpu, dev_q);
  dev_q.memcpy(rowGpu, rowCpu, rowSize).wait();

  memoryUsedGpu += rowSize;

  int colSize = colsGpu * sizeof(char);
  char *colGpu = sycl::malloc_device<char>(colsGpu, dev_q);

  dev_q.memcpy(colGpu, colCpu, colSize).wait();

  memoryUsedGpu += colSize;

  int hBusSize = colsGpu * sizeof(sycl::int2);

  sycl::int2 *hBus = sycl::malloc_shared<sycl::int2>(colsGpu, dev_q);
  for (int i = 0; i < colsGpu; ++i) {
    hBus[i] = sycl::int2(0, SCORE_MIN);
  }

  memoryUsedCpu += hBusSize;
  memoryUsedGpu += hBusSize;

  VBus vBusGpu;
  vBusGpu.mch = sycl::malloc_device<int>(blocks * threads, dev_q);
  vBusGpu.scr = sycl::malloc_device<sycl::int4>(blocks * threads, dev_q);
  vBusGpu.aff = sycl::malloc_device<sycl::int4>(blocks * threads, dev_q);

  dev_q.memset(vBusGpu.mch, 0, blocks * threads * sizeof(int)).wait();
  dev_q.memset(vBusGpu.scr, 0, blocks * threads * sizeof(sycl::int4)).wait();
  dev_q.memset(vBusGpu.aff, 0, blocks * threads * sizeof(sycl::int4)).wait();

  memoryUsedGpu += blocks * threads * sizeof(int);
  memoryUsedGpu += blocks * threads * sizeof(sycl::int4);
  memoryUsedGpu += blocks * threads * sizeof(sycl::int4);

  size_t subSize = subLen * subLen * sizeof(int);

  int *sub = sycl::malloc_shared<int>(subLen * subLen, dev_q);
  for (int i = 0; i < subLen; ++i) {
    for (int j = 0; j < subLen; ++j) {
      if (i < scorerLen && j < scorerLen) {
        sub[i * subLen + j] = scorerScore(scorer, i, j);
      } else {
        sub[i * subLen + j] = 0;
      }
    }
  }

  memoryUsedCpu += subSize;
  memoryUsedGpu += subSize;

  int match = sub[0];
  int mismatch = sub[1];

  /*
  LOG("Memory used CPU: %fMB", memoryUsedCpu / 1024. / 1024.);
  LOG("Memory used GPU: %fMB", memoryUsedGpu / 1024. / 1024.);
  */

  //**************************************************************************

  //**************************************************************************
  // KERNEL RUN

  // TIMER_START("Kernel");

  int *ga = sycl::malloc_shared<int>(1, dev_q);
  int *ge = sycl::malloc_shared<int>(1, dev_q);

  *ga = gapOpen;
  *ge = gapExtend;

  for (int diagonal = 0; diagonal < diagonals; ++diagonal) {
    if (scalar) {
      dev_q
          .submit([&](sycl::handler &cgh) {
            sycl::accessor<int, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                hBusScrShr_acc_ct1(sycl::range<1>(threads), cgh);
            sycl::accessor<int, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                hBusAffShr_acc_ct1(sycl::range<1>(threads), cgh);

            cgh.parallel_for(sycl::nd_range<1>(blocks * threads, threads),
                             [=](sycl::nd_item<1> item_ct1) {
                               solveShort(diagonal, vBusGpu, hBus,
                                          SubScalarRev(), item_ct1, *ga, *ge,
                                          dRow, rowsGpu, colsGpu, cellWidth,
                                          scorerLen, subLen, match, mismatch,
                                          hBusScrShr_acc_ct1.get_pointer(),
                                          hBusAffShr_acc_ct1.get_pointer(),
                                          rowGpu, colGpu, sub);
                             });
          })
          .wait();

      dev_q
          .submit([&](sycl::handler &cgh) {
            sycl::accessor<int, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                hBusScrShr_acc_ct1(sycl::range<1>(threads), cgh);
            sycl::accessor<int, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                hBusAffShr_acc_ct1(sycl::range<1>(threads), cgh);

            cgh.parallel_for(sycl::nd_range<1>(blocks * threads, threads),
                             [=](sycl::nd_item<1> item_ct1) {
                               solveLong(diagonal, vBusGpu, hBus,
                                         SubScalarRev(), item_ct1, *ga, *ge,
                                         dRow, rowsGpu, colsGpu, cellWidth,
                                         scorerLen, subLen, match, mismatch,
                                         hBusScrShr_acc_ct1.get_pointer(),
                                         hBusAffShr_acc_ct1.get_pointer(),
                                         rowGpu, colGpu, sub);
                             });
          })
          .wait();
    } else {
      dev_q
          .submit([&](sycl::handler &cgh) {
            sycl::accessor<int, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                hBusScrShr_acc_ct1(sycl::range<1>(threads), cgh);
            sycl::accessor<int, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                hBusAffShr_acc_ct1(sycl::range<1>(threads), cgh);

            cgh.parallel_for(sycl::nd_range<1>(blocks * threads, threads),
                             [=](sycl::nd_item<1> item_ct1) {
                               solveShort(diagonal, vBusGpu, hBus, SubVector(),
                                          item_ct1, *ga, *ge, dRow, rowsGpu,
                                          colsGpu, cellWidth, scorerLen, subLen,
                                          match, mismatch,
                                          hBusScrShr_acc_ct1.get_pointer(),
                                          hBusAffShr_acc_ct1.get_pointer(),
                                          rowGpu, colGpu, sub);
                             });
          })
          .wait();

      dev_q
          .submit([&](sycl::handler &cgh) {
            sycl::accessor<int, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                hBusScrShr_acc_ct1(sycl::range<1>(threads), cgh);
            sycl::accessor<int, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                hBusAffShr_acc_ct1(sycl::range<1>(threads), cgh);

            cgh.parallel_for(sycl::nd_range<1>(blocks * threads, threads),
                             [=](sycl::nd_item<1> item_ct1) {
                               solveLong(diagonal, vBusGpu, hBus, SubVector(),
                                         item_ct1, *ga, *ge, dRow, rowsGpu,
                                         colsGpu, cellWidth, scorerLen, subLen,
                                         match, mismatch,
                                         hBusScrShr_acc_ct1.get_pointer(),
                                         hBusAffShr_acc_ct1.get_pointer(),
                                         rowGpu, colGpu, sub);
                             });
          })
          .wait();
    }
  }

  // TIMER_STOP;
  //**************************************************************************

  //**************************************************************************
  // SAVE RESULTS

  *queryEnd = rows - 1;

  *outScore = hBus[0].x();
  *targetEnd = 0;

  for (int i = 1; i < cols; ++i) {
    if (hBus[i].x() > *outScore) {
      *outScore = hBus[i].x();
      *targetEnd = i;
    }
  }

  //**************************************************************************

  //**************************************************************************
  // CLEAN MEMORY

  free(rowCpu);
  free(colCpu);

  sycl::free(sub, dev_q);
  sycl::free(rowGpu, dev_q);
  sycl::free(colGpu, dev_q);
  sycl::free(vBusGpu.mch, dev_q);
  sycl::free(vBusGpu.scr, dev_q);
  sycl::free(vBusGpu.aff, dev_q);
  sycl::free(hBus, dev_q);

  free(params);

  //**************************************************************************

  TIMER_STOP;

  return NULL;
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

//------------------------------------------------------------------------------
//******************************************************************************
