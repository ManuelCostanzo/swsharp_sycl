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

#include <algorithm>

#define MAX_THREADS MAX(THREADS_SM1, THREADS_SM2)

#define THREADS_SM1 64
#define BLOCKS_SM1 240

#define THREADS_SM2 128
#define BLOCKS_SM2 360

#define SCORE4_MIN sycl::int4(SCORE_MIN, SCORE_MIN, SCORE_MIN, SCORE_MIN)

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
  int **scores;
  int **affines;
  Chain *query;
  int queryFrontGap;
  Chain *target;
  int targetFrontGap;
  Scorer *scorer;
  int pLeft;
  int pRight;
  int card;
} Context;

//******************************************************************************
// PUBLIC

extern void nwLinearDataGpu(int **scores, int **affines, Chain *query,
                            int queryFrontGap, Chain *target,
                            int targetFrontGap, Scorer *scorer, int pLeft,
                            int pRight, int card, Thread *thread);

//******************************************************************************

//******************************************************************************
// PRIVATE

static int gap(int idx, int queryFrontGap_, int gapOpen_, int gapExtend_,
               int gapDiff_, int dRow_);
static int aff(int idx, int targetFrontGap_, int dRow_);

template <class Sub>
static void solveShortDelegated(int d, VBus vBus, sycl::int2 *hBus, Sub sub,
                                sycl::nd_item<1> item_ct1, int queryFrontGap_,
                                int targetFrontGap_, int gapOpen_,
                                int gapExtend_, int gapDiff_, int dRow_,
                                int rows_, int cols_, int cellWidth_,
                                int pLeft_, int pRight_, int scorerLen_,
                                int subLen_, int match_, int mismatch_,
                                int *hBusScrShr, int *hBusAffShr,
                                sycl::char4 *rowGpu, char *colGpu, int *subGpu);

template <class Sub>
static void solveShortNormal(int d, VBus vBus, sycl::int2 *hBus, Sub sub,
                             sycl::nd_item<1> item_ct1, int queryFrontGap_,
                             int targetFrontGap_, int gapOpen_, int gapExtend_,
                             int gapDiff_, int dRow_, int rows_, int cols_,
                             int cellWidth_, int pLeft_, int pRight_,
                             int scorerLen_, int subLen_, int match_,
                             int mismatch_, int *hBusScrShr, int *hBusAffShr,
                             sycl::char4 *rowGpu, char *colGpu, int *subGpu);

template <class Sub>
static void
solveShort(int d, VBus vBus, sycl::int2 *hBus, Sub sub,
           sycl::nd_item<1> item_ct1, int queryFrontGap_, int targetFrontGap_,
           int gapOpen_, int gapExtend_, int gapDiff_, int dRow_, int rows_,
           int cols_, int cellWidth_, int pLeft_, int pRight_, int scorerLen_,
           int subLen_, int match_, int mismatch_, int *hBusScrShr,
           int *hBusAffShr, sycl::char4 *rowGpu, char *colGpu, int *subGpu);

template <class Sub>
static void
solveLong(int d, VBus vBus, sycl::int2 *hBus, Sub sub,
          sycl::nd_item<1> item_ct1, int queryFrontGap_, int targetFrontGap_,
          int gapOpen_, int gapExtend_, int gapDiff_, int dRow_, int rows_,
          int cols_, int cellWidth_, int pLeft_, int pRight_, int scorerLen_,
          int subLen_, int match_, int mismatch_, int *hBusScrShr,
          int *hBusAffShr, sycl::char4 *rowGpu, char *colGpu, int *subGpu);

static void *nwLinearDataGpuKernel(void *params);

//******************************************************************************

//******************************************************************************
// PUBLIC

extern void nwLinearDataGpu(int **scores, int **affines, Chain *query,
                            int queryFrontGap, Chain *target,
                            int targetFrontGap, Scorer *scorer, int pLeft,
                            int pRight, int card, Thread *thread) {

  Context *param = (Context *)malloc(sizeof(Context));

  param->scores = scores;
  param->affines = affines;
  param->query = query;
  param->queryFrontGap = queryFrontGap;
  param->target = target;
  param->targetFrontGap = targetFrontGap;
  param->scorer = scorer;
  param->pLeft = pLeft;
  param->pRight = pRight;
  param->card = card;

  if (thread == NULL) {
    nwLinearDataGpuKernel(param);
  } else {
    threadCreate(thread, nwLinearDataGpuKernel, (void *)param);
  }
}

//******************************************************************************

//------------------------------------------------------------------------------
// FUNCTORS

class SubScalar {
public:
  int operator()(char a, char b, int scorerLen_, int match_, int mismatch_,
                 int subLen_, int *subGpu) {
    return a == b ? match_ : mismatch_;
  }
};

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

static int gap(int idx, int queryFrontGap_, int gapOpen_, int gapExtend_,
               int gapDiff_, int dRow_) {
  if (idx == dRow_ - 1)
    return 0;
  if (idx < dRow_ - 1)
    return SCORE_MIN;
  return -gapOpen_ - gapExtend_ * (idx - dRow_) + queryFrontGap_ * gapDiff_;
}

static int aff(int idx, int targetFrontGap_, int dRow_) {
  if (dRow_ > 0 && idx == dRow_ - 1 && targetFrontGap_)
    return 0;
  return SCORE_MIN;
}

template <class Sub>
static void solveShortDelegated(
    int d, VBus vBus, sycl::int2 *hBus, Sub sub, sycl::nd_item<1> item_ct1,
    int queryFrontGap_, int targetFrontGap_, int gapOpen_, int gapExtend_,
    int gapDiff_, int dRow_, int rows_, int cols_, int cellWidth_, int pLeft_,
    int pRight_, int scorerLen_, int subLen_, int match_, int mismatch_,
    int *hBusScrShr, int *hBusAffShr, sycl::char4 *rowGpu, char *colGpu,
    int *subGpu) {
  const int groupRangeId = item_ct1.get_group_range(0);
  const int groupId = item_ct1.get_group(0);
  const int localId = item_ct1.get_local_id(0);
  const int localRangeId = item_ct1.get_local_range(0);

  int row = (d + groupId - groupRangeId + 1) * (localRangeId * 4) + localId * 4;
  int col = cellWidth_ * (groupRangeId - groupId - 1) - localId;
  bool doWork = true;

  if (row < 0)
    doWork = false;

  sycl::char4 rowCodes;
  Atom atom;

  if (doWork) {
    row -= (col < 0) * (groupRangeId * localRangeId * 4);
    col += (col < 0) * cols_;

    int x1 = cellWidth_ * (groupRangeId - groupId - 1) + localRangeId;
    int y1 = (d + groupId - groupRangeId + 1) * (localRangeId * 4);

    int x2 = cellWidth_ * (groupRangeId - groupId - 1) - localRangeId;
    int y2 = (d + groupId - groupRangeId + 2) * (localRangeId * 4);

    y2 -= (x2 < 0) * (groupRangeId * localRangeId * 4);
    x2 += (x2 < 0) * cols_;

    if (y1 - x1 > pLeft_ && (x2 - y2 > pRight_ || y2 < 0)) {

      row += (col != 0) * groupRangeId * localRangeId * 4;
      vBus.mch[(row >> 2) % (groupRangeId * localRangeId)] = SCORE_MIN;
      vBus.scr[(row >> 2) % (groupRangeId * localRangeId)] = SCORE4_MIN;
      vBus.aff[(row >> 2) % (groupRangeId * localRangeId)] = SCORE4_MIN;

      hBus[col] = sycl::int2(SCORE_MIN, SCORE_MIN);

      doWork = false;
    }

    if (doWork) {
      if (0 <= row && row < rows_ && col > 0) {
        atom.mch = vBus.mch[(row >> 2) % (groupRangeId * localRangeId)];
        atom.lScr = vBus.scr[(row >> 2) % (groupRangeId * localRangeId)];
        atom.lAff = vBus.aff[(row >> 2) % (groupRangeId * localRangeId)];
      } else {
        atom.mch =
            gap(row - 1, queryFrontGap_, gapOpen_, gapExtend_, gapDiff_, dRow_);
        atom.lScr = sycl::int4(
            gap(row, queryFrontGap_, gapOpen_, gapExtend_, gapDiff_, dRow_),
            gap(row + 1, queryFrontGap_, gapOpen_, gapExtend_, gapDiff_, dRow_),
            gap(row + 2, queryFrontGap_, gapOpen_, gapExtend_, gapDiff_, dRow_),
            gap(row + 3, queryFrontGap_, gapOpen_, gapExtend_, gapDiff_,
                dRow_));
        atom.lAff = sycl::int4(aff(row, targetFrontGap_, dRow_),
                               aff(row + 1, targetFrontGap_, dRow_),
                               aff(row + 2, targetFrontGap_, dRow_),
                               aff(row + 3, targetFrontGap_, dRow_));
      }

      hBusScrShr[localId] = hBus[col].x();
      hBusAffShr[localId] = hBus[col].y();

      rowCodes = rowGpu[row >> 2];
    }
  }

  int del;

  for (int i = 0; i < localRangeId; ++i) {

    if (doWork) {

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
        ins =
            sycl::max((atom.lScr.y() - gapOpen_), (atom.lAff.y() - gapExtend_));

        mch = atom.lScr.x() + sub(columnCode, rowCodes.y(), scorerLen_, match_,
                                  mismatch_, subLen_, subGpu);

        atom.rScr.y() = MAX3(mch, del, ins);
        atom.rAff.y() = ins;

        del = sycl::max((atom.rScr.y() - gapOpen_), (del - gapExtend_));
        ins =
            sycl::max((atom.lScr.z() - gapOpen_), (atom.lAff.z() - gapExtend_));

        mch = atom.lScr.y() + sub(columnCode, rowCodes.z(), scorerLen_, match_,
                                  mismatch_, subLen_, subGpu);

        atom.rScr.z() = MAX3(mch, del, ins);
        atom.rAff.z() = ins;

        del = sycl::max((atom.rScr.z() - gapOpen_), (del - gapExtend_));
        ins =
            sycl::max((atom.lScr.w() - gapOpen_), (atom.lAff.w() - gapExtend_));

        mch = atom.lScr.z() + sub(columnCode, rowCodes.w(), scorerLen_, match_,
                                  mismatch_, subLen_, subGpu);

        atom.rScr.w() = MAX3(mch, del, ins);
        atom.rAff.w() = ins;

        atom.mch = atom.up.x();
        atom.lScr = atom.rScr;
        atom.lAff = atom.rAff;
      }
    }

    int a = atom.rScr.w();
    int b = del;

    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (doWork) {
      if (0 <= row && row < rows_) {

        if (localId == localRangeId - 1 || i == localRangeId - 1 ||
            row == rows_ - 4) {
          hBus[col] = sycl::int2(a, b);
        } else {
          hBusScrShr[localId + 1] = a;
          hBusAffShr[localId + 1] = b;
        }
      }

      ++col;

      if (col == cols_) {

        col = 0;
        row = row + groupRangeId * localRangeId * 4;

        atom.mch =
            gap(row - 1, queryFrontGap_, gapOpen_, gapExtend_, gapDiff_, dRow_);
        atom.lScr = sycl::int4(
            gap(row, queryFrontGap_, gapOpen_, gapExtend_, gapDiff_, dRow_),
            gap(row + 1, queryFrontGap_, gapOpen_, gapExtend_, gapDiff_, dRow_),
            gap(row + 2, queryFrontGap_, gapOpen_, gapExtend_, gapDiff_, dRow_),
            gap(row + 3, queryFrontGap_, gapOpen_, gapExtend_, gapDiff_,
                dRow_));
        atom.lAff = sycl::int4(aff(row, targetFrontGap_, dRow_),
                               aff(row + 1, targetFrontGap_, dRow_),
                               aff(row + 2, targetFrontGap_, dRow_),
                               aff(row + 3, targetFrontGap_, dRow_));

        rowCodes = rowGpu[row >> 2];
      }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);
  }

  if (doWork) {

    if (row < 0 || row >= rows_)
      doWork = false;

    if (doWork) {
      vBus.mch[(row >> 2) % (groupRangeId * localRangeId)] = atom.up.x();
      vBus.scr[(row >> 2) % (groupRangeId * localRangeId)] = atom.lScr;
      vBus.aff[(row >> 2) % (groupRangeId * localRangeId)] = atom.lAff;
    }
  }
}

template <class Sub>
static void solveShortNormal(int d, VBus vBus, sycl::int2 *hBus, Sub sub,
                             sycl::nd_item<1> item_ct1, int queryFrontGap_,
                             int targetFrontGap_, int gapOpen_, int gapExtend_,
                             int gapDiff_, int dRow_, int rows_, int cols_,
                             int cellWidth_, int pLeft_, int pRight_,
                             int scorerLen_, int subLen_, int match_,
                             int mismatch_, int *hBusScrShr, int *hBusAffShr,
                             sycl::char4 *rowGpu, char *colGpu, int *subGpu) {

  const int groupRangeId = item_ct1.get_group_range(0);
  const int groupId = item_ct1.get_group(0);
  const int localId = item_ct1.get_local_id(0);
  const int localRangeId = item_ct1.get_local_range(0);

  int row = (d + groupId - groupRangeId + 1) * (localRangeId * 4) + localId * 4;
  int col = cellWidth_ * (groupRangeId - groupId - 1) - localId;

  bool doWork = true;
  if (row < 0 || row >= rows_)
    doWork = false;

  if (doWork) {
    int x1 = cellWidth_ * (groupRangeId - groupId - 1) + localRangeId;
    int y1 = (d + groupId - groupRangeId + 1) * (localRangeId * 4);

    if (y1 - x1 > pLeft_) {

      const int id =
          sycl::min(row >> 2, (int)(rows_ / 4)) % (groupRangeId * localRangeId);
      vBus.mch[id] = SCORE_MIN;
      vBus.scr[id] = SCORE4_MIN;
      vBus.aff[id] = SCORE4_MIN;
      doWork = false;
    }

    if (doWork) {
      int x2 = cellWidth_ * (groupRangeId - groupId - 1) - localRangeId;
      int y2 = (d + groupId - groupRangeId + 2) * (localRangeId * 4);

      if (x2 - y2 > pRight_) {

        // clear right
        vBus.mch[(row >> 2) % (groupRangeId * localRangeId)] = SCORE_MIN;
        vBus.scr[(row >> 2) % (groupRangeId * localRangeId)] = SCORE4_MIN;
        vBus.aff[(row >> 2) % (groupRangeId * localRangeId)] = SCORE4_MIN;

        // clear down
        hBus[col] = sycl::int2(SCORE_MIN, SCORE_MIN);
        hBus[col + localRangeId] = sycl::int2(SCORE_MIN, SCORE_MIN);

        doWork = false;
      }
    }
  }

  Atom atom;
  sycl::char4 rowCodes;
  int del;
  int a;

  if (doWork) {
    atom.mch = vBus.mch[(row >> 2) % (groupRangeId * localRangeId)];
    atom.lScr = vBus.scr[(row >> 2) % (groupRangeId * localRangeId)];
    atom.lAff = vBus.aff[(row >> 2) % (groupRangeId * localRangeId)];

    hBusScrShr[localId] = hBus[col].x();
    hBusAffShr[localId] = hBus[col].y();

    rowCodes = rowGpu[row >> 2];
  }

  for (int i = 0; i < localRangeId; ++i, ++col) {

    if (doWork) {
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

    a = atom.rScr.w();
    int b = del;

    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (doWork) {
      if (localId == localRangeId - 1 || row == rows_ - 4) {
        hBus[col] = sycl::int2(a, b);
      } else {
        hBusScrShr[localId + 1] = a;
        hBusAffShr[localId + 1] = b;
      }
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
  }

  int c = a;
  int dd = del;

  if (doWork) {
    hBus[col - 1] = sycl::int2(c, dd);

    vBus.mch[(row >> 2) % (groupRangeId * localRangeId)] = atom.up.x();
    vBus.scr[(row >> 2) % (groupRangeId * localRangeId)] = atom.lScr;
    vBus.aff[(row >> 2) % (groupRangeId * localRangeId)] = atom.lAff;
  }
}

template <class Sub>
static void
solveShort(int d, VBus vBus, sycl::int2 *hBus, Sub sub,
           sycl::nd_item<1> item_ct1, int queryFrontGap_, int targetFrontGap_,
           int gapOpen_, int gapExtend_, int gapDiff_, int dRow_, int rows_,
           int cols_, int cellWidth_, int pLeft_, int pRight_, int scorerLen_,
           int subLen_, int match_, int mismatch_, int *hBusScrShr,
           int *hBusAffShr, sycl::char4 *rowGpu, char *colGpu, int *subGpu) {
  const int groupRangeId = item_ct1.get_group_range(0);
  const int groupId = item_ct1.get_group(0);

  if (groupId == (groupRangeId - 1)) {
    solveShortDelegated(d, vBus, hBus, sub, item_ct1, queryFrontGap_,
                        targetFrontGap_, gapOpen_, gapExtend_, gapDiff_, dRow_,
                        rows_, cols_, cellWidth_, pLeft_, pRight_, scorerLen_,
                        subLen_, match_, mismatch_, hBusScrShr, hBusAffShr,
                        rowGpu, colGpu, subGpu);
  } else {
    solveShortNormal(d, vBus, hBus, sub, item_ct1, queryFrontGap_,
                     targetFrontGap_, gapOpen_, gapExtend_, gapDiff_, dRow_,
                     rows_, cols_, cellWidth_, pLeft_, pRight_, scorerLen_,
                     subLen_, match_, mismatch_, hBusScrShr, hBusAffShr, rowGpu,
                     colGpu, subGpu);
  }
}

template <class Sub>
static void
solveLong(int d, VBus vBus, sycl::int2 *hBus, Sub sub,
          sycl::nd_item<1> item_ct1, int queryFrontGap_, int targetFrontGap_,
          int gapOpen_, int gapExtend_, int gapDiff_, int dRow_, int rows_,
          int cols_, int cellWidth_, int pLeft_, int pRight_, int scorerLen_,
          int subLen_, int match_, int mismatch_, int *hBusScrShr,
          int *hBusAffShr, sycl::char4 *rowGpu, char *colGpu, int *subGpu) {
  const int groupRangeId = item_ct1.get_group_range(0);
  const int groupId = item_ct1.get_group(0);
  const int localId = item_ct1.get_local_id(0);
  const int localRangeId = item_ct1.get_local_range(0);

  int row = (d + groupId - groupRangeId + 1) * (localRangeId * 4) + localId * 4;
  int col = cellWidth_ * (groupRangeId - groupId - 1) - localId + localRangeId;

  bool doWork = true;

  if (row < 0 || row >= rows_)
    doWork = false;

  if (doWork) {
    int x1 = cellWidth_ * (groupRangeId - groupId - 1) + cellWidth_;
    int y1 = (d + groupId - groupRangeId + 1) * (localRangeId * 4);

    if (y1 - x1 > pLeft_) {
      const int id =
          sycl::min(row >> 2, (int)(rows_ / 4)) % (groupRangeId * localRangeId);
      vBus.mch[id] = SCORE_MIN;
      vBus.scr[id] = SCORE4_MIN;
      vBus.aff[id] = SCORE4_MIN;
      doWork = false;
    }

    if (doWork) {
      int x2 = cellWidth_ * (groupRangeId - groupId - 1);
      int y2 = (d + groupId - groupRangeId + 2) * (localRangeId * 4);

      if (x2 - y2 > pRight_) {

        vBus.mch[(row >> 2) % (groupRangeId * localRangeId)] = SCORE_MIN;
        vBus.scr[(row >> 2) % (groupRangeId * localRangeId)] = SCORE4_MIN;
        vBus.aff[(row >> 2) % (groupRangeId * localRangeId)] = SCORE4_MIN;

        for (int i = 0; i < cellWidth_ - localRangeId; i += localRangeId) {
          hBus[col + i] = sycl::int2(SCORE_MIN, SCORE_MIN);
        }

        doWork = false;
      }
    }
  }

  Atom atom;
  sycl::char4 rowCodes;
  int del;
  int a;

  if (doWork) {
    atom.mch = vBus.mch[(row >> 2) % (groupRangeId * localRangeId)];
    atom.lScr = vBus.scr[(row >> 2) % (groupRangeId * localRangeId)];
    atom.lAff = vBus.aff[(row >> 2) % (groupRangeId * localRangeId)];

    hBusScrShr[localId] = hBus[col].x();
    hBusAffShr[localId] = hBus[col].y();

    rowCodes = rowGpu[row >> 2];
  }

  for (int i = 0; i < cellWidth_ - localRangeId; ++i, ++col) {
    if (doWork) {
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

    a = atom.rScr.w();
    int b = del;

    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (doWork) {
      if (localId == localRangeId - 1 || row == rows_ - 4) {
        hBus[col] = sycl::int2(a, b);
      } else {
        hBusScrShr[localId + 1] = a;
        hBusAffShr[localId + 1] = b;
      }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);
  }

  int c = a;
  int dd = del;

  if (doWork) {
    hBus[col - 1] = sycl::int2(c, dd);

    vBus.mch[(row >> 2) % (groupRangeId * localRangeId)] = atom.up.x();
    vBus.scr[(row >> 2) % (groupRangeId * localRangeId)] = atom.lScr;
    vBus.aff[(row >> 2) % (groupRangeId * localRangeId)] = atom.lAff;
  }
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// CPU KERNELS

static void *nwLinearDataGpuKernel(void *params) try {

  Context *context = (Context *)params;

  int **scores = context->scores;
  int **affines = context->affines;
  Chain *query = context->query;
  int queryFrontGap = context->queryFrontGap;
  Chain *target = context->target;
  int targetFrontGap = context->targetFrontGap;
  Scorer *scorer = context->scorer;
  int pLeft = context->pLeft;
  int pRight = context->pRight;
  int card = context->card;

  int currentCard;

  sycl::queue dev_q = queues[card];

  // TIMER_START("Linear data %d %d", chainGetLength(query),
  // chainGetLength(target));

  int rows = chainGetLength(query);
  int cols = chainGetLength(target);
  int gapOpen = scorerGetGapOpen(scorer);
  int gapExtend = scorerGetGapExtend(scorer);
  int gapDiff = gapOpen - gapExtend;
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

  if (pLeft < 0) {
    pLeft = std::max(rows, cols) + dRow;
  } else {
    pLeft += dRow;
  }

  if (pRight < 0) {
    pRight = std::max(rows, cols) + dRow;
  } else {
    pRight += dRow;
  }

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

  size_t rowSize = rowsGpu * sizeof(char);
  sycl::char4 *rowGpu = sycl::malloc_device<sycl::char4>(rowsGpu, dev_q);

  dev_q.memcpy(rowGpu, rowCpu, rowSize).wait();

  memoryUsedGpu += rowSize;

  size_t colSize = colsGpu * sizeof(char);
  char *colGpu = sycl::malloc_device<char>(colsGpu, dev_q);

  dev_q.memcpy(colGpu, colCpu, colSize).wait();

  memoryUsedGpu += colSize;

  size_t hBusSize = colsGpu * sizeof(sycl::int2);

  sycl::int2 *hBus = sycl::malloc_shared<sycl::int2>(colsGpu, dev_q);
  for (int i = 0; i < colsGpu; ++i) {
    int gap = -gapOpen - gapExtend * i + targetFrontGap * gapDiff;
    hBus[i] = dRow == 0 ? sycl::int2(gap, SCORE_MIN)
                        : sycl::int2(SCORE_MIN, SCORE_MIN);
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

  for (int diagonal = 0; diagonal < diagonals; ++diagonal) {
    if (scalar) {
      if (sub[0] >= sub[1]) {
        dev_q
            .submit([&](sycl::handler &cgh) {
              sycl::accessor<int, 1, sycl::access_mode::read_write,
                             sycl::access::target::local>
                  hBusScrShr_acc_ct1(sycl::range<1>(threads), cgh);
              sycl::accessor<int, 1, sycl::access_mode::read_write,
                             sycl::access::target::local>
                  hBusAffShr_acc_ct1(sycl::range<1>(threads), cgh);

              cgh.parallel_for(
                  sycl::nd_range<1>(blocks * threads, threads),
                  [=](sycl::nd_item<1> item_ct1) {
                    solveShort(
                        diagonal, vBusGpu, hBus, SubScalar(), item_ct1,
                        queryFrontGap, targetFrontGap, gapOpen, gapExtend,
                        gapDiff, dRow, rowsGpu, colsGpu, cellWidth, pLeft,
                        pRight, scorerLen, subLen, match, mismatch,
                        hBusScrShr_acc_ct1.get_pointer(),
                        hBusAffShr_acc_ct1.get_pointer(), rowGpu, colGpu, sub);
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

              cgh.parallel_for(
                  sycl::nd_range<1>(blocks * threads, threads),
                  [=](sycl::nd_item<1> item_ct1) {
                    solveLong(diagonal, vBusGpu, hBus, SubScalar(), item_ct1,
                              queryFrontGap, targetFrontGap, gapOpen, gapExtend,
                              gapDiff, dRow, rowsGpu, colsGpu, cellWidth, pLeft,
                              pRight, scorerLen, subLen, match, mismatch,
                              hBusScrShr_acc_ct1.get_pointer(),
                              hBusAffShr_acc_ct1.get_pointer(), rowGpu, colGpu,
                              sub);
                  });
            })
            .wait();
      } else {
        // cannot use mismatch negative trick
        dev_q
            .submit([&](sycl::handler &cgh) {
              sycl::accessor<int, 1, sycl::access_mode::read_write,
                             sycl::access::target::local>
                  hBusScrShr_acc_ct1(sycl::range<1>(threads), cgh);
              sycl::accessor<int, 1, sycl::access_mode::read_write,
                             sycl::access::target::local>
                  hBusAffShr_acc_ct1(sycl::range<1>(threads), cgh);

              cgh.parallel_for(
                  sycl::nd_range<1>(blocks * threads, threads),
                  [=](sycl::nd_item<1> item_ct1) {
                    solveShort(
                        diagonal, vBusGpu, hBus, SubScalarRev(), item_ct1,
                        queryFrontGap, targetFrontGap, gapOpen, gapExtend,
                        gapDiff, dRow, rowsGpu, colsGpu, cellWidth, pLeft,
                        pRight, scorerLen, subLen, match, mismatch,
                        hBusScrShr_acc_ct1.get_pointer(),
                        hBusAffShr_acc_ct1.get_pointer(), rowGpu, colGpu, sub);
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

              cgh.parallel_for(
                  sycl::nd_range<1>(blocks * threads, threads),
                  [=](sycl::nd_item<1> item_ct1) {
                    solveLong(diagonal, vBusGpu, hBus, SubScalarRev(), item_ct1,
                              queryFrontGap, targetFrontGap, gapOpen, gapExtend,
                              gapDiff, dRow, rowsGpu, colsGpu, cellWidth, pLeft,
                              pRight, scorerLen, subLen, match, mismatch,
                              hBusScrShr_acc_ct1.get_pointer(),
                              hBusAffShr_acc_ct1.get_pointer(), rowGpu, colGpu,
                              sub);
                  });
            })
            .wait();
      }
    } else {
      dev_q
          .submit([&](sycl::handler &cgh) {
            sycl::accessor<int, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                hBusScrShr_acc_ct1(sycl::range<1>(threads), cgh);
            sycl::accessor<int, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                hBusAffShr_acc_ct1(sycl::range<1>(threads), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(blocks * threads, threads),
                [=](sycl::nd_item<1> item_ct1) {
                  solveShort(diagonal, vBusGpu, hBus, SubVector(), item_ct1,
                             queryFrontGap, targetFrontGap, gapOpen, gapExtend,
                             gapDiff, dRow, rowsGpu, colsGpu, cellWidth, pLeft,
                             pRight, scorerLen, subLen, match, mismatch,
                             hBusScrShr_acc_ct1.get_pointer(),
                             hBusAffShr_acc_ct1.get_pointer(), rowGpu, colGpu,
                             sub);
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

            cgh.parallel_for(
                sycl::nd_range<1>(blocks * threads, threads),
                [=](sycl::nd_item<1> item_ct1) {
                  solveLong(diagonal, vBusGpu, hBus, SubVector(), item_ct1,
                            queryFrontGap, targetFrontGap, gapOpen, gapExtend,
                            gapDiff, dRow, rowsGpu, colsGpu, cellWidth, pLeft,
                            pRight, scorerLen, subLen, match, mismatch,
                            hBusScrShr_acc_ct1.get_pointer(),
                            hBusAffShr_acc_ct1.get_pointer(), rowGpu, colGpu,
                            sub);
                });
          })
          .wait();
    }
  }

  // TIMER_STOP;
  //**************************************************************************

  //**************************************************************************
  // SAVE RESULTS

  if (scores != NULL) {
    *scores = (int *)malloc(cols * sizeof(int));

    for (int i = 0; i < cols; ++i) {
      if (i < rows - 1 - pLeft) {
        (*scores)[i] = SCORE_MIN;
      } else {
        (*scores)[i] = hBus[i].x();
      }
    }
  }

  if (affines != NULL) {
    *affines = (int *)malloc(cols * sizeof(int));

    for (int i = 0; i < cols; ++i) {
      if (i < rows - 1 - pLeft) {
        (*affines)[i] = SCORE_MIN;
      } else {
        (*affines)[i] = hBus[i].y();
      }
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

  // TIMER_STOP;

  return NULL;
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

//------------------------------------------------------------------------------
//******************************************************************************
