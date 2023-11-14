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
#define BLOCKS_SM2 480

#define INT4_ZERO sycl::int4(0, 0, 0, 0)

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

extern void swEndDataGpu(int *queryEnd, int *targetEnd, int *outScore,
                         int **scores, int **affines, Chain *query,
                         Chain *target, Scorer *scorer, int score, int card,
                         Thread *thread);

//******************************************************************************

//******************************************************************************
// PRIVATE

template <class Sub>
static void
solveShortDelegated(int d, VBus vBus, sycl::int2 *hBus, sycl::int3 *results,
                    Sub sub, sycl::nd_item<1> item_ct1, int gapOpen_,
                    int gapExtend_, int rows_, int cols_, int cellWidth_,
                    int pruneLow_, int pruneHigh_, int scorerLen_, int subLen_,
                    int match_, int mismatch_, int *hBusScrShr, int *hBusAffShr,
                    sycl::char4 *rowGpu, char *colGpu, int *subGpu);

template <class Sub>
static void
solveShortNormal(int d, VBus vBus, sycl::int2 *hBus, sycl::int3 *results,
                 Sub sub, sycl::nd_item<1> item_ct1, int gapOpen_,
                 int gapExtend_, int rows_, int cols_, int cellWidth_,
                 int pruneLow_, int pruneHigh_, int scorerLen_, int subLen_,
                 int match_, int mismatch_, int *hBusScrShr, int *hBusAffShr,
                 sycl::char4 *rowGpu, char *colGpu, int *subGpu);

template <class Sub>
static void solveShort(int d, VBus vBus, sycl::int2 *hBus, sycl::int3 *results,
                       Sub sub, sycl::nd_item<1> item_ct1, int gapOpen_,
                       int gapExtend_, int rows_, int cols_, int cellWidth_,
                       int pruneLow_, int pruneHigh_, int scorerLen_,
                       int subLen_, int match_, int mismatch_, int *hBusScrShr,
                       int *hBusAffShr, sycl::char4 *rowGpu, char *colGpu,
                       int *subGpu);

template <class Sub>
static void solveLong(int d, VBus vBus, sycl::int2 *hBus, int *bBus,
                      sycl::int3 *results, Sub sub, sycl::nd_item<1> item_ct1,
                      int gapOpen_, int gapExtend_, int rows_, int cols_,
                      int cellWidth_, int pruneLow_, int pruneHigh_,
                      int scorerLen_, int subLen_, int match_, int mismatch_,
                      int *hBusScrShr, int *hBusAffShr, sycl::char4 *rowGpu,
                      char *colGpu, int *subGpu);

static void *swEndDataGpuKernel(void *params);

//******************************************************************************

//******************************************************************************
// PUBLIC

extern void swEndDataGpu(int *queryEnd, int *targetEnd, int *outScore,
                         int **scores, int **affines, Chain *query,
                         Chain *target, Scorer *scorer, int score, int card,
                         Thread *thread) {

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

  if (thread == NULL) {
    swEndDataGpuKernel(param);
  } else {
    threadCreate(thread, swEndDataGpuKernel, (void *)param);
  }
}

//******************************************************************************

//******************************************************************************
// PRIVATE

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

template <class Sub>
static void
solveShortDelegated(int d, VBus vBus, sycl::int2 *hBus, sycl::int3 *results,
                    Sub sub, sycl::nd_item<1> item_ct1, int gapOpen_,
                    int gapExtend_, int rows_, int cols_, int cellWidth_,
                    int pruneLow_, int pruneHigh_, int scorerLen_, int subLen_,
                    int match_, int mismatch_, int *hBusScrShr, int *hBusAffShr,
                    sycl::char4 *rowGpu, char *colGpu, int *subGpu) {

  const int groupRangeId = item_ct1.get_group_range(0);
  const int groupId = item_ct1.get_group(0);
  const int localId = item_ct1.get_local_id(0);
  const int localRangeId = item_ct1.get_local_range(0);

  bool doWork = true;

  if (pruneLow_ >= 0 && pruneHigh_ < groupRangeId) {
    doWork = false;
  }

  int row, col;
  Atom atom;
  sycl::char4 rowCodes;
  sycl::int3 res;

  if (doWork) {
    row = (d + groupId - groupRangeId + 1) * (localRangeId * 4) + localId * 4;
    col = cellWidth_ * (groupRangeId - groupId - 1) - localId;

    if (row < 0)
      doWork = false;

    if (doWork) {
      row -= (col < 0) * (groupRangeId * localRangeId * 4);
      col += (col < 0) * cols_;

      if (0 <= row && row < rows_ && col > 0) {
        atom.mch = vBus.mch[(row >> 2) % (groupRangeId * localRangeId)];
        atom.lScr = vBus.scr[(row >> 2) % (groupRangeId * localRangeId)];

        atom.lAff = vBus.aff[(row >> 2) % (groupRangeId * localRangeId)];
      } else {
        atom.mch = 0;
        atom.lScr = INT4_ZERO;
        atom.lAff = INT4_ZERO;
      }

      hBusScrShr[localId] = hBus[col].x();
      hBusAffShr[localId] = hBus[col].y();

      rowCodes = rowGpu[row >> 2];

      res = {0, 0, 0};
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

        atom.rScr.x() = MAX4(0, mch, del, ins);
        atom.rAff.x() = ins;

        del = sycl::max((atom.rScr.x() - gapOpen_), (del - gapExtend_));
        ins =
            sycl::max((atom.lScr.y() - gapOpen_), (atom.lAff.y() - gapExtend_));

        mch = atom.lScr.x() + sub(columnCode, rowCodes.y(), scorerLen_, match_,
                                  mismatch_, subLen_, subGpu);

        atom.rScr.y() = MAX4(0, mch, del, ins);
        atom.rAff.y() = ins;

        del = sycl::max((atom.rScr.y() - gapOpen_), (del - gapExtend_));
        ins =
            sycl::max((atom.lScr.z() - gapOpen_), (atom.lAff.z() - gapExtend_));

        mch = atom.lScr.y() + sub(columnCode, rowCodes.z(), scorerLen_, match_,
                                  mismatch_, subLen_, subGpu);

        atom.rScr.z() = MAX4(0, mch, del, ins);
        atom.rAff.z() = ins;

        del = sycl::max((atom.rScr.z() - gapOpen_), (del - gapExtend_));
        ins =
            sycl::max((atom.lScr.w() - gapOpen_), (atom.lAff.w() - gapExtend_));

        mch = atom.lScr.z() + sub(columnCode, rowCodes.w(), scorerLen_, match_,
                                  mismatch_, subLen_, subGpu);

        atom.rScr.w() = MAX4(0, mch, del, ins);
        atom.rAff.w() = ins;

        if (atom.rScr.x() > res.x()) {
          res.x() = atom.rScr.x();
          res.y() = row;
          res.z() = col;
        }
        if (atom.rScr.y() > res.x()) {
          res.x() = atom.rScr.y();
          res.y() = row + 1;
          res.z() = col;
        }
        if (atom.rScr.z() > res.x()) {
          res.x() = atom.rScr.z();
          res.y() = row + 2;
          res.z() = col;
        }
        if (atom.rScr.w() > res.x()) {
          res.x() = atom.rScr.w();
          res.y() = row + 3;
          res.z() = col;
        }

        atom.mch = atom.up.x();
        atom.lScr = atom.rScr;
        atom.lAff = atom.rAff;
      }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (doWork) {
      if (0 <= row && row < rows_) {

        if (localId == localRangeId - 1 || i == localRangeId - 1) {
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

        atom.mch = 0;
        atom.lScr = INT4_ZERO;
        atom.lAff = atom.lScr;

        rowCodes = rowGpu[row >> 2];
      }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);
  }

  if (!doWork)
    return;

  if (res.x() > results[groupId * localRangeId + localId].x()) {
    results[groupId * localRangeId + localId] = res;
  }

  if (row < 0 || row >= rows_)
    return;

  vBus.mch[(row >> 2) % (groupRangeId * localRangeId)] = atom.up.x();
  vBus.scr[(row >> 2) % (groupRangeId * localRangeId)] = atom.lScr;
  vBus.aff[(row >> 2) % (groupRangeId * localRangeId)] = atom.lAff;
}

template <class Sub>
static void
solveShortNormal(int d, VBus vBus, sycl::int2 *hBus, sycl::int3 *results,
                 Sub sub, sycl::nd_item<1> item_ct1, int gapOpen_,
                 int gapExtend_, int rows_, int cols_, int cellWidth_,
                 int pruneLow_, int pruneHigh_, int scorerLen_, int subLen_,
                 int match_, int mismatch_, int *hBusScrShr, int *hBusAffShr,
                 sycl::char4 *rowGpu, char *colGpu, int *subGpu) {

  const int groupRangeId = item_ct1.get_group_range(0);
  const int groupId = item_ct1.get_group(0);
  const int localId = item_ct1.get_local_id(0);
  const int localRangeId = item_ct1.get_local_range(0);

  bool doWork = true;

  if ((int)groupId <= pruneLow_ || groupId >= pruneHigh_) {
    doWork = false;
  }

  int row, col;
  Atom atom;
  sycl::char4 rowCodes;
  sycl::int3 res;

  if (doWork) {
    row = (d + groupId - groupRangeId + 1) * (localRangeId * 4) + localId * 4;
    col = cellWidth_ * (groupRangeId - groupId - 1) - localId;

    if (row < 0 || row >= rows_)
      doWork = false;

    if (doWork) {
      atom.mch = vBus.mch[(row >> 2) % (groupRangeId * localRangeId)];

      atom.lScr = vBus.scr[(row >> 2) % (groupRangeId * localRangeId)];
      atom.lAff = vBus.aff[(row >> 2) % (groupRangeId * localRangeId)];

      hBusScrShr[localId] = hBus[col].x();
      hBusAffShr[localId] = hBus[col].y();

      rowCodes = rowGpu[row >> 2];
      res = {0, 0, 0};
    }
  }

  int del;

  for (int i = 0; i < localRangeId; ++i, ++col) {

    if (doWork) {
      char columnCode = colGpu[col];

      if (localId == 0) {
        atom.up = hBus[col];
      } else {
        atom.up = sycl::int2(hBusScrShr[localId], hBusAffShr[localId]);
      }

      del = sycl::max((atom.up.x() - gapOpen_), (atom.up.y() - gapExtend_));
      int ins =
          sycl::max((atom.lScr.x() - gapOpen_), (atom.lAff.x() - gapExtend_));

      int mch = atom.mch + sub(columnCode, rowCodes.x(), scorerLen_, match_,
                               mismatch_, subLen_, subGpu);

      atom.rScr.x() = MAX4(0, mch, del, ins);
      atom.rAff.x() = ins;

      del = sycl::max((atom.rScr.x() - gapOpen_), (del - gapExtend_));
      ins = sycl::max((atom.lScr.y() - gapOpen_), (atom.lAff.y() - gapExtend_));

      mch = atom.lScr.x() + sub(columnCode, rowCodes.y(), scorerLen_, match_,
                                mismatch_, subLen_, subGpu);

      atom.rScr.y() = MAX4(0, mch, del, ins);
      atom.rAff.y() = ins;

      del = sycl::max((atom.rScr.y() - gapOpen_), (del - gapExtend_));
      ins = sycl::max((atom.lScr.z() - gapOpen_), (atom.lAff.z() - gapExtend_));

      mch = atom.lScr.y() + sub(columnCode, rowCodes.z(), scorerLen_, match_,
                                mismatch_, subLen_, subGpu);

      atom.rScr.z() = MAX4(0, mch, del, ins);
      atom.rAff.z() = ins;

      del = sycl::max((atom.rScr.z() - gapOpen_), (del - gapExtend_));
      ins = sycl::max((atom.lScr.w() - gapOpen_), (atom.lAff.w() - gapExtend_));

      mch = atom.lScr.z() + sub(columnCode, rowCodes.w(), scorerLen_, match_,
                                mismatch_, subLen_, subGpu);

      atom.rScr.w() = MAX4(0, mch, del, ins);
      atom.rAff.w() = ins;

      if (atom.rScr.x() > res.x()) {
        res.x() = atom.rScr.x();
        res.y() = row;
        res.z() = col;
      }
      if (atom.rScr.y() > res.x()) {
        res.x() = atom.rScr.y();
        res.y() = row + 1;
        res.z() = col;
      }
      if (atom.rScr.z() > res.x()) {
        res.x() = atom.rScr.z();
        res.y() = row + 2;
        res.z() = col;
      }
      if (atom.rScr.w() > res.x()) {
        res.x() = atom.rScr.w();
        res.y() = row + 3;
        res.z() = col;
      }

      atom.mch = atom.up.x();
      atom.lScr = atom.rScr;
      atom.lAff = atom.rAff;
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (doWork) {
      if (localId == localRangeId - 1) {
        hBus[col] = sycl::int2(atom.rScr.w(), del);
      } else {
        hBusScrShr[localId + 1] = atom.rScr.w();
        hBusAffShr[localId + 1] = del;
      }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);
  }

  if (!doWork)
    return;

  const int vBusIdx = (row >> 2) % (groupRangeId * localRangeId);
  vBus.mch[vBusIdx] = atom.up.x();
  vBus.scr[vBusIdx] = atom.lScr;
  vBus.aff[vBusIdx] = atom.lAff;

  hBus[col - 1] = sycl::int2(atom.rScr.w(), del);

  if (res.x() > results[groupId * localRangeId + localId].x()) {
    results[groupId * localRangeId + localId] = res;
  }
}

template <class Sub>
static void solveShort(int d, VBus vBus, sycl::int2 *hBus, sycl::int3 *results,
                       Sub sub, sycl::nd_item<1> item_ct1, int gapOpen_,
                       int gapExtend_, int rows_, int cols_, int cellWidth_,
                       int pruneLow_, int pruneHigh_, int scorerLen_,
                       int subLen_, int match_, int mismatch_, int *hBusScrShr,
                       int *hBusAffShr, sycl::char4 *rowGpu, char *colGpu,
                       int *subGpu) {

  const int groupRangeId = item_ct1.get_group_range(0);
  const int groupId = item_ct1.get_group(0);

  if (groupId == (groupRangeId - 1)) {
    solveShortDelegated(d, vBus, hBus, results, sub, item_ct1, gapOpen_,
                        gapExtend_, rows_, cols_, cellWidth_, pruneLow_,
                        pruneHigh_, scorerLen_, subLen_, match_, mismatch_,
                        hBusScrShr, hBusAffShr, rowGpu, colGpu, subGpu);
  } else {
    solveShortNormal(d, vBus, hBus, results, sub, item_ct1, gapOpen_,
                     gapExtend_, rows_, cols_, cellWidth_, pruneLow_,
                     pruneHigh_, scorerLen_, subLen_, match_, mismatch_,
                     hBusScrShr, hBusAffShr, rowGpu, colGpu, subGpu);
  }
}

template <class Sub>
static void solveLong(int d, VBus vBus, sycl::int2 *hBus, int *bBus,
                      sycl::int3 *results, Sub sub, sycl::nd_item<1> item_ct1,
                      int gapOpen_, int gapExtend_, int rows_, int cols_,
                      int cellWidth_, int pruneLow_, int pruneHigh_,
                      int scorerLen_, int subLen_, int match_, int mismatch_,
                      int *hBusScrShr, int *hBusAffShr, sycl::char4 *rowGpu,
                      char *colGpu, int *subGpu) {

  const int groupRangeId = item_ct1.get_group_range(0);
  const int groupId = item_ct1.get_group(0);
  const int localId = item_ct1.get_local_id(0);
  const int localRangeId = item_ct1.get_local_range(0);

  hBusScrShr[localId] = 0;

  bool doWork = true;

  if ((int)groupId <= pruneLow_ || groupId > pruneHigh_) {
    doWork = false;
  }

  int row, col;
  Atom atom;
  sycl::char4 rowCodes;
  sycl::int3 res;

  if (doWork) {

    row = (d + groupId - groupRangeId + 1) * (localRangeId * 4) + localId * 4;
    col = cellWidth_ * (groupRangeId - groupId - 1) - localId + localRangeId;

    if (row < 0 || row >= rows_)
      doWork = false;

    if (doWork) {
      if (groupId == pruneHigh_) {

        // clear only the last steepness
        vBus.mch[(row >> 2) % (groupRangeId * localRangeId)] = 0;
        vBus.scr[(row >> 2) % (groupRangeId * localRangeId)] = INT4_ZERO;
        vBus.aff[(row >> 2) % (groupRangeId * localRangeId)] = INT4_ZERO;

        hBus[col + cellWidth_ - localRangeId - 1] = sycl::int2(0, 0);

        doWork = false;
      }

      if (doWork) {
        atom.mch = vBus.mch[(row >> 2) % (groupRangeId * localRangeId)];
        atom.lScr = vBus.scr[(row >> 2) % (groupRangeId * localRangeId)];
        atom.lAff = vBus.aff[(row >> 2) % (groupRangeId * localRangeId)];

        hBusScrShr[localId] = hBus[col].x();
        hBusAffShr[localId] = hBus[col].y();

        rowCodes = rowGpu[row >> 2];
        res = {0, 0, 0};
      }
    }
  }

  int del;

  for (int i = 0; i < cellWidth_ - localRangeId; ++i, ++col) {

    if (doWork) {
      char columnCode = colGpu[col];

      if (localId == 0) {
        atom.up = hBus[col];
      } else {
        atom.up = sycl::int2(hBusScrShr[localId], hBusAffShr[localId]);
      }

      del = sycl::max((atom.up.x() - gapOpen_), (atom.up.y() - gapExtend_));
      int ins =
          sycl::max((atom.lScr.x() - gapOpen_), (atom.lAff.x() - gapExtend_));

      int mch = atom.mch + sub(columnCode, rowCodes.x(), scorerLen_, match_,
                               mismatch_, subLen_, subGpu);

      atom.rScr.x() = MAX4(0, mch, del, ins);
      atom.rAff.x() = ins;

      del = sycl::max((atom.rScr.x() - gapOpen_), (del - gapExtend_));
      ins = sycl::max((atom.lScr.y() - gapOpen_), (atom.lAff.y() - gapExtend_));

      mch = atom.lScr.x() + sub(columnCode, rowCodes.y(), scorerLen_, match_,
                                mismatch_, subLen_, subGpu);

      atom.rScr.y() = MAX4(0, mch, del, ins);
      atom.rAff.y() = ins;

      del = sycl::max((atom.rScr.y() - gapOpen_), (del - gapExtend_));
      ins = sycl::max((atom.lScr.z() - gapOpen_), (atom.lAff.z() - gapExtend_));

      mch = atom.lScr.y() + sub(columnCode, rowCodes.z(), scorerLen_, match_,
                                mismatch_, subLen_, subGpu);

      atom.rScr.z() = MAX4(0, mch, del, ins);
      atom.rAff.z() = ins;

      del = sycl::max((atom.rScr.z() - gapOpen_), (del - gapExtend_));
      ins = sycl::max((atom.lScr.w() - gapOpen_), (atom.lAff.w() - gapExtend_));

      mch = atom.lScr.z() + sub(columnCode, rowCodes.w(), scorerLen_, match_,
                                mismatch_, subLen_, subGpu);

      atom.rScr.w() = MAX4(0, mch, del, ins);
      atom.rAff.w() = ins;

      if (atom.rScr.x() > res.x()) {
        res.x() = atom.rScr.x();
        res.y() = row;
        res.z() = col;
      }
      if (atom.rScr.y() > res.x()) {
        res.x() = atom.rScr.y();
        res.y() = row + 1;
        res.z() = col;
      }
      if (atom.rScr.z() > res.x()) {
        res.x() = atom.rScr.z();
        res.y() = row + 2;
        res.z() = col;
      }
      if (atom.rScr.w() > res.x()) {
        res.x() = atom.rScr.w();
        res.y() = row + 3;
        res.z() = col;
      }

      atom.mch = atom.up.x();
      atom.lScr = atom.rScr;
      atom.lAff = atom.rAff;
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (doWork) {
      if (localId == localRangeId - 1) {
        hBus[col] = sycl::int2(atom.rScr.w(), del);
      } else {
        hBusScrShr[localId + 1] = atom.rScr.w();
        hBusAffShr[localId + 1] = del;
      }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);
  }

  if (doWork) {
    const int vBusIdx = (row >> 2) % (groupRangeId * localRangeId);
    vBus.mch[vBusIdx] = atom.up.x();
    vBus.scr[vBusIdx] = atom.lScr;
    vBus.aff[vBusIdx] = atom.lAff;

    hBus[col - 1] = sycl::int2(atom.rScr.w(), del);

    if (res.x() > results[groupId * localRangeId + localId].x()) {
      results[groupId * localRangeId + localId] = res;
    }

    // reuse
    hBusScrShr[localId] = res.x();
  }

  item_ct1.barrier(sycl::access::fence_space::local_space);

  if (!doWork)
    return;

  int score = 0;
  int idx = 0;

  for (int i = 0; i < localRangeId; ++i) {

    int shr = hBusScrShr[i];

    if (shr > score) {
      score = shr;
      idx = i;
    }
  }

  if (localId == idx)
    bBus[groupId] = score;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// CPU KERNELS

static void *swEndDataGpuKernel(void *params) try {

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
  if (scorerGetMaxScore(scorer) <= 0) {
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

  sycl::queue dev_q = queues[card];

  int rows = chainGetLength(query);
  int cols = chainGetLength(target);
  int gapOpen = scorerGetGapOpen(scorer);
  int gapExtend = scorerGetGapExtend(scorer);
  int scorerLen = scorerGetMaxCode(scorer);
  int subLen = scorerLen + 1;
  int scalar = scorerIsScalar(scorer);

  TIMER_START("Sw end data %d %d", rows, cols);

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
  sycl::char4 *rowGpu = sycl::malloc_device<sycl::char4>(rowsGpu, dev_q);

  dev_q.memcpy(rowGpu, rowCpu, rowSize).wait();

  memoryUsedGpu += rowSize;

  size_t colSize = colsGpu * sizeof(char);
  char *colGpu = sycl::malloc_device<char>(colsGpu, dev_q);
  dev_q.memcpy(colGpu, colCpu, colSize).wait();

  memoryUsedGpu += colSize;

  size_t hBusSize = colsGpu * sizeof(sycl::int2);
  sycl::int2 *hBus = sycl::malloc_shared<sycl::int2>(colsGpu, dev_q);

  dev_q.memset(hBus, 0, hBusSize).wait();

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

  size_t resultsSize = blocks * threads * sizeof(sycl::int3);
  sycl::int3 *results =
      sycl::malloc_shared<sycl::int3>(blocks * threads, dev_q);
  dev_q.memset(results, 0, resultsSize).wait();

  memoryUsedCpu += resultsSize;
  memoryUsedGpu += resultsSize;

  size_t bSize = blocks * sizeof(int);
  int *b = sycl::malloc_shared<int>(blocks, dev_q);
  dev_q.memset(b, 0, bSize).wait();

  memoryUsedCpu += bSize;
  memoryUsedGpu += bSize;

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

              cgh.parallel_for(sycl::nd_range<1>(blocks * threads, threads),
                               [=](sycl::nd_item<1> item_ct1) {
                                 solveShort(diagonal, vBusGpu, hBus, results,
                                            SubScalar(), item_ct1, gapOpen,
                                            gapExtend, rowsGpu, colsGpu,
                                            cellWidth, pruneLow, pruneHigh,
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

              cgh.parallel_for(
                  sycl::nd_range<1>(blocks * threads, threads),
                  [=](sycl::nd_item<1> item_ct1) {
                    solveLong(diagonal, vBusGpu, hBus, b, results, SubScalar(),
                              item_ct1, gapOpen, gapExtend, rowsGpu, colsGpu,
                              cellWidth, pruneLow, pruneHigh, scorerLen, subLen,
                              match, mismatch, hBusScrShr_acc_ct1.get_pointer(),
                              hBusAffShr_acc_ct1.get_pointer(), rowGpu, colGpu,
                              sub);
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
                                 solveShort(diagonal, vBusGpu, hBus, results,
                                            SubScalarRev(), item_ct1, gapOpen,
                                            gapExtend, rowsGpu, colsGpu,
                                            cellWidth, pruneLow, pruneHigh,
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
                                 solveLong(diagonal, vBusGpu, hBus, b, results,
                                           SubScalarRev(), item_ct1, gapOpen,
                                           gapExtend, rowsGpu, colsGpu,
                                           cellWidth, pruneLow, pruneHigh,
                                           scorerLen, subLen, match, mismatch,
                                           hBusScrShr_acc_ct1.get_pointer(),
                                           hBusAffShr_acc_ct1.get_pointer(),
                                           rowGpu, colGpu, sub);
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
                  solveShort(diagonal, vBusGpu, hBus, results, SubVector(),
                             item_ct1, gapOpen, gapExtend, rowsGpu, colsGpu,
                             cellWidth, pruneLow, pruneHigh, scorerLen, subLen,
                             match, mismatch, hBusScrShr_acc_ct1.get_pointer(),
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
                  solveLong(diagonal, vBusGpu, hBus, b, results, SubVector(),
                            item_ct1, gapOpen, gapExtend, rowsGpu, colsGpu,
                            cellWidth, pruneLow, pruneHigh, scorerLen, subLen,
                            match, mismatch, hBusScrShr_acc_ct1.get_pointer(),
                            hBusAffShr_acc_ct1.get_pointer(), rowGpu, colGpu,
                            sub);
                });
          })
          .wait();
    }

    if (pruning) {
      size_t bSize = pruneHigh * sizeof(int);

      if (score == NO_SCORE) {
        for (int i = 0; i < pruneHigh; ++i) {
          best = std::max(best, b[i]);
        }
      }

      // delta j pruning
      pruneLow = -1;
      for (int i = 0; i < blocks; ++i) {
        int row = (diagonal + 1 + i - blocks + 1) * (threads * 4);
        int col = cellWidth * (blocks - i - 1) - threads;
        if (row >= rowsGpu)
          break;
        if (rowsGpu * (halfPruning ? 2 : 1) - row < cols - col)
          break;
        int d = cols - col;
        int scr = i == blocks - 1 ? b[i] : std::max(b[i], b[i + 1]);
        if ((scr + d * pruneFactor) < best)
          pruneLow = i;
        else
          break;
      }

      // delta i pruning
      if (!halfPruning) {
        pruneHighOld = pruneHigh;
        for (int i = pruneHighOld - 1; i >= 0; --i) {
          int row = (diagonal + 1 + i - blocks + 1) * (threads * 4);
          int col = cellWidth * (blocks - i - 1) - threads;
          if (row < rowsGpu / 2)
            break;
          if (row >= rowsGpu)
            continue;
          if (rowsGpu - row > cols - col)
            break;
          int d = rowsGpu - row;
          int scr1 = d * pruneFactor + (i == blocks - 1 ? 0 : b[i + 1]);
          int scr2 = (d + threads * 2) * pruneFactor + b[i];
          if (scr1 < best && scr2 < best)
            pruneHigh = i;
          else
            break;
        }
      }

      pruned += blocks - (pruneHigh - pruneLow - 1);

      if (pruneLow >= pruneHigh) {
        break;
      }

      if (pruneLow >= 0) {
        int offset = (blocks - pruneLow - 1) * cellWidth - threads;
        size_t size = (colsGpu - offset) * sizeof(sycl::int2);

        dev_q.memset(hBus + offset, 0, size).wait();
      }
    }
  }

  // TIMER_STOP;

  LOG("Pruned percentage %.2f%%", 100.0 * pruned / (diagonals * blocks));

  //**************************************************************************

  //**************************************************************************
  // SAVE RESULTS

  // save only if needed
  if (scores != NULL && affines != NULL) {

    *scores = (int *)malloc(cols * sizeof(int));
    *affines = (int *)malloc(cols * sizeof(int));

    for (int i = 0; i < cols; ++i) {
      (*scores)[i] = hBus[i].x();
      (*affines)[i] = hBus[i].y();
    }
  }

  sycl::int3 res = results[0];
  for (int i = 1; i < blocks * threads; ++i) {
    if (results[i].x() > res.x()) {
      res = results[i];
    }
  }

  res.y() -= (rowsGpu - rows); // restore padding

  // check if the result updated in the padded part
  if (res.y() >= rows) {
    res.z() += rows - res.y() - 1;
    res.y() += rows - res.y() - 1;
  }

  if (res.z() >= cols) {
    res.y() += cols - res.z() - 1;
    res.z() += cols - res.z() - 1;
  }

  *outScore = res.x();
  *queryEnd = res.y();
  *targetEnd = res.z();

  LOG("Score: %d, (%d, %d)", *outScore, *queryEnd, *targetEnd);

  //**************************************************************************

  //**************************************************************************
  // CLEAN MEMORY

  // free(sub);
  free(rowCpu);
  free(colCpu);
  // free(results);

  sycl::free(sub, dev_q);
  sycl::free(b, dev_q);
  sycl::free(hBus, dev_q);
  // sycl::free(sub, dev_q);
  sycl::free(rowGpu, dev_q);
  sycl::free(colGpu, dev_q);
  sycl::free(vBusGpu.mch, dev_q);
  sycl::free(vBusGpu.scr, dev_q);
  sycl::free(vBusGpu.aff, dev_q);
  // sycl::free(hBus, dev_q);
  sycl::free(results, dev_q);
  // sycl::free(bGpu, dev_q);

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
