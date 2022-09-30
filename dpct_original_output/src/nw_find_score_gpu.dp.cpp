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
#include <cmath>

#define MAX_THREADS MAX(THREADS_SM1, THREADS_SM2)

#define THREADS_SM1 64
#define BLOCKS_SM1  240

#define THREADS_SM2 128
#define BLOCKS_SM2  480

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
    int* mch;
    sycl::int4 *scr;
    sycl::int4 *aff;
} VBus;

typedef struct Context {
    int* queryStart;
    int* targetStart;
    int score; 
    Chain* query;
    int queryFrontGap;
    Chain* target;
    Scorer* scorer;
    int card;
} Context;

static dpct::constant_memory<int, 0> queryFrontGap_;

static dpct::constant_memory<int, 0> gapOpen_;
static dpct::constant_memory<int, 0> gapExtend_;
static dpct::constant_memory<int, 0> gapDiff_;

static dpct::constant_memory<int, 0> rows_;
static dpct::constant_memory<int, 0> cols_;
static dpct::constant_memory<int, 0> cellWidth_;

static dpct::constant_memory<int, 0> score_;
static dpct::constant_memory<int, 0> pLeft_;
static dpct::constant_memory<int, 0> pRight_;

static dpct::constant_memory<int, 0> scorerLen_;
static dpct::constant_memory<int, 0> subLen_;

static dpct::constant_memory<int, 0> match_;
static dpct::constant_memory<int, 0> mismatch_;

dpct::image_wrapper<sycl::char4, 1> rowTexture;
/*
DPCT1059:363: SYCL only supports 4-channel image format. Adjust the code.
*/
dpct::image_wrapper<char, 1> colTexture;
/*
DPCT1059:364: SYCL only supports 4-channel image format. Adjust the code.
*/
dpct::image_wrapper<sycl::int2, 1> hBusTexture;
/*
DPCT1059:365: SYCL only supports 4-channel image format. Adjust the code.
*/
dpct::image_wrapper<int, 1> subTexture;

static dpct::global_memory<sycl::int2, 0> res_;

//******************************************************************************
// PUBLIC

extern void nwFindScoreGpu(int* queryStart, int* targetStart, Chain* query, 
    int queryFrontGap, Chain* target, Scorer* scorer, int score, int card, 
    Thread* thread);

//******************************************************************************

//******************************************************************************
// PRIVATE

// With visual c++ compiler and prototypes declared cuda global memory variables
// do not work. No questions asked.
#ifndef _WIN32
static int gap(int idx, int queryFrontGap_, int gapOpen_, int gapExtend_,
               int gapDiff_);

template <class Sub>
static void
solveShortDelegated(int d, VBus vBus, sycl::int2 *hBus, Sub sub,
                    sycl::nd_item<3> item_ct1, int queryFrontGap_, int gapOpen_,
                    int gapExtend_, int gapDiff_, int subLen_, int *hBusScrShr,
                    int *hBusAffShr,
                    dpct::image_accessor_ext<int, 1> subTexture,
                    dpct::image_accessor_ext<sycl::char4, 1> rowTexture,
                    dpct::image_accessor_ext<char, 1> colTexture,
                    dpct::image_accessor_ext<sycl::int2, 1> hBusTexture);

template <class Sub>
static void
solveShortNormal(int d, VBus vBus, sycl::int2 *hBus, Sub sub,
                 sycl::nd_item<3> item_ct1, int subLen_, int *hBusScrShr,
                 int *hBusAffShr, dpct::image_accessor_ext<int, 1> subTexture,
                 dpct::image_accessor_ext<sycl::char4, 1> rowTexture,
                 dpct::image_accessor_ext<char, 1> colTexture,
                 dpct::image_accessor_ext<sycl::int2, 1> hBusTexture);

template <class Sub>
static void solveShort(int d, VBus vBus, sycl::int2 *hBus, Sub sub,
                       sycl::nd_item<3> item_ct1, int queryFrontGap_,
                       int gapOpen_, int gapExtend_, int gapDiff_, int subLen_,
                       int *hBusScrShr, int *hBusAffShr,
                       dpct::image_accessor_ext<int, 1> subTexture,
                       dpct::image_accessor_ext<sycl::char4, 1> rowTexture,
                       dpct::image_accessor_ext<char, 1> colTexture,
                       dpct::image_accessor_ext<sycl::int2, 1> hBusTexture);

template <class Sub>
static void solveLong(int d, VBus vBus, sycl::int2 *hBus, Sub sub,
                      sycl::nd_item<3> item_ct1, int subLen_, int *hBusScrShr,
                      int *hBusAffShr,
                      dpct::image_accessor_ext<int, 1> subTexture,
                      dpct::image_accessor_ext<sycl::char4, 1> rowTexture,
                      dpct::image_accessor_ext<char, 1> colTexture,
                      dpct::image_accessor_ext<sycl::int2, 1> hBusTexture);

#endif

static void* kernel(void* params);
    
//******************************************************************************

//******************************************************************************
// PUBLIC

extern void nwFindScoreGpu(int* queryStart, int* targetStart, Chain* query, 
    int queryFrontGap, Chain* target, Scorer* scorer, int score, int card, 
    Thread* thread) {
    
    Context* param = (Context*) malloc(sizeof(Context));

    param->queryStart = queryStart;
    param->targetStart = targetStart;
    param->score = score;
    param->query = query;
    param->queryFrontGap = queryFrontGap;
    param->target = target;
    param->scorer = scorer;
    param->card = card;
    
    if (thread == NULL) {
        kernel(param);
    } else {
        threadCreate(thread, kernel, (void*) param);
    }
}

//******************************************************************************

//******************************************************************************
// PRIVATE

//------------------------------------------------------------------------------
// FUNCTORS

class SubScalar {
public:
    int operator () (char a, char b, int match_, int mismatch_) {
        return a == b ? match_ : mismatch_;
    }
};

class SubScalarRev {
public:
    int operator () (char a, char b, int scorerLen_, int match_, int mismatch_) {
        return (a == b ? match_ : mismatch_) * (a < scorerLen_ && b < scorerLen_);
    }
};

class SubVector {
public:
    SYCL_EXTERNAL int operator()(char a, char b, int subLen_,
                                 dpct::image_accessor_ext<int, 1> subTexture) {
        return subTexture.read((a * subLen_) + b);
    }
};

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// GPU KERNELS

static int gap(int idx, int queryFrontGap_, int gapOpen_, int gapExtend_,
               int gapDiff_) {
    return -gapOpen_ - gapExtend_ * idx + queryFrontGap_ * gapDiff_;
}

template <class Sub>
static void
solveShortDelegated(int d, VBus vBus, sycl::int2 *hBus, Sub sub,
                    sycl::nd_item<3> item_ct1, int queryFrontGap_, int gapOpen_,
                    int gapExtend_, int gapDiff_, int subLen_, int *hBusScrShr,
                    int *hBusAffShr,
                    dpct::image_accessor_ext<int, 1> subTexture,
                    dpct::image_accessor_ext<sycl::char4, 1> rowTexture,
                    dpct::image_accessor_ext<char, 1> colTexture,
                    dpct::image_accessor_ext<sycl::int2, 1> hBusTexture) {

    int row = (d + item_ct1.get_group(2) - item_ct1.get_group_range(2) + 1) *
                  (item_ct1.get_local_range(2) * 4) +
              item_ct1.get_local_id(2) * 4;
    int col =
        cellWidth_ * (item_ct1.get_group_range(2) - item_ct1.get_group(2) - 1) -
        item_ct1.get_local_id(2);

    if (row < 0) return;

    row -= (col < 0) *
           (item_ct1.get_group_range(2) * item_ct1.get_local_range(2) * 4);
    col += (col < 0) * cols_;

    int x1 =
        cellWidth_ * (item_ct1.get_group_range(2) - item_ct1.get_group(2) - 1) +
        item_ct1.get_local_range(2);
    int y1 = (d + item_ct1.get_group(2) - item_ct1.get_group_range(2) + 1) *
             (item_ct1.get_local_range(2) * 4);

    int x2 =
        cellWidth_ * (item_ct1.get_group_range(2) - item_ct1.get_group(2) - 1) -
        item_ct1.get_local_range(2);
    int y2 = (d + item_ct1.get_group(2) - item_ct1.get_group_range(2) + 2) *
             (item_ct1.get_local_range(2) * 4);

    y2 -= (x2 < 0) *
          (item_ct1.get_group_range(2) * item_ct1.get_local_range(2) * 4);
    x2 += (x2 < 0) * cols_;
    
    if (y1 - x1 > pLeft_ && (x2 - y2 > pRight_ || y2 < 0)) {

        row += (col != 0) * item_ct1.get_group_range(2) *
               item_ct1.get_local_range(2) * 4;
        vBus.mch[(row >> 2) % (item_ct1.get_group_range(2) *
                               item_ct1.get_local_range(2))] = SCORE_MIN;
        vBus.scr[(row >> 2) % (item_ct1.get_group_range(2) *
                               item_ct1.get_local_range(2))] = SCORE4_MIN;
        vBus.aff[(row >> 2) % (item_ct1.get_group_range(2) *
                               item_ct1.get_local_range(2))] = SCORE4_MIN;

        hBus[col] = sycl::int2(SCORE_MIN, SCORE_MIN);

        return;
    }
    
    Atom atom;

    if (0 <= row && row < rows_ && col > 0) {
        atom.mch = vBus.mch[(row >> 2) % (item_ct1.get_group_range(2) *
                                          item_ct1.get_local_range(2))];
        VEC4_ASSIGN(atom.lScr,
                    vBus.scr[(row >> 2) % (item_ct1.get_group_range(2) *
                                           item_ct1.get_local_range(2))]);
        VEC4_ASSIGN(atom.lAff,
                    vBus.aff[(row >> 2) % (item_ct1.get_group_range(2) *
                                           item_ct1.get_local_range(2))]);
    } else {
        atom.mch =
            gap(row - 1, queryFrontGap_, gapOpen_, gapExtend_, gapDiff_) *
            (row != 0);
        atom.lScr = sycl::int4(
            gap(row, queryFrontGap_, gapOpen_, gapExtend_, gapDiff_),
            gap(row + 1, queryFrontGap_, gapOpen_, gapExtend_, gapDiff_),
            gap(row + 2, queryFrontGap_, gapOpen_, gapExtend_, gapDiff_),
            gap(row + 3, queryFrontGap_, gapOpen_, gapExtend_, gapDiff_));
        atom.lAff = SCORE4_MIN;
    }

    hBusScrShr[item_ct1.get_local_id(2)] = hBusTexture.read(col).x();
    hBusAffShr[item_ct1.get_local_id(2)] = hBusTexture.read(col).y();

    sycl::char4 rowCodes = rowTexture.read(row >> 2);

    int del;

    for (int i = 0; i < item_ct1.get_local_range(2); ++i) {

        if (0 <= row && row < rows_) {

            char columnCode = colTexture.read(col);

            if (item_ct1.get_local_id(2) == 0) {
                atom.up = hBusTexture.read(col);
            } else {
                atom.up.x() = hBusScrShr[item_ct1.get_local_id(2)];
                atom.up.y() = hBusAffShr[item_ct1.get_local_id(2)];
            }

            del = sycl::max((int)(atom.up.x() - gapOpen_),
                            (int)(atom.up.y() - gapExtend_));
            int ins = sycl::max((int)(atom.lScr.x() - gapOpen_),
                                (int)(atom.lAff.x() - gapExtend_));
            /*
            DPCT1084:368: The function call has multiple migration results in
            different template instantiations that could not be unified. You may
            need to adjust the code.
            */
            int mch = atom.mch + sub(columnCode, rowCodes.x(), subLen_, subTexture);

            atom.rScr.x() = MAX3(mch, del, ins);
            atom.rAff.x() = ins;

            del = sycl::max((int)(atom.rScr.x() - gapOpen_), (int)(del - gapExtend_));
            ins = sycl::max((int)(atom.lScr.y() - gapOpen_),
                            (int)(atom.lAff.y() - gapExtend_));
            /*
            DPCT1084:369: The function call has multiple migration results in
            different template instantiations that could not be unified. You may
            need to adjust the code.
            */
            mch = atom.lScr.x() + sub(columnCode, rowCodes.y(), subLen_, subTexture);

            atom.rScr.y() = MAX3(mch, del, ins);
            atom.rAff.y() = ins;

            del = sycl::max((int)(atom.rScr.y() - gapOpen_), (int)(del - gapExtend_));
            ins = sycl::max((int)(atom.lScr.z() - gapOpen_),
                            (int)(atom.lAff.z() - gapExtend_));
            /*
            DPCT1084:370: The function call has multiple migration results in
            different template instantiations that could not be unified. You may
            need to adjust the code.
            */
            mch = atom.lScr.y() + sub(columnCode, rowCodes.z(), subLen_, subTexture);

            atom.rScr.z() = MAX3(mch, del, ins);
            atom.rAff.z() = ins;

            del = sycl::max((int)(atom.rScr.z() - gapOpen_), (int)(del - gapExtend_));
            ins = sycl::max((int)(atom.lScr.w() - gapOpen_),
                            (int)(atom.lAff.w() - gapExtend_));
            /*
            DPCT1084:371: The function call has multiple migration results in
            different template instantiations that could not be unified. You may
            need to adjust the code.
            */
            mch = atom.lScr.z() + sub(columnCode, rowCodes.w(), subLen_, subTexture);

            atom.rScr.w() = MAX3(mch, del, ins);
            atom.rAff.w() = ins;

            if (atom.rScr.x() == score_) {
             res_.x() = row; res_.y() = col;
            } else if (atom.rScr.y() == score_) {
             res_.x() = row + 1; res_.y() = col;
            } else if (atom.rScr.z() == score_) {
             res_.x() = row + 2; res_.y() = col;
            } else if (atom.rScr.w() == score_) {
             res_.x() = row + 3; res_.y() = col;
            }

            atom.mch = atom.up.x();
            VEC4_ASSIGN(atom.lScr, atom.rScr);
            VEC4_ASSIGN(atom.lAff, atom.rAff);
        }

        /*
        DPCT1065:366: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        if (0 <= row && row < rows_) {

            if (item_ct1.get_local_id(2) == item_ct1.get_local_range(2) - 1 ||
                i == item_ct1.get_local_range(2) - 1) {
                VEC2_ASSIGN(hBus[col], sycl::int2(atom.rScr.w(), del));
            } else {
                hBusScrShr[item_ct1.get_local_id(2) + 1] = atom.rScr.w();
                hBusAffShr[item_ct1.get_local_id(2) + 1] = del;
            }
        }

        ++col;

        if (col == cols_) {

            col = 0;
            row = row + item_ct1.get_group_range(2) * item_ct1.get_local_range(2) * 4;

            atom.mch =
                gap(row - 1, queryFrontGap_, gapOpen_, gapExtend_, gapDiff_) *
                (row != 0);
            atom.lScr = sycl::int4(
                gap(row, queryFrontGap_, gapOpen_, gapExtend_, gapDiff_),
                gap(row + 1, queryFrontGap_, gapOpen_, gapExtend_, gapDiff_),
                gap(row + 2, queryFrontGap_, gapOpen_, gapExtend_, gapDiff_),
                gap(row + 3, queryFrontGap_, gapOpen_, gapExtend_, gapDiff_));
            atom.lAff = SCORE4_MIN;

            rowCodes = rowTexture.read(row >> 2);
        }

        /*
        DPCT1065:367: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    if (row < 0 || row >= rows_) return;

    vBus.mch[(row >> 2) % (item_ct1.get_group_range(2) *
                           item_ct1.get_local_range(2))] = atom.up.x();
    VEC4_ASSIGN(vBus.scr[(row >> 2) % (item_ct1.get_group_range(2) *
                                       item_ct1.get_local_range(2))],
                atom.lScr);
    VEC4_ASSIGN(vBus.aff[(row >> 2) % (item_ct1.get_group_range(2) *
                                       item_ct1.get_local_range(2))],
                atom.lAff);
}

template <class Sub>
static void
solveShortNormal(int d, VBus vBus, sycl::int2 *hBus, Sub sub,
                 sycl::nd_item<3> item_ct1, int subLen_, int *hBusScrShr,
                 int *hBusAffShr, dpct::image_accessor_ext<int, 1> subTexture,
                 dpct::image_accessor_ext<sycl::char4, 1> rowTexture,
                 dpct::image_accessor_ext<char, 1> colTexture,
                 dpct::image_accessor_ext<sycl::int2, 1> hBusTexture) {

    int row = (d + item_ct1.get_group(2) - item_ct1.get_group_range(2) + 1) *
                  (item_ct1.get_local_range(2) * 4) +
              item_ct1.get_local_id(2) * 4;
    int col =
        cellWidth_ * (item_ct1.get_group_range(2) - item_ct1.get_group(2) - 1) -
        item_ct1.get_local_id(2);

    if (row < 0 || row >= rows_) return;

    int x1 =
        cellWidth_ * (item_ct1.get_group_range(2) - item_ct1.get_group(2) - 1) +
        item_ct1.get_local_range(2);
    int y1 = (d + item_ct1.get_group(2) - item_ct1.get_group_range(2) + 1) *
             (item_ct1.get_local_range(2) * 4);

    if (y1 - x1 > pLeft_) {

        // only clear right, down is irelevant
        vBus.mch[(row >> 2) % (item_ct1.get_group_range(2) *
                               item_ct1.get_local_range(2))] = SCORE_MIN;
        vBus.scr[(row >> 2) % (item_ct1.get_group_range(2) *
                               item_ct1.get_local_range(2))] = SCORE4_MIN;
        vBus.aff[(row >> 2) % (item_ct1.get_group_range(2) *
                               item_ct1.get_local_range(2))] = SCORE4_MIN;

        return;
    }

    int x2 =
        cellWidth_ * (item_ct1.get_group_range(2) - item_ct1.get_group(2) - 1) -
        item_ct1.get_local_range(2);
    int y2 = (d + item_ct1.get_group(2) - item_ct1.get_group_range(2) + 2) *
             (item_ct1.get_local_range(2) * 4);

    if (x2 - y2 > pRight_) {

        // clear right
        vBus.mch[(row >> 2) % (item_ct1.get_group_range(2) *
                               item_ct1.get_local_range(2))] = SCORE_MIN;
        vBus.scr[(row >> 2) % (item_ct1.get_group_range(2) *
                               item_ct1.get_local_range(2))] = SCORE4_MIN;
        vBus.aff[(row >> 2) % (item_ct1.get_group_range(2) *
                               item_ct1.get_local_range(2))] = SCORE4_MIN;

        // clear down
        hBus[col] = sycl::int2(SCORE_MIN, SCORE_MIN);
        hBus[col + item_ct1.get_local_range(2)] = sycl::int2(SCORE_MIN, SCORE_MIN);

        return;
    }

    Atom atom;
    atom.mch = vBus.mch[(row >> 2) % (item_ct1.get_group_range(2) *
                                      item_ct1.get_local_range(2))];
    VEC4_ASSIGN(atom.lScr,
                vBus.scr[(row >> 2) % (item_ct1.get_group_range(2) *
                                       item_ct1.get_local_range(2))]);
    VEC4_ASSIGN(atom.lAff,
                vBus.aff[(row >> 2) % (item_ct1.get_group_range(2) *
                                       item_ct1.get_local_range(2))]);

    hBusScrShr[item_ct1.get_local_id(2)] = hBusTexture.read(col).x();
    hBusAffShr[item_ct1.get_local_id(2)] = hBusTexture.read(col).y();

    const sycl::char4 rowCodes = rowTexture.read(row >> 2);

    int del;

    for (int i = 0; i < item_ct1.get_local_range(2); ++i, ++col) {

        char columnCode = colTexture.read(col);

        if (item_ct1.get_local_id(2) == 0) {
            atom.up = hBusTexture.read(col);
        } else {
            atom.up.x() = hBusScrShr[item_ct1.get_local_id(2)];
            atom.up.y() = hBusAffShr[item_ct1.get_local_id(2)];
        }

        del = sycl::max((int)(atom.up.x() - gapOpen_),
                        (int)(atom.up.y() - gapExtend_));
        int ins = sycl::max((int)(atom.lScr.x() - gapOpen_),
                            (int)(atom.lAff.x() - gapExtend_));
        /*
        DPCT1084:374: The function call has multiple migration results in
        different template instantiations that could not be unified. You may
        need to adjust the code.
        */
        int mch = atom.mch + sub(columnCode, rowCodes.x(), subLen_, subTexture);

        atom.rScr.x() = MAX3(mch, del, ins);
        atom.rAff.x() = ins;

        del = sycl::max((int)(atom.rScr.x() - gapOpen_), (int)(del - gapExtend_));
        ins = sycl::max((int)(atom.lScr.y() - gapOpen_),
                        (int)(atom.lAff.y() - gapExtend_));
        /*
        DPCT1084:375: The function call has multiple migration results in
        different template instantiations that could not be unified. You may
        need to adjust the code.
        */
        mch = atom.lScr.x() + sub(columnCode, rowCodes.y(), subLen_, subTexture);

        atom.rScr.y() = MAX3(mch, del, ins);
        atom.rAff.y() = ins;

        del = sycl::max((int)(atom.rScr.y() - gapOpen_), (int)(del - gapExtend_));
        ins = sycl::max((int)(atom.lScr.z() - gapOpen_),
                        (int)(atom.lAff.z() - gapExtend_));
        /*
        DPCT1084:376: The function call has multiple migration results in
        different template instantiations that could not be unified. You may
        need to adjust the code.
        */
        mch = atom.lScr.y() + sub(columnCode, rowCodes.z(), subLen_, subTexture);

        atom.rScr.z() = MAX3(mch, del, ins);
        atom.rAff.z() = ins;

        del = sycl::max((int)(atom.rScr.z() - gapOpen_), (int)(del - gapExtend_));
        ins = sycl::max((int)(atom.lScr.w() - gapOpen_),
                        (int)(atom.lAff.w() - gapExtend_));
        /*
        DPCT1084:377: The function call has multiple migration results in
        different template instantiations that could not be unified. You may
        need to adjust the code.
        */
        mch = atom.lScr.z() + sub(columnCode, rowCodes.w(), subLen_, subTexture);

        atom.rScr.w() = MAX3(mch, del, ins);
        atom.rAff.w() = ins;

        if (atom.rScr.x() == score_) {
         res_.x() = row; res_.y() = col;
        } else if (atom.rScr.y() == score_) {
         res_.x() = row + 1; res_.y() = col;
        } else if (atom.rScr.z() == score_) {
         res_.x() = row + 2; res_.y() = col;
        } else if (atom.rScr.w() == score_) {
         res_.x() = row + 3; res_.y() = col;
        }

        atom.mch = atom.up.x();
        VEC4_ASSIGN(atom.lScr, atom.rScr);
        VEC4_ASSIGN(atom.lAff, atom.rAff);

        /*
        DPCT1065:372: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        if (item_ct1.get_local_id(2) == item_ct1.get_local_range(2) - 1) {
            VEC2_ASSIGN(hBus[col], sycl::int2(atom.rScr.w(), del));
        } else {
            hBusScrShr[item_ct1.get_local_id(2) + 1] = atom.rScr.w();
            hBusAffShr[item_ct1.get_local_id(2) + 1] = del;
        }

        /*
        DPCT1065:373: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    VEC2_ASSIGN(hBus[col - 1], sycl::int2(atom.rScr.w(), del));

    const int vBusIdx = (row >> 2) % (item_ct1.get_group_range(2) *
                                      item_ct1.get_local_range(2));
    vBus.mch[vBusIdx] = atom.up.x();
    VEC4_ASSIGN(vBus.scr[vBusIdx], atom.lScr);
    VEC4_ASSIGN(vBus.aff[vBusIdx], atom.lAff);
}

template <class Sub>
static void solveShort(int d, VBus vBus, sycl::int2 *hBus, Sub sub,
                       sycl::nd_item<3> item_ct1, int queryFrontGap_,
                       int gapOpen_, int gapExtend_, int gapDiff_, int subLen_,
                       int *hBusScrShr, int *hBusAffShr,
                       dpct::image_accessor_ext<int, 1> subTexture,
                       dpct::image_accessor_ext<sycl::char4, 1> rowTexture,
                       dpct::image_accessor_ext<char, 1> colTexture,
                       dpct::image_accessor_ext<sycl::int2, 1> hBusTexture) {

    if (item_ct1.get_group(2) == (item_ct1.get_group_range(2) - 1)) {
        solveShortDelegated(d, vBus, hBus, sub, item_ct1, queryFrontGap_,
                            gapOpen_, gapExtend_, gapDiff_, subLen_, hBusScrShr,
                            hBusAffShr, subTexture, rowTexture, colTexture,
                            hBusTexture);
    } else {
        solveShortNormal(d, vBus, hBus, sub, item_ct1, subLen_, hBusScrShr,
                         hBusAffShr, subTexture, rowTexture, colTexture,
                         hBusTexture);
    }
}

template <class Sub>
static void solveLong(int d, VBus vBus, sycl::int2 *hBus, Sub sub,
                      sycl::nd_item<3> item_ct1, int subLen_, int *hBusScrShr,
                      int *hBusAffShr,
                      dpct::image_accessor_ext<int, 1> subTexture,
                      dpct::image_accessor_ext<sycl::char4, 1> rowTexture,
                      dpct::image_accessor_ext<char, 1> colTexture,
                      dpct::image_accessor_ext<sycl::int2, 1> hBusTexture) {

    int row = (d + item_ct1.get_group(2) - item_ct1.get_group_range(2) + 1) *
                  (item_ct1.get_local_range(2) * 4) +
              item_ct1.get_local_id(2) * 4;
    int col =
        cellWidth_ * (item_ct1.get_group_range(2) - item_ct1.get_group(2) - 1) -
        item_ct1.get_local_id(2) + item_ct1.get_local_range(2);

    if (row < 0 || row >= rows_) return;

    int x1 =
        cellWidth_ * (item_ct1.get_group_range(2) - item_ct1.get_group(2) - 1) +
        cellWidth_;
    int y1 = (d + item_ct1.get_group(2) - item_ct1.get_group_range(2) + 1) *
             (item_ct1.get_local_range(2) * 4);

    if (y1 - x1 > pLeft_) {

        vBus.mch[(row >> 2) % (item_ct1.get_group_range(2) *
                               item_ct1.get_local_range(2))] = SCORE_MIN;
        vBus.scr[(row >> 2) % (item_ct1.get_group_range(2) *
                               item_ct1.get_local_range(2))] = SCORE4_MIN;
        vBus.aff[(row >> 2) % (item_ct1.get_group_range(2) *
                               item_ct1.get_local_range(2))] = SCORE4_MIN;

        return;
    }

    int x2 =
        cellWidth_ * (item_ct1.get_group_range(2) - item_ct1.get_group(2) - 1);
    int y2 = (d + item_ct1.get_group(2) - item_ct1.get_group_range(2) + 2) *
             (item_ct1.get_local_range(2) * 4);

    if (x2 - y2 > pRight_) {

        vBus.mch[(row >> 2) % (item_ct1.get_group_range(2) *
                               item_ct1.get_local_range(2))] = SCORE_MIN;
        vBus.scr[(row >> 2) % (item_ct1.get_group_range(2) *
                               item_ct1.get_local_range(2))] = SCORE4_MIN;
        vBus.aff[(row >> 2) % (item_ct1.get_group_range(2) *
                               item_ct1.get_local_range(2))] = SCORE4_MIN;

        for (int i = 0; i < cellWidth_ - item_ct1.get_local_range(2);
             i += item_ct1.get_local_range(2)) {
            hBus[col + i] = sycl::int2(SCORE_MIN, SCORE_MIN);
        }
        
        return;
    }
    
    Atom atom;
    atom.mch = vBus.mch[(row >> 2) % (item_ct1.get_group_range(2) *
                                      item_ct1.get_local_range(2))];
    VEC4_ASSIGN(atom.lScr,
                vBus.scr[(row >> 2) % (item_ct1.get_group_range(2) *
                                       item_ct1.get_local_range(2))]);
    VEC4_ASSIGN(atom.lAff,
                vBus.aff[(row >> 2) % (item_ct1.get_group_range(2) *
                                       item_ct1.get_local_range(2))]);

    hBusScrShr[item_ct1.get_local_id(2)] = hBusTexture.read(col).x();
    hBusAffShr[item_ct1.get_local_id(2)] = hBusTexture.read(col).y();

    const sycl::char4 rowCodes = rowTexture.read(row >> 2);

    int del;

    for (int i = 0; i < cellWidth_ - item_ct1.get_local_range(2); ++i, ++col) {

        char columnCode = colTexture.read(col);

        if (item_ct1.get_local_id(2) == 0) {
            atom.up = hBusTexture.read(col);
        } else {
            atom.up.x() = hBusScrShr[item_ct1.get_local_id(2)];
            atom.up.y() = hBusAffShr[item_ct1.get_local_id(2)];
        }

        del = sycl::max((int)(atom.up.x() - gapOpen_),
                        (int)(atom.up.y() - gapExtend_));
        int ins = sycl::max((int)(atom.lScr.x() - gapOpen_),
                            (int)(atom.lAff.x() - gapExtend_));
        /*
        DPCT1084:380: The function call has multiple migration results in
        different template instantiations that could not be unified. You may
        need to adjust the code.
        */
        int mch = atom.mch + sub(columnCode, rowCodes.x(), subLen_, subTexture);

        atom.rScr.x() = MAX3(mch, del, ins);
        atom.rAff.x() = ins;

        del = sycl::max((int)(atom.rScr.x() - gapOpen_), (int)(del - gapExtend_));
        ins = sycl::max((int)(atom.lScr.y() - gapOpen_),
                        (int)(atom.lAff.y() - gapExtend_));
        /*
        DPCT1084:381: The function call has multiple migration results in
        different template instantiations that could not be unified. You may
        need to adjust the code.
        */
        mch = atom.lScr.x() + sub(columnCode, rowCodes.y(), subLen_, subTexture);

        atom.rScr.y() = MAX3(mch, del, ins);
        atom.rAff.y() = ins;

        del = sycl::max((int)(atom.rScr.y() - gapOpen_), (int)(del - gapExtend_));
        ins = sycl::max((int)(atom.lScr.z() - gapOpen_),
                        (int)(atom.lAff.z() - gapExtend_));
        /*
        DPCT1084:382: The function call has multiple migration results in
        different template instantiations that could not be unified. You may
        need to adjust the code.
        */
        mch = atom.lScr.y() + sub(columnCode, rowCodes.z(), subLen_, subTexture);

        atom.rScr.z() = MAX3(mch, del, ins);
        atom.rAff.z() = ins;

        del = sycl::max((int)(atom.rScr.z() - gapOpen_), (int)(del - gapExtend_));
        ins = sycl::max((int)(atom.lScr.w() - gapOpen_),
                        (int)(atom.lAff.w() - gapExtend_));
        /*
        DPCT1084:383: The function call has multiple migration results in
        different template instantiations that could not be unified. You may
        need to adjust the code.
        */
        mch = atom.lScr.z() + sub(columnCode, rowCodes.w(), subLen_, subTexture);

        atom.rScr.w() = MAX3(mch, del, ins);
        atom.rAff.w() = ins;

        if (atom.rScr.x() == score_) {
         res_.x() = row; res_.y() = col;
        } else if (atom.rScr.y() == score_) {
         res_.x() = row + 1; res_.y() = col;
        } else if (atom.rScr.z() == score_) {
         res_.x() = row + 2; res_.y() = col;
        } else if (atom.rScr.w() == score_) {
         res_.x() = row + 3; res_.y() = col;
        }

        atom.mch = atom.up.x();
        VEC4_ASSIGN(atom.lScr, atom.rScr);
        VEC4_ASSIGN(atom.lAff, atom.rAff);

        /*
        DPCT1065:378: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        if (item_ct1.get_local_id(2) == item_ct1.get_local_range(2) - 1) {
            VEC2_ASSIGN(hBus[col], sycl::int2(atom.rScr.w(), del));
        } else {
            hBusScrShr[item_ct1.get_local_id(2) + 1] = atom.rScr.w();
            hBusAffShr[item_ct1.get_local_id(2) + 1] = del;
        }

        /*
        DPCT1065:379: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    VEC2_ASSIGN(hBus[col - 1], sycl::int2(atom.rScr.w(), del));

    const int vBusIdx = (row >> 2) % (item_ct1.get_group_range(2) *
                                      item_ct1.get_local_range(2));
    vBus.mch[vBusIdx] = atom.up.x();
    VEC4_ASSIGN(vBus.scr[vBusIdx], atom.lScr);
    VEC4_ASSIGN(vBus.aff[vBusIdx], atom.lAff);  
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// CPU KERNELS

static void *kernel(void *params) try {

    Context* context = (Context*) params;
  
    int* queryStart = context->queryStart;
    int* targetStart = context->targetStart;
    int score = context->score;
    Chain* query = context->query;
    int queryFrontGap = context->queryFrontGap;
    Chain* target = context->target;
    Scorer* scorer = context->scorer;
    int card = context->card;
    
    int currentCard;
    CUDA_SAFE_CALL(currentCard = dpct::dev_mgr::instance().current_device_id());
    if (currentCard != card) {
        // CUDA_SAFE_CALL(cudaThreadExit());
        /*
        DPCT1093:384: The "card" may not be the best XPU device. Adjust the
        selected device if needed.
        */
        /*
        DPCT1003:385: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_SAFE_CALL((dpct::dev_mgr::instance().select_device(card), 0));
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

    dpct::device_info properties;
    CUDA_SAFE_CALL(
        (dpct::dev_mgr::instance().get_device(card).get_device_info(properties),
         0));

    int threads;
    int blocks;
    /*
    DPCT1005:386: The SYCL device version is different from CUDA Compute
    Compatibility. You may need to rewrite this code.
    */
    if (properties.get_major_version() < 2) {
        threads = THREADS_SM1;
        blocks = BLOCKS_SM1;
    } else {
        threads = THREADS_SM2;
        blocks = BLOCKS_SM2;
    }
    
    ASSERT(threads * 2 <= cols, "too short gpu target chain");

    if (threads * blocks * 2 > cols) {
        blocks = (int) (cols / (threads * 2.));
        blocks = blocks <= 30 ? blocks : blocks - (blocks % 30);
        LOG("Blocks trimmed to: %d", blocks);
    }
    
    int cellHeight = 4 * threads;
    int rowsGpu = rows + (4 - rows % 4) % 4;
    int dRow = (rowsGpu - rows);
    
    int colsGpu = cols + (blocks - cols % blocks) % blocks;
    int cellWidth = colsGpu / blocks;

    int diagonals = blocks + (int) ceil((float) rowsGpu / cellHeight);

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
    
    char* rowCpu = (char*) malloc(rowsGpu * sizeof(char));
    memset(rowCpu + rows, scorerLen, dRow * sizeof(char));
    chainCopyCodes(query, rowCpu);
    memoryUsedCpu += rowsGpu * sizeof(char);

    char* colCpu = (char*) malloc(colsGpu * sizeof(char));
    memset(colCpu + cols, scorerLen + scalar, (colsGpu - cols) * sizeof(char));
    chainCopyCodes(target, colCpu);
    memoryUsedCpu += colsGpu * sizeof(char);
    
    //**************************************************************************

    //**************************************************************************
    // INIT GPU
    size_t rowSize = rowsGpu * sizeof(char);
    sycl::char4 *rowGpu;
    CUDA_SAFE_CALL((rowGpu = (sycl::char4 *)sycl::malloc_device(
                        rowSize, dpct::get_default_queue()),
                    0));
    CUDA_SAFE_CALL(
        (dpct::get_default_queue().memcpy(rowGpu, rowCpu, rowSize).wait(), 0));
    CUDA_SAFE_CALL((rowTexture.attach(rowGpu, rowSize), 0));
    memoryUsedGpu += rowSize;

    size_t colSize = colsGpu * sizeof(char);
    char* colGpu;
    CUDA_SAFE_CALL((colGpu = (char *)sycl::malloc_device(
                        colSize, dpct::get_default_queue()),
                    0));
    CUDA_SAFE_CALL(
        (dpct::get_default_queue().memcpy(colGpu, colCpu, colSize).wait(), 0));
    CUDA_SAFE_CALL((colTexture.attach(colGpu, colSize), 0));
    memoryUsedGpu += colSize;

    size_t hBusSize = colsGpu * sizeof(sycl::int2);
    sycl::int2 *hBusCpu = (sycl::int2 *)malloc(hBusSize);
    sycl::int2 *hBusGpu;
    for (int i = 0; i < colsGpu; ++i) {
        hBusCpu[i] = sycl::int2(-gapOpen - gapExtend * i, SCORE_MIN);
    }
    CUDA_SAFE_CALL((hBusGpu = (sycl::int2 *)sycl::malloc_device(
                        hBusSize, dpct::get_default_queue()),
                    0));
    CUDA_SAFE_CALL(
        (dpct::get_default_queue().memcpy(hBusGpu, hBusCpu, hBusSize).wait(),
         0));
    CUDA_SAFE_CALL((hBusTexture.attach(hBusGpu, hBusSize), 0));
    memoryUsedCpu += hBusSize;
    memoryUsedGpu += hBusSize;
    
    VBus vBusGpu;
    CUDA_SAFE_CALL((vBusGpu.mch = sycl::malloc_device<int>(
                        blocks * threads, dpct::get_default_queue()),
                    0));
    CUDA_SAFE_CALL((vBusGpu.scr = sycl::malloc_device<sycl::int4>(
                        blocks * threads, dpct::get_default_queue()),
                    0));
    CUDA_SAFE_CALL((vBusGpu.aff = sycl::malloc_device<sycl::int4>(
                        blocks * threads, dpct::get_default_queue()),
                    0));
    memoryUsedGpu += blocks * threads * sizeof(int);
    memoryUsedGpu += blocks * threads * sizeof(sycl::int4);
    memoryUsedGpu += blocks * threads * sizeof(sycl::int4);

    size_t subSize = subLen * subLen * sizeof(int);
    int* subCpu = (int*) malloc(subSize);
    int* subGpu;
    for (int i = 0; i < subLen; ++i) {
        for (int j = 0; j < subLen; ++j) {
            if (i < scorerLen && j < scorerLen) {
                subCpu[i * subLen + j] = scorerScore(scorer, i, j);
            } else {
                subCpu[i * subLen + j] = 0;
            }
        }
    }
    CUDA_SAFE_CALL((
        subGpu = (int *)sycl::malloc_device(subSize, dpct::get_default_queue()),
        0));
    CUDA_SAFE_CALL(
        (dpct::get_default_queue().memcpy(subGpu, subCpu, subSize).wait(), 0));
    CUDA_SAFE_CALL((subTexture.attach(subGpu, subSize), 0));
    memoryUsedCpu += subSize;
    memoryUsedGpu += subSize;

    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(match_.get_ptr(), &(subCpu[0]), sizeof(int))
                        .wait(),
                    0));
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(mismatch_.get_ptr(), &(subCpu[1]), sizeof(int))
                        .wait(),
                    0));
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(gapOpen_.get_ptr(), &gapOpen, sizeof(int))
                        .wait(),
                    0));
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(gapExtend_.get_ptr(), &gapExtend, sizeof(int))
                        .wait(),
                    0));
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(gapDiff_.get_ptr(), &gapDiff, sizeof(int))
                        .wait(),
                    0));
    CUDA_SAFE_CALL(
        (dpct::get_default_queue()
             .memcpy(queryFrontGap_.get_ptr(), &queryFrontGap, sizeof(int))
             .wait(),
         0));
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(scorerLen_.get_ptr(), &scorerLen, sizeof(int))
                        .wait(),
                    0));
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(subLen_.get_ptr(), &subLen, sizeof(int))
                        .wait(),
                    0));
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(rows_.get_ptr(), &rowsGpu, sizeof(int))
                        .wait(),
                    0));
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(cols_.get_ptr(), &colsGpu, sizeof(int))
                        .wait(),
                    0));
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(cellWidth_.get_ptr(), &cellWidth, sizeof(int))
                        .wait(),
                    0));
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(pLeft_.get_ptr(), &pLeft, sizeof(int))
                        .wait(),
                    0));
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(pRight_.get_ptr(), &pRight, sizeof(int))
                        .wait(),
                    0));
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(score_.get_ptr(), &score, sizeof(int))
                        .wait(),
                    0));

    /*
    LOG("Memory used CPU: %fMB", memoryUsedCpu / 1024. / 1024.);
    LOG("Memory used GPU: %fMB", memoryUsedGpu / 1024. / 1024.);
    */
    
    //**************************************************************************

    //**************************************************************************
    // KERNEL RUN
    
    // TIMER_START("Kernel");

    sycl::int2 res = {-1, -1};
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(res_.get_ptr(), &res, sizeof(sycl::int2))
                        .wait(),
                    0));

    for (int diagonal = 0; diagonal < diagonals; ++diagonal) {
    
        if (scalar) {
            if (subCpu[0] >= subCpu[1]) {
                /*
                DPCT1049:387: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                    extern dpct::constant_memory<int, 0> subLen_;

                    queryFrontGap_.init();
                    gapOpen_.init();
                    gapExtend_.init();
                    gapDiff_.init();
                    subLen_.init();

                    auto queryFrontGap__ptr_ct1 = queryFrontGap_.get_ptr();
                    auto gapOpen__ptr_ct1 = gapOpen_.get_ptr();
                    auto gapExtend__ptr_ct1 = gapExtend_.get_ptr();
                    auto gapDiff__ptr_ct1 = gapDiff_.get_ptr();
                    auto subLen__ptr_ct1 = subLen_.get_ptr();

                    sycl::accessor<int, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        hBusScrShr_acc_ct1(sycl::range<1>(128 /*MAX_THREADS*/),
                                           cgh);
                    sycl::accessor<int, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        hBusAffShr_acc_ct1(sycl::range<1>(128 /*MAX_THREADS*/),
                                           cgh);

                    auto subTexture_acc = subTexture.get_access(cgh);
                    auto rowTexture_acc = rowTexture.get_access(cgh);
                    auto colTexture_acc = colTexture.get_access(cgh);
                    auto hBusTexture_acc = hBusTexture.get_access(cgh);

                    auto subTexture_smpl = subTexture.get_sampler();
                    auto rowTexture_smpl = rowTexture.get_sampler();
                    auto colTexture_smpl = colTexture.get_sampler();
                    auto hBusTexture_smpl = hBusTexture.get_sampler();

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                              sycl::range<3>(1, 1, threads),
                                          sycl::range<3>(1, 1, threads)),
                        [=](sycl::nd_item<3> item_ct1) {
                            solveShort(diagonal, vBusGpu, hBusGpu, SubScalar(),
                                       item_ct1, *queryFrontGap__ptr_ct1,
                                       *gapOpen__ptr_ct1, *gapExtend__ptr_ct1,
                                       *gapDiff__ptr_ct1, *subLen__ptr_ct1,
                                       hBusScrShr_acc_ct1.get_pointer(),
                                       hBusAffShr_acc_ct1.get_pointer(),
                                       dpct::image_accessor_ext<int, 1>(
                                           subTexture_smpl, subTexture_acc),
                                       dpct::image_accessor_ext<sycl::char4, 1>(
                                           rowTexture_smpl, rowTexture_acc),
                                       dpct::image_accessor_ext<char, 1>(
                                           colTexture_smpl, colTexture_acc),
                                       dpct::image_accessor_ext<sycl::int2, 1>(
                                           hBusTexture_smpl, hBusTexture_acc));
                        });
                });
            } else {
                /*
                DPCT1049:388: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                    extern dpct::constant_memory<int, 0> subLen_;

                    queryFrontGap_.init();
                    gapOpen_.init();
                    gapExtend_.init();
                    gapDiff_.init();
                    subLen_.init();

                    auto queryFrontGap__ptr_ct1 = queryFrontGap_.get_ptr();
                    auto gapOpen__ptr_ct1 = gapOpen_.get_ptr();
                    auto gapExtend__ptr_ct1 = gapExtend_.get_ptr();
                    auto gapDiff__ptr_ct1 = gapDiff_.get_ptr();
                    auto subLen__ptr_ct1 = subLen_.get_ptr();

                    sycl::accessor<int, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        hBusScrShr_acc_ct1(sycl::range<1>(128 /*MAX_THREADS*/),
                                           cgh);
                    sycl::accessor<int, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        hBusAffShr_acc_ct1(sycl::range<1>(128 /*MAX_THREADS*/),
                                           cgh);

                    auto subTexture_acc = subTexture.get_access(cgh);
                    auto rowTexture_acc = rowTexture.get_access(cgh);
                    auto colTexture_acc = colTexture.get_access(cgh);
                    auto hBusTexture_acc = hBusTexture.get_access(cgh);

                    auto subTexture_smpl = subTexture.get_sampler();
                    auto rowTexture_smpl = rowTexture.get_sampler();
                    auto colTexture_smpl = colTexture.get_sampler();
                    auto hBusTexture_smpl = hBusTexture.get_sampler();

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                              sycl::range<3>(1, 1, threads),
                                          sycl::range<3>(1, 1, threads)),
                        [=](sycl::nd_item<3> item_ct1) {
                            solveShort(diagonal, vBusGpu, hBusGpu,
                                       SubScalarRev(), item_ct1,
                                       *queryFrontGap__ptr_ct1,
                                       *gapOpen__ptr_ct1, *gapExtend__ptr_ct1,
                                       *gapDiff__ptr_ct1, *subLen__ptr_ct1,
                                       hBusScrShr_acc_ct1.get_pointer(),
                                       hBusAffShr_acc_ct1.get_pointer(),
                                       dpct::image_accessor_ext<int, 1>(
                                           subTexture_smpl, subTexture_acc),
                                       dpct::image_accessor_ext<sycl::char4, 1>(
                                           rowTexture_smpl, rowTexture_acc),
                                       dpct::image_accessor_ext<char, 1>(
                                           colTexture_smpl, colTexture_acc),
                                       dpct::image_accessor_ext<sycl::int2, 1>(
                                           hBusTexture_smpl, hBusTexture_acc));
                        });
                });
            }
        } else {
            /*
            DPCT1049:389: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                extern dpct::constant_memory<int, 0> subLen_;

                queryFrontGap_.init();
                gapOpen_.init();
                gapExtend_.init();
                gapDiff_.init();
                subLen_.init();

                auto queryFrontGap__ptr_ct1 = queryFrontGap_.get_ptr();
                auto gapOpen__ptr_ct1 = gapOpen_.get_ptr();
                auto gapExtend__ptr_ct1 = gapExtend_.get_ptr();
                auto gapDiff__ptr_ct1 = gapDiff_.get_ptr();
                auto subLen__ptr_ct1 = subLen_.get_ptr();

                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    hBusScrShr_acc_ct1(sycl::range<1>(128 /*MAX_THREADS*/),
                                       cgh);
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    hBusAffShr_acc_ct1(sycl::range<1>(128 /*MAX_THREADS*/),
                                       cgh);

                auto subTexture_acc = subTexture.get_access(cgh);
                auto rowTexture_acc = rowTexture.get_access(cgh);
                auto colTexture_acc = colTexture.get_access(cgh);
                auto hBusTexture_acc = hBusTexture.get_access(cgh);

                auto subTexture_smpl = subTexture.get_sampler();
                auto rowTexture_smpl = rowTexture.get_sampler();
                auto colTexture_smpl = colTexture.get_sampler();
                auto hBusTexture_smpl = hBusTexture.get_sampler();

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                          sycl::range<3>(1, 1, threads),
                                      sycl::range<3>(1, 1, threads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        solveShort(
                            diagonal, vBusGpu, hBusGpu, SubVector(), item_ct1,
                            *queryFrontGap__ptr_ct1, *gapOpen__ptr_ct1,
                            *gapExtend__ptr_ct1, *gapDiff__ptr_ct1,
                            *subLen__ptr_ct1, hBusScrShr_acc_ct1.get_pointer(),
                            hBusAffShr_acc_ct1.get_pointer(),
                            dpct::image_accessor_ext<int, 1>(subTexture_smpl,
                                                             subTexture_acc),
                            dpct::image_accessor_ext<sycl::char4, 1>(
                                rowTexture_smpl, rowTexture_acc),
                            dpct::image_accessor_ext<char, 1>(colTexture_smpl,
                                                              colTexture_acc),
                            dpct::image_accessor_ext<sycl::int2, 1>(
                                hBusTexture_smpl, hBusTexture_acc));
                    });
            });
        }

        CUDA_SAFE_CALL((dpct::get_default_queue()
                            .memcpy(&res, res_.get_ptr(), sizeof(sycl::int2))
                            .wait(),
                        0));
        if (res.x() != -1) break;

        if (scalar) {
            if (subCpu[0] >= subCpu[1]) {
                /*
                DPCT1049:390: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                    extern dpct::constant_memory<int, 0> subLen_;

                    subLen_.init();

                    auto subLen__ptr_ct1 = subLen_.get_ptr();

                    sycl::accessor<int, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        hBusScrShr_acc_ct1(sycl::range<1>(128 /*MAX_THREADS*/),
                                           cgh);
                    sycl::accessor<int, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        hBusAffShr_acc_ct1(sycl::range<1>(128 /*MAX_THREADS*/),
                                           cgh);

                    auto subTexture_acc = subTexture.get_access(cgh);
                    auto rowTexture_acc = rowTexture.get_access(cgh);
                    auto colTexture_acc = colTexture.get_access(cgh);
                    auto hBusTexture_acc = hBusTexture.get_access(cgh);

                    auto subTexture_smpl = subTexture.get_sampler();
                    auto rowTexture_smpl = rowTexture.get_sampler();
                    auto colTexture_smpl = colTexture.get_sampler();
                    auto hBusTexture_smpl = hBusTexture.get_sampler();

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                              sycl::range<3>(1, 1, threads),
                                          sycl::range<3>(1, 1, threads)),
                        [=](sycl::nd_item<3> item_ct1) {
                            solveLong(diagonal, vBusGpu, hBusGpu, SubScalar(),
                                      item_ct1, *subLen__ptr_ct1,
                                      hBusScrShr_acc_ct1.get_pointer(),
                                      hBusAffShr_acc_ct1.get_pointer(),
                                      dpct::image_accessor_ext<int, 1>(
                                          subTexture_smpl, subTexture_acc),
                                      dpct::image_accessor_ext<sycl::char4, 1>(
                                          rowTexture_smpl, rowTexture_acc),
                                      dpct::image_accessor_ext<char, 1>(
                                          colTexture_smpl, colTexture_acc),
                                      dpct::image_accessor_ext<sycl::int2, 1>(
                                          hBusTexture_smpl, hBusTexture_acc));
                        });
                });
            } else {
                /*
                DPCT1049:391: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                    extern dpct::constant_memory<int, 0> subLen_;

                    subLen_.init();

                    auto subLen__ptr_ct1 = subLen_.get_ptr();

                    sycl::accessor<int, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        hBusScrShr_acc_ct1(sycl::range<1>(128 /*MAX_THREADS*/),
                                           cgh);
                    sycl::accessor<int, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        hBusAffShr_acc_ct1(sycl::range<1>(128 /*MAX_THREADS*/),
                                           cgh);

                    auto subTexture_acc = subTexture.get_access(cgh);
                    auto rowTexture_acc = rowTexture.get_access(cgh);
                    auto colTexture_acc = colTexture.get_access(cgh);
                    auto hBusTexture_acc = hBusTexture.get_access(cgh);

                    auto subTexture_smpl = subTexture.get_sampler();
                    auto rowTexture_smpl = rowTexture.get_sampler();
                    auto colTexture_smpl = colTexture.get_sampler();
                    auto hBusTexture_smpl = hBusTexture.get_sampler();

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                              sycl::range<3>(1, 1, threads),
                                          sycl::range<3>(1, 1, threads)),
                        [=](sycl::nd_item<3> item_ct1) {
                            solveLong(diagonal, vBusGpu, hBusGpu,
                                      SubScalarRev(), item_ct1,
                                      *subLen__ptr_ct1,
                                      hBusScrShr_acc_ct1.get_pointer(),
                                      hBusAffShr_acc_ct1.get_pointer(),
                                      dpct::image_accessor_ext<int, 1>(
                                          subTexture_smpl, subTexture_acc),
                                      dpct::image_accessor_ext<sycl::char4, 1>(
                                          rowTexture_smpl, rowTexture_acc),
                                      dpct::image_accessor_ext<char, 1>(
                                          colTexture_smpl, colTexture_acc),
                                      dpct::image_accessor_ext<sycl::int2, 1>(
                                          hBusTexture_smpl, hBusTexture_acc));
                        });
                });
            }
        } else {
            /*
            DPCT1049:392: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                extern dpct::constant_memory<int, 0> subLen_;

                subLen_.init();

                auto subLen__ptr_ct1 = subLen_.get_ptr();

                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    hBusScrShr_acc_ct1(sycl::range<1>(128 /*MAX_THREADS*/),
                                       cgh);
                sycl::accessor<int, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    hBusAffShr_acc_ct1(sycl::range<1>(128 /*MAX_THREADS*/),
                                       cgh);

                auto subTexture_acc = subTexture.get_access(cgh);
                auto rowTexture_acc = rowTexture.get_access(cgh);
                auto colTexture_acc = colTexture.get_access(cgh);
                auto hBusTexture_acc = hBusTexture.get_access(cgh);

                auto subTexture_smpl = subTexture.get_sampler();
                auto rowTexture_smpl = rowTexture.get_sampler();
                auto colTexture_smpl = colTexture.get_sampler();
                auto hBusTexture_smpl = hBusTexture.get_sampler();

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                          sycl::range<3>(1, 1, threads),
                                      sycl::range<3>(1, 1, threads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        solveLong(
                            diagonal, vBusGpu, hBusGpu, SubVector(), item_ct1,
                            *subLen__ptr_ct1, hBusScrShr_acc_ct1.get_pointer(),
                            hBusAffShr_acc_ct1.get_pointer(),
                            dpct::image_accessor_ext<int, 1>(subTexture_smpl,
                                                             subTexture_acc),
                            dpct::image_accessor_ext<sycl::char4, 1>(
                                rowTexture_smpl, rowTexture_acc),
                            dpct::image_accessor_ext<char, 1>(colTexture_smpl,
                                                              colTexture_acc),
                            dpct::image_accessor_ext<sycl::int2, 1>(
                                hBusTexture_smpl, hBusTexture_acc));
                    });
            });
        }

        CUDA_SAFE_CALL((dpct::get_default_queue()
                            .memcpy(&res, res_.get_ptr(), sizeof(sycl::int2))
                            .wait(),
                        0));
        if (res.x() != -1) break;
    }
    
    // TIMER_STOP;
    
    //**************************************************************************

    //**************************************************************************
    // SAVE RESULTS

    if (res.x() >= rows) {
        res.y() += rows - res.x() - 1;
        res.x() += rows - res.x() - 1;
    }

    if (res.y() >= cols) {
        res.x() += cols - res.y() - 1;
        res.y() += cols - res.y() - 1;
    }

    if (res.x() == -1) {
        LOG("Not found: %d", score);
    } else {
        LOG("Found: %d (%d, %d) (%d, %d)", score, res.x, res.y, rows, cols);
    }

    *queryStart = res.x();
    *targetStart = res.y();
    //**************************************************************************
    
    //**************************************************************************
    // CLEAN MEMORY

    free(subCpu);
    free(rowCpu);
    free(colCpu);
    free(hBusCpu);

    CUDA_SAFE_CALL((sycl::free(subGpu, dpct::get_default_queue()), 0));
    CUDA_SAFE_CALL((sycl::free(rowGpu, dpct::get_default_queue()), 0));
    CUDA_SAFE_CALL((sycl::free(colGpu, dpct::get_default_queue()), 0));
    CUDA_SAFE_CALL((sycl::free(vBusGpu.mch, dpct::get_default_queue()), 0));
    CUDA_SAFE_CALL((sycl::free(vBusGpu.scr, dpct::get_default_queue()), 0));
    CUDA_SAFE_CALL((sycl::free(vBusGpu.aff, dpct::get_default_queue()), 0));
    CUDA_SAFE_CALL((sycl::free(hBusGpu, dpct::get_default_queue()), 0));

    CUDA_SAFE_CALL((rowTexture.detach(), 0));
    CUDA_SAFE_CALL((colTexture.detach(), 0));
    CUDA_SAFE_CALL((hBusTexture.detach(), 0));
    CUDA_SAFE_CALL((subTexture.detach(), 0));

    free(params);   

    //**************************************************************************
    
    TIMER_STOP;
    
    return NULL;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

//------------------------------------------------------------------------------
//******************************************************************************

#endif // __CUDACC__

