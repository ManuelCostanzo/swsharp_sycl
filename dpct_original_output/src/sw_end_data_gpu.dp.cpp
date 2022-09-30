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

#include <algorithm>

#define MAX_THREADS MAX(THREADS_SM1, THREADS_SM2)

#define THREADS_SM1 64
#define BLOCKS_SM1  240

#define THREADS_SM2 128
#define BLOCKS_SM2  480

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
    int* mch;
    sycl::int4 *scr;
    sycl::int4 *aff;
} VBus;

typedef struct Context {
    int** scores;
    int** affines;
    int* queryEnd;
    int* targetEnd;
    int* outScore;
    Chain* query;
    Chain* target;
    Scorer* scorer;
    int score;
    int card;
} Context;

static dpct::constant_memory<int, 0> gapOpen_;
static dpct::constant_memory<int, 0> gapExtend_;

static dpct::constant_memory<int, 0> rows_;
static dpct::constant_memory<int, 0> cols_;

static dpct::constant_memory<int, 0> cellWidth_;

static dpct::constant_memory<int, 0> pruneLow_;
static dpct::constant_memory<int, 0> pruneHigh_;

static dpct::constant_memory<int, 0> scorerLen_;
static dpct::constant_memory<int, 0> subLen_;

static dpct::constant_memory<int, 0> match_;
static dpct::constant_memory<int, 0> mismatch_;

dpct::image_wrapper<sycl::char4, 1> rowTexture;
/*
DPCT1059:70: SYCL only supports 4-channel image format. Adjust the code.
*/
dpct::image_wrapper<char, 1> colTexture;
/*
DPCT1059:71: SYCL only supports 4-channel image format. Adjust the code.
*/
dpct::image_wrapper<sycl::int2, 1> hBusTexture;
/*
DPCT1059:72: SYCL only supports 4-channel image format. Adjust the code.
*/
dpct::image_wrapper<int, 1> subTexture;

//******************************************************************************
// PUBLIC

extern void swEndDataGpu(int* queryEnd, int* targetEnd, int* outScore, 
    int** scores, int** affines, Chain* query, Chain* target, Scorer* scorer, 
    int score, int card, Thread* thread);

//******************************************************************************

//******************************************************************************
// PRIVATE


// With visual c++ compiler and prototypes declared cuda global memory variables
// do not work. No questions asked.
#ifndef _WIN32

template <class Sub>
static void
solveShortDelegated(int d, VBus vBus, sycl::int2 *hBus, sycl::int3 *results,
                    Sub sub, sycl::nd_item<3> item_ct1, int subLen_,
                    int *hBusScrShr, int *hBusAffShr,
                    dpct::image_accessor_ext<sycl::char4, 1> rowTexture,
                    dpct::image_accessor_ext<char, 1> colTexture,
                    dpct::image_accessor_ext<int, 1> subTexture,
                    dpct::image_accessor_ext<sycl::int2, 1> hBusTexture);

template <class Sub>
static void
solveShortNormal(int d, VBus vBus, sycl::int2 *hBus, sycl::int3 *results,
                 Sub sub, sycl::nd_item<3> item_ct1, int subLen_,
                 int *hBusScrShr, int *hBusAffShr,
                 dpct::image_accessor_ext<sycl::char4, 1> rowTexture,
                 dpct::image_accessor_ext<char, 1> colTexture,
                 dpct::image_accessor_ext<int, 1> subTexture,
                 dpct::image_accessor_ext<sycl::int2, 1> hBusTexture);

template <class Sub>
static void solveShort(int d, VBus vBus, sycl::int2 *hBus, sycl::int3 *results,
                       Sub sub, sycl::nd_item<3> item_ct1, int subLen_,
                       int *hBusScrShr, int *hBusAffShr,
                       dpct::image_accessor_ext<sycl::char4, 1> rowTexture,
                       dpct::image_accessor_ext<char, 1> colTexture,
                       dpct::image_accessor_ext<int, 1> subTexture,
                       dpct::image_accessor_ext<sycl::int2, 1> hBusTexture);

template <class Sub>
static void solveLong(int d, VBus vBus, sycl::int2 *hBus, int *bBus,
                      sycl::int3 *results, Sub sub, sycl::nd_item<3> item_ct1,
                      int subLen_, int *hBusScrShr, int *hBusAffShr,
                      dpct::image_accessor_ext<sycl::char4, 1> rowTexture,
                      dpct::image_accessor_ext<char, 1> colTexture,
                      dpct::image_accessor_ext<int, 1> subTexture,
                      dpct::image_accessor_ext<sycl::int2, 1> hBusTexture);

#endif

static void* kernel(void* params);
    
//******************************************************************************

//******************************************************************************
// PUBLIC

extern void swEndDataGpu(int* queryEnd, int* targetEnd, int* outScore, 
    int** scores, int** affines, Chain* query, Chain* target, Scorer* scorer, 
    int score, int card, Thread* thread) {
    
    Context* param = (Context*) malloc(sizeof(Context));

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

template <class Sub>
static void
solveShortDelegated(int d, VBus vBus, sycl::int2 *hBus, sycl::int3 *results,
                    Sub sub, sycl::nd_item<3> item_ct1, int subLen_,
                    int *hBusScrShr, int *hBusAffShr,
                    dpct::image_accessor_ext<sycl::char4, 1> rowTexture,
                    dpct::image_accessor_ext<char, 1> colTexture,
                    dpct::image_accessor_ext<int, 1> subTexture,
                    dpct::image_accessor_ext<sycl::int2, 1> hBusTexture) {

    if (pruneLow_ >= 0 && pruneHigh_ < item_ct1.get_group_range(2)) {
        return;
    }

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
        atom.mch = 0;
        VEC4_ASSIGN(atom.lScr, INT4_ZERO);
        VEC4_ASSIGN(atom.lAff, INT4_ZERO);
    }

    hBusScrShr[item_ct1.get_local_id(2)] = hBusTexture.read(col).x();
    hBusAffShr[item_ct1.get_local_id(2)] = hBusTexture.read(col).y();

    sycl::char4 rowCodes = rowTexture.read(row >> 2);
    sycl::int3 res = {0, 0, 0};

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
            DPCT1084:75: The function call has multiple migration results in
            different template instantiations that could not be unified. You may
            need to adjust the code.
            */
            int mch = atom.mch + sub(columnCode, rowCodes.x(), subLen_, subTexture);

            atom.rScr.x() = MAX4(0, mch, del, ins);
            atom.rAff.x() = ins;

            del = sycl::max((int)(atom.rScr.x() - gapOpen_), (int)(del - gapExtend_));
            ins = sycl::max((int)(atom.lScr.y() - gapOpen_),
                            (int)(atom.lAff.y() - gapExtend_));
            /*
            DPCT1084:76: The function call has multiple migration results in
            different template instantiations that could not be unified. You may
            need to adjust the code.
            */
            mch = atom.lScr.x() + sub(columnCode, rowCodes.y(), subLen_, subTexture);

            atom.rScr.y() = MAX4(0, mch, del, ins);
            atom.rAff.y() = ins;

            del = sycl::max((int)(atom.rScr.y() - gapOpen_), (int)(del - gapExtend_));
            ins = sycl::max((int)(atom.lScr.z() - gapOpen_),
                            (int)(atom.lAff.z() - gapExtend_));
            /*
            DPCT1084:77: The function call has multiple migration results in
            different template instantiations that could not be unified. You may
            need to adjust the code.
            */
            mch = atom.lScr.y() + sub(columnCode, rowCodes.z(), subLen_, subTexture);

            atom.rScr.z() = MAX4(0, mch, del, ins);
            atom.rAff.z() = ins;

            del = sycl::max((int)(atom.rScr.z() - gapOpen_), (int)(del - gapExtend_));
            ins = sycl::max((int)(atom.lScr.w() - gapOpen_),
                            (int)(atom.lAff.w() - gapExtend_));
            /*
            DPCT1084:78: The function call has multiple migration results in
            different template instantiations that could not be unified. You may
            need to adjust the code.
            */
            mch = atom.lScr.z() + sub(columnCode, rowCodes.w(), subLen_, subTexture);

            atom.rScr.w() = MAX4(0, mch, del, ins);
            atom.rAff.w() = ins;

            if (atom.rScr.x() > res.x()) {
             res.x() = atom.rScr.x(); res.y() = row; res.z() = col;
            }
            if (atom.rScr.y() > res.x()) {
             res.x() = atom.rScr.y(); res.y() = row + 1; res.z() = col;
            }
            if (atom.rScr.z() > res.x()) {
             res.x() = atom.rScr.z(); res.y() = row + 2; res.z() = col;
            }
            if (atom.rScr.w() > res.x()) {
             res.x() = atom.rScr.w(); res.y() = row + 3; res.z() = col;
            }

            atom.mch = atom.up.x();
            VEC4_ASSIGN(atom.lScr, atom.rScr);
            VEC4_ASSIGN(atom.lAff, atom.rAff);
        }

        /*
        DPCT1065:73: Consider replacing sycl::nd_item::barrier() with
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

            atom.mch = 0;
            VEC4_ASSIGN(atom.lScr, INT4_ZERO);
            atom.lAff = atom.lScr;

            rowCodes = rowTexture.read(row >> 2);
        }

        /*
        DPCT1065:74: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    if (res.x() > results[item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                          item_ct1.get_local_id(2)]
                      .x()) {
        VEC3_ASSIGN(
            results[item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2)],
            res);
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
solveShortNormal(int d, VBus vBus, sycl::int2 *hBus, sycl::int3 *results,
                 Sub sub, sycl::nd_item<3> item_ct1, int subLen_,
                 int *hBusScrShr, int *hBusAffShr,
                 dpct::image_accessor_ext<sycl::char4, 1> rowTexture,
                 dpct::image_accessor_ext<char, 1> colTexture,
                 dpct::image_accessor_ext<int, 1> subTexture,
                 dpct::image_accessor_ext<sycl::int2, 1> hBusTexture) {

    if ((int)item_ct1.get_group(2) <= pruneLow_ ||
        item_ct1.get_group(2) >= pruneHigh_) {
        return;
    }

    int row = (d + item_ct1.get_group(2) - item_ct1.get_group_range(2) + 1) *
                  (item_ct1.get_local_range(2) * 4) +
              item_ct1.get_local_id(2) * 4;
    int col =
        cellWidth_ * (item_ct1.get_group_range(2) - item_ct1.get_group(2) - 1) -
        item_ct1.get_local_id(2);

    if (row < 0 || row >= rows_) return;
    
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
    sycl::int3 res = {0, 0, 0};

    int del;

    for (int i = 0; i < item_ct1.get_local_range(2); ++i, ++col) {

        char columnCode = colTexture.read(col);

        if (item_ct1.get_local_id(2) == 0) {
            atom.up = hBusTexture.read(col);
        } else {
            atom.up = sycl::int2(hBusScrShr[item_ct1.get_local_id(2)],
                                 hBusAffShr[item_ct1.get_local_id(2)]);
        }

        del = sycl::max((int)(atom.up.x() - gapOpen_),
                        (int)(atom.up.y() - gapExtend_));
        int ins = sycl::max((int)(atom.lScr.x() - gapOpen_),
                            (int)(atom.lAff.x() - gapExtend_));
        /*
        DPCT1084:81: The function call has multiple migration results in
        different template instantiations that could not be unified. You may
        need to adjust the code.
        */
        int mch = atom.mch + sub(columnCode, rowCodes.x(), subLen_, subTexture);

        atom.rScr.x() = MAX4(0, mch, del, ins);
        atom.rAff.x() = ins;

        del = sycl::max((int)(atom.rScr.x() - gapOpen_), (int)(del - gapExtend_));
        ins = sycl::max((int)(atom.lScr.y() - gapOpen_),
                        (int)(atom.lAff.y() - gapExtend_));
        /*
        DPCT1084:82: The function call has multiple migration results in
        different template instantiations that could not be unified. You may
        need to adjust the code.
        */
        mch = atom.lScr.x() + sub(columnCode, rowCodes.y(), subLen_, subTexture);

        atom.rScr.y() = MAX4(0, mch, del, ins);
        atom.rAff.y() = ins;

        del = sycl::max((int)(atom.rScr.y() - gapOpen_), (int)(del - gapExtend_));
        ins = sycl::max((int)(atom.lScr.z() - gapOpen_),
                        (int)(atom.lAff.z() - gapExtend_));
        /*
        DPCT1084:83: The function call has multiple migration results in
        different template instantiations that could not be unified. You may
        need to adjust the code.
        */
        mch = atom.lScr.y() + sub(columnCode, rowCodes.z(), subLen_, subTexture);

        atom.rScr.z() = MAX4(0, mch, del, ins);
        atom.rAff.z() = ins;

        del = sycl::max((int)(atom.rScr.z() - gapOpen_), (int)(del - gapExtend_));
        ins = sycl::max((int)(atom.lScr.w() - gapOpen_),
                        (int)(atom.lAff.w() - gapExtend_));
        /*
        DPCT1084:84: The function call has multiple migration results in
        different template instantiations that could not be unified. You may
        need to adjust the code.
        */
        mch = atom.lScr.z() + sub(columnCode, rowCodes.w(), subLen_, subTexture);

        atom.rScr.w() = MAX4(0, mch, del, ins);
        atom.rAff.w() = ins;

        if (atom.rScr.x() > res.x()) {
         res.x() = atom.rScr.x(); res.y() = row; res.z() = col;
        }
        if (atom.rScr.y() > res.x()) {
         res.x() = atom.rScr.y(); res.y() = row + 1; res.z() = col;
        }
        if (atom.rScr.z() > res.x()) {
         res.x() = atom.rScr.z(); res.y() = row + 2; res.z() = col;
        }
        if (atom.rScr.w() > res.x()) {
         res.x() = atom.rScr.w(); res.y() = row + 3; res.z() = col;
        }

        atom.mch = atom.up.x();
        VEC4_ASSIGN(atom.lScr, atom.rScr);
        VEC4_ASSIGN(atom.lAff, atom.rAff);

        /*
        DPCT1065:79: Consider replacing sycl::nd_item::barrier() with
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
        DPCT1065:80: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    const int vBusIdx = (row >> 2) % (item_ct1.get_group_range(2) *
                                      item_ct1.get_local_range(2));
    vBus.mch[vBusIdx] = atom.up.x();
    VEC4_ASSIGN(vBus.scr[vBusIdx], atom.lScr);
    VEC4_ASSIGN(vBus.aff[vBusIdx], atom.lAff);

    VEC2_ASSIGN(hBus[col - 1], sycl::int2(atom.rScr.w(), del));

    if (res.x() > results[item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                          item_ct1.get_local_id(2)]
                      .x()) {
        VEC3_ASSIGN(
            results[item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2)],
            res);
    }
}

template <class Sub>
static void solveShort(int d, VBus vBus, sycl::int2 *hBus, sycl::int3 *results,
                       Sub sub, sycl::nd_item<3> item_ct1, int subLen_,
                       int *hBusScrShr, int *hBusAffShr,
                       dpct::image_accessor_ext<sycl::char4, 1> rowTexture,
                       dpct::image_accessor_ext<char, 1> colTexture,
                       dpct::image_accessor_ext<int, 1> subTexture,
                       dpct::image_accessor_ext<sycl::int2, 1> hBusTexture) {

    if (item_ct1.get_group(2) == (item_ct1.get_group_range(2) - 1)) {
        solveShortDelegated(d, vBus, hBus, results, sub, item_ct1, subLen_,
                            hBusScrShr, hBusAffShr, rowTexture, colTexture,
                            subTexture, hBusTexture);
    } else {
        solveShortNormal(d, vBus, hBus, results, sub, item_ct1, subLen_,
                         hBusScrShr, hBusAffShr, rowTexture, colTexture,
                         subTexture, hBusTexture);
    }
}

template <class Sub>
static void solveLong(int d, VBus vBus, sycl::int2 *hBus, int *bBus,
                      sycl::int3 *results, Sub sub, sycl::nd_item<3> item_ct1,
                      int subLen_, int *hBusScrShr, int *hBusAffShr,
                      dpct::image_accessor_ext<sycl::char4, 1> rowTexture,
                      dpct::image_accessor_ext<char, 1> colTexture,
                      dpct::image_accessor_ext<int, 1> subTexture,
                      dpct::image_accessor_ext<sycl::int2, 1> hBusTexture) {

    hBusScrShr[item_ct1.get_local_id(2)] = 0;

    if ((int)item_ct1.get_group(2) <= pruneLow_ ||
        item_ct1.get_group(2) > pruneHigh_) {
        return;
    }

    int row = (d + item_ct1.get_group(2) - item_ct1.get_group_range(2) + 1) *
                  (item_ct1.get_local_range(2) * 4) +
              item_ct1.get_local_id(2) * 4;
    int col =
        cellWidth_ * (item_ct1.get_group_range(2) - item_ct1.get_group(2) - 1) -
        item_ct1.get_local_id(2) + item_ct1.get_local_range(2);

    if (row < 0 || row >= rows_) return;

    if (item_ct1.get_group(2) == pruneHigh_) {

        // clear only the last steepness
        vBus.mch[(row >> 2) % (item_ct1.get_group_range(2) *
                               item_ct1.get_local_range(2))] = 0;
        vBus.scr[(row >> 2) % (item_ct1.get_group_range(2) *
                               item_ct1.get_local_range(2))] = INT4_ZERO;
        vBus.aff[(row >> 2) % (item_ct1.get_group_range(2) *
                               item_ct1.get_local_range(2))] = INT4_ZERO;

        VEC2_ASSIGN(hBus[col + cellWidth_ - item_ct1.get_local_range(2) - 1],
                    sycl::int2(0, 0));

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
    sycl::int3 res = {0, 0, 0};

    int del;

    for (int i = 0; i < cellWidth_ - item_ct1.get_local_range(2); ++i, ++col) {

        char columnCode = colTexture.read(col);

        if (item_ct1.get_local_id(2) == 0) {
            atom.up = hBusTexture.read(col);
        } else {
            atom.up = sycl::int2(hBusScrShr[item_ct1.get_local_id(2)],
                                 hBusAffShr[item_ct1.get_local_id(2)]);
        }

        del = sycl::max((int)(atom.up.x() - gapOpen_),
                        (int)(atom.up.y() - gapExtend_));
        int ins = sycl::max((int)(atom.lScr.x() - gapOpen_),
                            (int)(atom.lAff.x() - gapExtend_));
        /*
        DPCT1084:88: The function call has multiple migration results in
        different template instantiations that could not be unified. You may
        need to adjust the code.
        */
        int mch = atom.mch + sub(columnCode, rowCodes.x(), subLen_, subTexture);

        atom.rScr.x() = MAX4(0, mch, del, ins);
        atom.rAff.x() = ins;

        del = sycl::max((int)(atom.rScr.x() - gapOpen_), (int)(del - gapExtend_));
        ins = sycl::max((int)(atom.lScr.y() - gapOpen_),
                        (int)(atom.lAff.y() - gapExtend_));
        /*
        DPCT1084:89: The function call has multiple migration results in
        different template instantiations that could not be unified. You may
        need to adjust the code.
        */
        mch = atom.lScr.x() + sub(columnCode, rowCodes.y(), subLen_, subTexture);

        atom.rScr.y() = MAX4(0, mch, del, ins);
        atom.rAff.y() = ins;

        del = sycl::max((int)(atom.rScr.y() - gapOpen_), (int)(del - gapExtend_));
        ins = sycl::max((int)(atom.lScr.z() - gapOpen_),
                        (int)(atom.lAff.z() - gapExtend_));
        /*
        DPCT1084:90: The function call has multiple migration results in
        different template instantiations that could not be unified. You may
        need to adjust the code.
        */
        mch = atom.lScr.y() + sub(columnCode, rowCodes.z(), subLen_, subTexture);

        atom.rScr.z() = MAX4(0, mch, del, ins);
        atom.rAff.z() = ins;

        del = sycl::max((int)(atom.rScr.z() - gapOpen_), (int)(del - gapExtend_));
        ins = sycl::max((int)(atom.lScr.w() - gapOpen_),
                        (int)(atom.lAff.w() - gapExtend_));
        /*
        DPCT1084:91: The function call has multiple migration results in
        different template instantiations that could not be unified. You may
        need to adjust the code.
        */
        mch = atom.lScr.z() + sub(columnCode, rowCodes.w(), subLen_, subTexture);

        atom.rScr.w() = MAX4(0, mch, del, ins);
        atom.rAff.w() = ins;

        if (atom.rScr.x() > res.x()) {
         res.x() = atom.rScr.x(); res.y() = row; res.z() = col;
        }
        if (atom.rScr.y() > res.x()) {
         res.x() = atom.rScr.y(); res.y() = row + 1; res.z() = col;
        }
        if (atom.rScr.z() > res.x()) {
         res.x() = atom.rScr.z(); res.y() = row + 2; res.z() = col;
        }
        if (atom.rScr.w() > res.x()) {
         res.x() = atom.rScr.w(); res.y() = row + 3; res.z() = col;
        }

        atom.mch = atom.up.x();
        VEC4_ASSIGN(atom.lScr, atom.rScr);
        VEC4_ASSIGN(atom.lAff, atom.rAff);

        /*
        DPCT1065:86: Consider replacing sycl::nd_item::barrier() with
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
        DPCT1065:87: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    const int vBusIdx = (row >> 2) % (item_ct1.get_group_range(2) *
                                      item_ct1.get_local_range(2));
    vBus.mch[vBusIdx] = atom.up.x();
    VEC4_ASSIGN(vBus.scr[vBusIdx], atom.lScr);
    VEC4_ASSIGN(vBus.aff[vBusIdx], atom.lAff);

    VEC2_ASSIGN(hBus[col - 1], sycl::int2(atom.rScr.w(), del));

    if (res.x() > results[item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                          item_ct1.get_local_id(2)]
                      .x()) {
        VEC3_ASSIGN(
            results[item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2)],
            res);
    }

    // reuse
    hBusScrShr[item_ct1.get_local_id(2)] = res.x();
    /*
    DPCT1065:85: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    int score = 0;
    int idx = 0;

    for (int i = 0; i < item_ct1.get_local_range(2); ++i) {

        int shr = hBusScrShr[i];
        
        if (shr > score) {
            score = shr;
            idx = i;
        }
    }

    if (item_ct1.get_local_id(2) == idx) bBus[item_ct1.get_group(2)] = score;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// CPU KERNELS

static void *kernel(void *params) try {

    Context* context = (Context*) params;
    
    int** scores = context->scores;
    int** affines = context->affines;
    int* queryEnd = context->queryEnd;
    int* targetEnd = context->targetEnd;
    int* outScore = context->outScore;
    Chain* query = context->query;
    Chain* target = context->target;
    Scorer* scorer = context->scorer;
    int score = context->score;
    int card = context->card;

    // if negative matrix, no need for SW, score will not be found
    if (scorerGetMaxScore(scorer) <= 0) {
        *outScore = NO_SCORE;
        *queryEnd = 0;
        *targetEnd = 0;
        if (scores != NULL) *scores = NULL;
        if (affines != NULL) *affines = NULL;
        free(params);
        return NULL;
    }

    int currentCard;
    CUDA_SAFE_CALL(currentCard = dpct::dev_mgr::instance().current_device_id());
    if (currentCard != card) {
        // CUDA_SAFE_CALL(cudaThreadExit());
        /*
        DPCT1093:92: The "card" may not be the best XPU device. Adjust the
        selected device if needed.
        */
        /*
        DPCT1003:93: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_SAFE_CALL((dpct::dev_mgr::instance().select_device(card), 0));
    }
    
    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);
    int scorerLen = scorerGetMaxCode(scorer);
    int subLen = scorerLen + 1;
    int scalar = scorerIsScalar(scorer);
    
    TIMER_START("Sw end data %d %d", rows, cols);

    dpct::device_info properties;
    CUDA_SAFE_CALL(
        (dpct::dev_mgr::instance().get_device(card).get_device_info(properties),
         0));

    int threads;
    int blocks;
    /*
    DPCT1005:94: The SYCL device version is different from CUDA Compute
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
        // LOG("Blocks trimmed to: %d", blocks);
    }
    
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
    char* rowCpu = (char*) malloc(rowsGpu * sizeof(char));
    memset(rowCpu, scorerLen, (rowsGpu - rows) * sizeof(char));
    chainCopyCodes(query, rowCpu + (rowsGpu - rows));
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
    /*
    DPCT1003:95: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((rowGpu = (sycl::char4 *)sycl::malloc_device(
                        rowSize, dpct::get_default_queue()),
                    0));
    /*
    DPCT1003:96: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL(
        (dpct::get_default_queue().memcpy(rowGpu, rowCpu, rowSize).wait(), 0));
    /*
    DPCT1003:97: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((rowTexture.attach(rowGpu, rowSize), 0));
    memoryUsedGpu += rowSize;

    size_t colSize = colsGpu * sizeof(char);
    char* colGpu;
    /*
    DPCT1003:98: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((colGpu = (char *)sycl::malloc_device(
                        colSize, dpct::get_default_queue()),
                    0));
    /*
    DPCT1003:99: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL(
        (dpct::get_default_queue().memcpy(colGpu, colCpu, colSize).wait(), 0));
    /*
    DPCT1003:100: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((colTexture.attach(colGpu, colSize), 0));
    memoryUsedGpu += colSize;

    size_t hBusSize = colsGpu * sizeof(sycl::int2);
    sycl::int2 *hBusCpu;
    sycl::int2 *hBusGpu;
    /*
    DPCT1003:101: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((hBusCpu = (sycl::int2 *)sycl::malloc_host(
                        hBusSize, dpct::get_default_queue()),
                    0));
    /*
    DPCT1003:102: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((hBusGpu = (sycl::int2 *)sycl::malloc_device(
                        hBusSize, dpct::get_default_queue()),
                    0));
    /*
    DPCT1003:103: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL(
        (dpct::get_default_queue().memset(hBusGpu, 0, hBusSize).wait(), 0));
    /*
    DPCT1003:104: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((hBusTexture.attach(hBusGpu, hBusSize), 0));
    memoryUsedCpu += hBusSize;
    memoryUsedGpu += hBusSize;
    
    VBus vBusGpu;
    /*
    DPCT1003:105: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((vBusGpu.mch = sycl::malloc_device<int>(
                        blocks * threads, dpct::get_default_queue()),
                    0));
    /*
    DPCT1003:106: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((vBusGpu.scr = sycl::malloc_device<sycl::int4>(
                        blocks * threads, dpct::get_default_queue()),
                    0));
    /*
    DPCT1003:107: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((vBusGpu.aff = sycl::malloc_device<sycl::int4>(
                        blocks * threads, dpct::get_default_queue()),
                    0));
    memoryUsedGpu += blocks * threads * sizeof(int);
    memoryUsedGpu += blocks * threads * sizeof(sycl::int4);
    memoryUsedGpu += blocks * threads * sizeof(sycl::int4);

    size_t resultsSize = blocks * threads * sizeof(sycl::int3);
    sycl::int3 *resultsCpu = (sycl::int3 *)malloc(resultsSize);
    sycl::int3 *resultsGpu;
    /*
    DPCT1003:108: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((resultsGpu = (sycl::int3 *)sycl::malloc_device(
                        resultsSize, dpct::get_default_queue()),
                    0));
    /*
    DPCT1003:109: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL(
        (dpct::get_default_queue().memset(resultsGpu, 0, resultsSize).wait(),
         0));
    memoryUsedCpu += resultsSize;
    memoryUsedGpu += resultsSize;
    
    size_t bSize = blocks * sizeof(int);
    int* bCpu;
    int* bGpu;
    /*
    DPCT1003:110: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL(
        (bCpu = (int *)sycl::malloc_host(bSize, dpct::get_default_queue()), 0));
    /*
    DPCT1003:111: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL(
        (bGpu = (int *)sycl::malloc_device(bSize, dpct::get_default_queue()),
         0));
    /*
    DPCT1003:112: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((dpct::get_default_queue().memset(bGpu, 0, bSize).wait(), 0));
    memoryUsedCpu += bSize;
    memoryUsedGpu += bSize;
    
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
    /*
    DPCT1003:113: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(gapOpen_.get_ptr(), &gapOpen, sizeof(int))
                        .wait(),
                    0));
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(gapExtend_.get_ptr(), &gapExtend, sizeof(int))
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
    /*
    DPCT1003:114: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(cols_.get_ptr(), &colsGpu, sizeof(int))
                        .wait(),
                    0));
    /*
    DPCT1003:115: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(cellWidth_.get_ptr(), &cellWidth, sizeof(int))
                        .wait(),
                    0));
    /*
    DPCT1003:116: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(pruneLow_.get_ptr(), &pruneLow, sizeof(int))
                        .wait(),
                    0));
    /*
    DPCT1003:117: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(pruneHigh_.get_ptr(), &pruneHigh, sizeof(int))
                        .wait(),
                    0));

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
            if (subCpu[0] >= subCpu[1]) {
                /*
                DPCT1049:118: The work-group size passed to the SYCL kernel may
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

                    auto rowTexture_acc = rowTexture.get_access(cgh);
                    auto colTexture_acc = colTexture.get_access(cgh);
                    auto subTexture_acc = subTexture.get_access(cgh);
                    auto hBusTexture_acc = hBusTexture.get_access(cgh);

                    auto rowTexture_smpl = rowTexture.get_sampler();
                    auto colTexture_smpl = colTexture.get_sampler();
                    auto subTexture_smpl = subTexture.get_sampler();
                    auto hBusTexture_smpl = hBusTexture.get_sampler();

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                              sycl::range<3>(1, 1, threads),
                                          sycl::range<3>(1, 1, threads)),
                        [=](sycl::nd_item<3> item_ct1) {
                            solveShort(diagonal, vBusGpu, hBusGpu, resultsGpu,
                                       SubScalar(), item_ct1, *subLen__ptr_ct1,
                                       hBusScrShr_acc_ct1.get_pointer(),
                                       hBusAffShr_acc_ct1.get_pointer(),
                                       dpct::image_accessor_ext<sycl::char4, 1>(
                                           rowTexture_smpl, rowTexture_acc),
                                       dpct::image_accessor_ext<char, 1>(
                                           colTexture_smpl, colTexture_acc),
                                       dpct::image_accessor_ext<int, 1>(
                                           subTexture_smpl, subTexture_acc),
                                       dpct::image_accessor_ext<sycl::int2, 1>(
                                           hBusTexture_smpl, hBusTexture_acc));
                        });
                });
                /*
                DPCT1049:119: The work-group size passed to the SYCL kernel may
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

                    auto rowTexture_acc = rowTexture.get_access(cgh);
                    auto colTexture_acc = colTexture.get_access(cgh);
                    auto subTexture_acc = subTexture.get_access(cgh);
                    auto hBusTexture_acc = hBusTexture.get_access(cgh);

                    auto rowTexture_smpl = rowTexture.get_sampler();
                    auto colTexture_smpl = colTexture.get_sampler();
                    auto subTexture_smpl = subTexture.get_sampler();
                    auto hBusTexture_smpl = hBusTexture.get_sampler();

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                              sycl::range<3>(1, 1, threads),
                                          sycl::range<3>(1, 1, threads)),
                        [=](sycl::nd_item<3> item_ct1) {
                            solveLong(diagonal, vBusGpu, hBusGpu, bGpu,
                                      resultsGpu, SubScalar(), item_ct1,
                                      *subLen__ptr_ct1,
                                      hBusScrShr_acc_ct1.get_pointer(),
                                      hBusAffShr_acc_ct1.get_pointer(),
                                      dpct::image_accessor_ext<sycl::char4, 1>(
                                          rowTexture_smpl, rowTexture_acc),
                                      dpct::image_accessor_ext<char, 1>(
                                          colTexture_smpl, colTexture_acc),
                                      dpct::image_accessor_ext<int, 1>(
                                          subTexture_smpl, subTexture_acc),
                                      dpct::image_accessor_ext<sycl::int2, 1>(
                                          hBusTexture_smpl, hBusTexture_acc));
                        });
                });
            } else {
                // cannot use mismatch negative trick
                /*
                DPCT1049:120: The work-group size passed to the SYCL kernel may
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

                    auto rowTexture_acc = rowTexture.get_access(cgh);
                    auto colTexture_acc = colTexture.get_access(cgh);
                    auto subTexture_acc = subTexture.get_access(cgh);
                    auto hBusTexture_acc = hBusTexture.get_access(cgh);

                    auto rowTexture_smpl = rowTexture.get_sampler();
                    auto colTexture_smpl = colTexture.get_sampler();
                    auto subTexture_smpl = subTexture.get_sampler();
                    auto hBusTexture_smpl = hBusTexture.get_sampler();

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                              sycl::range<3>(1, 1, threads),
                                          sycl::range<3>(1, 1, threads)),
                        [=](sycl::nd_item<3> item_ct1) {
                            solveShort(diagonal, vBusGpu, hBusGpu, resultsGpu,
                                       SubScalarRev(), item_ct1,
                                       *subLen__ptr_ct1,
                                       hBusScrShr_acc_ct1.get_pointer(),
                                       hBusAffShr_acc_ct1.get_pointer(),
                                       dpct::image_accessor_ext<sycl::char4, 1>(
                                           rowTexture_smpl, rowTexture_acc),
                                       dpct::image_accessor_ext<char, 1>(
                                           colTexture_smpl, colTexture_acc),
                                       dpct::image_accessor_ext<int, 1>(
                                           subTexture_smpl, subTexture_acc),
                                       dpct::image_accessor_ext<sycl::int2, 1>(
                                           hBusTexture_smpl, hBusTexture_acc));
                        });
                });
                /*
                DPCT1049:121: The work-group size passed to the SYCL kernel may
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

                    auto rowTexture_acc = rowTexture.get_access(cgh);
                    auto colTexture_acc = colTexture.get_access(cgh);
                    auto subTexture_acc = subTexture.get_access(cgh);
                    auto hBusTexture_acc = hBusTexture.get_access(cgh);

                    auto rowTexture_smpl = rowTexture.get_sampler();
                    auto colTexture_smpl = colTexture.get_sampler();
                    auto subTexture_smpl = subTexture.get_sampler();
                    auto hBusTexture_smpl = hBusTexture.get_sampler();

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                              sycl::range<3>(1, 1, threads),
                                          sycl::range<3>(1, 1, threads)),
                        [=](sycl::nd_item<3> item_ct1) {
                            solveLong(diagonal, vBusGpu, hBusGpu, bGpu,
                                      resultsGpu, SubScalarRev(), item_ct1,
                                      *subLen__ptr_ct1,
                                      hBusScrShr_acc_ct1.get_pointer(),
                                      hBusAffShr_acc_ct1.get_pointer(),
                                      dpct::image_accessor_ext<sycl::char4, 1>(
                                          rowTexture_smpl, rowTexture_acc),
                                      dpct::image_accessor_ext<char, 1>(
                                          colTexture_smpl, colTexture_acc),
                                      dpct::image_accessor_ext<int, 1>(
                                          subTexture_smpl, subTexture_acc),
                                      dpct::image_accessor_ext<sycl::int2, 1>(
                                          hBusTexture_smpl, hBusTexture_acc));
                        });
                });
            }
        } else {
            /*
            DPCT1049:122: The work-group size passed to the SYCL kernel may
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

                auto rowTexture_acc = rowTexture.get_access(cgh);
                auto colTexture_acc = colTexture.get_access(cgh);
                auto subTexture_acc = subTexture.get_access(cgh);
                auto hBusTexture_acc = hBusTexture.get_access(cgh);

                auto rowTexture_smpl = rowTexture.get_sampler();
                auto colTexture_smpl = colTexture.get_sampler();
                auto subTexture_smpl = subTexture.get_sampler();
                auto hBusTexture_smpl = hBusTexture.get_sampler();

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                          sycl::range<3>(1, 1, threads),
                                      sycl::range<3>(1, 1, threads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        solveShort(diagonal, vBusGpu, hBusGpu, resultsGpu,
                                   SubVector(), item_ct1, *subLen__ptr_ct1,
                                   hBusScrShr_acc_ct1.get_pointer(),
                                   hBusAffShr_acc_ct1.get_pointer(),
                                   dpct::image_accessor_ext<sycl::char4, 1>(
                                       rowTexture_smpl, rowTexture_acc),
                                   dpct::image_accessor_ext<char, 1>(
                                       colTexture_smpl, colTexture_acc),
                                   dpct::image_accessor_ext<int, 1>(
                                       subTexture_smpl, subTexture_acc),
                                   dpct::image_accessor_ext<sycl::int2, 1>(
                                       hBusTexture_smpl, hBusTexture_acc));
                    });
            });
            /*
            DPCT1049:123: The work-group size passed to the SYCL kernel may
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

                auto rowTexture_acc = rowTexture.get_access(cgh);
                auto colTexture_acc = colTexture.get_access(cgh);
                auto subTexture_acc = subTexture.get_access(cgh);
                auto hBusTexture_acc = hBusTexture.get_access(cgh);

                auto rowTexture_smpl = rowTexture.get_sampler();
                auto colTexture_smpl = colTexture.get_sampler();
                auto subTexture_smpl = subTexture.get_sampler();
                auto hBusTexture_smpl = hBusTexture.get_sampler();

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                          sycl::range<3>(1, 1, threads),
                                      sycl::range<3>(1, 1, threads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        solveLong(diagonal, vBusGpu, hBusGpu, bGpu, resultsGpu,
                                  SubVector(), item_ct1, *subLen__ptr_ct1,
                                  hBusScrShr_acc_ct1.get_pointer(),
                                  hBusAffShr_acc_ct1.get_pointer(),
                                  dpct::image_accessor_ext<sycl::char4, 1>(
                                      rowTexture_smpl, rowTexture_acc),
                                  dpct::image_accessor_ext<char, 1>(
                                      colTexture_smpl, colTexture_acc),
                                  dpct::image_accessor_ext<int, 1>(
                                      subTexture_smpl, subTexture_acc),
                                  dpct::image_accessor_ext<sycl::int2, 1>(
                                      hBusTexture_smpl, hBusTexture_acc));
                    });
            });
        }

        if (pruning) {
        
            size_t bSize = pruneHigh * sizeof(int);
            /*
            DPCT1003:124: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            CUDA_SAFE_CALL((
                dpct::get_default_queue().memcpy(bCpu, bGpu, bSize).wait(), 0));

            if (score == NO_SCORE) {
                for (int i = 0; i < pruneHigh; ++i) {
                    best = std::max(best, bCpu[i]);
                }
            }

            // delta j pruning
            pruneLow = -1;
            for (int i = 0; i < blocks; ++i) {
                int row = (diagonal + 1 + i - blocks + 1) * (threads * 4);
                int col = cellWidth * (blocks - i - 1) - threads;
                if (row >= rowsGpu) break;
                if (rowsGpu * (halfPruning ? 2 : 1) - row < cols - col) break;
                int d = cols - col;
                int scr = i == blocks - 1 ? bCpu[i] : std::max(bCpu[i], bCpu[i + 1]);
                if ((scr + d * pruneFactor) < best) pruneLow = i;
                else break;
            }

            // delta i pruning
            if (!halfPruning) {
                pruneHighOld = pruneHigh;
                for (int i = pruneHighOld - 1; i >= 0; --i) {
                    int row = (diagonal + 1 + i - blocks + 1) * (threads * 4);
                    int col = cellWidth * (blocks - i - 1) - threads;
                    if (row < rowsGpu / 2) break;
                    if (row >= rowsGpu) continue;
                    if (rowsGpu - row > cols - col) break;
                    int d = rowsGpu - row;
                    int scr1 = d * pruneFactor + (i == blocks - 1 ? 0 : bCpu[i + 1]);
                    int scr2 = (d + threads * 2) * pruneFactor + bCpu[i];
                    if (scr1 < best && scr2 < best) pruneHigh = i; 
                    else break;
                }
            }

            pruned += blocks - (pruneHigh - pruneLow - 1);
            
            if (pruneLow >= pruneHigh) {
                break;
            }

            /*
            DPCT1003:125: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            CUDA_SAFE_CALL(
                (dpct::get_default_queue()
                     .memcpy(pruneLow_.get_ptr(), &pruneLow, sizeof(int))
                     .wait(),
                 0));
            /*
            DPCT1003:126: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            CUDA_SAFE_CALL(
                (dpct::get_default_queue()
                     .memcpy(pruneHigh_.get_ptr(), &pruneHigh, sizeof(int))
                     .wait(),
                 0));

            if (pruneLow >= 0) {
                int offset = (blocks - pruneLow - 1) * cellWidth - threads;
                size_t size = (colsGpu - offset) * sizeof(sycl::int2);
                /*
                DPCT1003:127: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                CUDA_SAFE_CALL((dpct::get_default_queue()
                                    .memset(hBusGpu + offset, 0, size)
                                    .wait(),
                                0));
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

        /*
        DPCT1003:128: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CUDA_SAFE_CALL((
            dpct::get_default_queue().memcpy(hBusCpu, hBusGpu, hBusSize).wait(),
            0));

        *scores = (int*) malloc(cols * sizeof(int));
        *affines = (int*) malloc(cols * sizeof(int));
        
        for (int i = 0; i < cols; ++i) {
            (*scores)[i] = hBusCpu[i].x();
            (*affines)[i] = hBusCpu[i].y();
        }
    }

    /*
    DPCT1003:129: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((dpct::get_default_queue()
                        .memcpy(resultsCpu, resultsGpu, resultsSize)
                        .wait(),
                    0));

    sycl::int3 res = resultsCpu[0];
    for (int i = 1; i < blocks * threads; ++i) {
        if (resultsCpu[i].x() > res.x()) {
            res = resultsCpu[i];
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

    free(subCpu);
    free(rowCpu);
    free(colCpu);
    free(resultsCpu);

    /*
    DPCT1003:130: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((sycl::free(bCpu, dpct::get_default_queue()), 0));
    /*
    DPCT1003:131: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((sycl::free(hBusCpu, dpct::get_default_queue()), 0));

    /*
    DPCT1003:132: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((sycl::free(subGpu, dpct::get_default_queue()), 0));
    /*
    DPCT1003:133: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((sycl::free(rowGpu, dpct::get_default_queue()), 0));
    /*
    DPCT1003:134: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((sycl::free(colGpu, dpct::get_default_queue()), 0));
    /*
    DPCT1003:135: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((sycl::free(vBusGpu.mch, dpct::get_default_queue()), 0));
    /*
    DPCT1003:136: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((sycl::free(vBusGpu.scr, dpct::get_default_queue()), 0));
    /*
    DPCT1003:137: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((sycl::free(vBusGpu.aff, dpct::get_default_queue()), 0));
    /*
    DPCT1003:138: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((sycl::free(hBusGpu, dpct::get_default_queue()), 0));
    /*
    DPCT1003:139: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((sycl::free(resultsGpu, dpct::get_default_queue()), 0));
    /*
    DPCT1003:140: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((sycl::free(bGpu, dpct::get_default_queue()), 0));

    /*
    DPCT1003:141: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((rowTexture.detach(), 0));
    /*
    DPCT1003:142: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((colTexture.detach(), 0));
    /*
    DPCT1003:143: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((hBusTexture.detach(), 0));
    /*
    DPCT1003:144: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
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

