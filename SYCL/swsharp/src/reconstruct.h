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
/**
@file

@brief Reconstruction functions header.
*/

#ifndef __SW_SHARP_RECONSTRUCTH__
#define __SW_SHARP_RECONSTRUCTH__

#include "chain.h"
#include "scorer.h"
#include "thread.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /*!
    @brief Needleman-Wunsch reconstruction implementation.

    If the score is provided function uses Ukkonen's banded optimization. Function
    also utilizes Hirschberg's algorithm and therefore is linear in memory.
    QueryFrontGap and targetFrontGap arguments can't both be not equal to 0.
    QueryBackGap and targetBackGap arguments can't both be not equal to 0.
    For path format see ::Alignment.
    If needed function utilzes provided CUDA cards.

    @param path output path
    @param pathLen output path length
    @param outScore output score
    @param query query chain
    @param queryFrontGap if not equal to 0, force that path starts in #MOVE_UP
    @param queryBackGap if not equal to 0, force that path ends in #MOVE_UP
    @param target target chain
    @param targetFrontGap if not equal to 0, force that path starts in #MOVE_LEFT
    @param targetBackGap if not equal to 0, force that path ends in #MOVE_LEFT
    @param scorer scorer object used for alignment
    @param score input alignment score if known, otherwise #NO_SCORE
    @param cards cuda cards index array
    @param cardsLen cuda cards index array length
    @param thread thread on which the function will be executed, if NULL function is
        executed on the current thread
    */
    extern void nwReconstruct(char **path, int *pathLen, int *outScore,
                              Chain *query, int queryFrontGap, int queryBackGap, Chain *target,
                              int targetFrontGap, int targetBackGap, Scorer *scorer, int score,
                              int *cards, int cardsLen, Thread *thread);

#ifdef __cplusplus
}
#endif
#endif // __SW_SHARP_RECONSTRUCTH__
