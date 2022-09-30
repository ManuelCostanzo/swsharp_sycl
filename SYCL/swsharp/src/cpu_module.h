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

@brief CPU implementations of common functions.
*/

#ifndef __SW_SHARP_CPU_MODULEH__
#define __SW_SHARP_CPU_MODULEH__

#include "alignment.h"
#include "chain.h"
#include "scorer.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /*!
    @brief Pairwise alignment function.

    Function aligns query and the target chain with the scorer object.

    @param alignment output alignment object
    @param query query chain
    @param target target chain
    @param scorer scorer object used for alignment
    @param type aligning type, can be #SW_ALIGN, #NW_ALIGN, #HW_ALIGN or #OV_ALIGN
    */
    extern void alignPairCpu(Alignment **alignment, int type, Chain *query,
                             Chain *target, Scorer *scorer);

    /*!
    @brief Pairwise alignment function.

    Function aligns previously score query and the target chain with the scorer
    object. If the score isn't valid for the produced alignment an error will occur.
    Function is primaraly provided to get alignments after calling #scorePairCpu
    function.

    @param alignment output alignment object
    @param type aligning type, can be #SW_ALIGN, #NW_ALIGN, #HW_ALIGN or #OV_ALIGN
    @param query query chain
    @param target target chain
    @param scorer scorer object used for alignment
    @param score alignment score
    */
    extern void alignScoredPairCpu(Alignment **alignment, int type, Chain *query,
                                   Chain *target, Scorer *scorer, int score);

    /*!
    @brief Score finding function.

    Method uses Needleman-Wunsch algorithm with all of the start conditions set to
    infinity. This assures path contains the first cell and does not start with gaps.
    If the score is found it return the coordinates of the cell with the provided
    score, (-1, -1) otherwise.

    @param queryStart output, if found query index of found cell, -1 otherwise
    @param targetStart output, if found target index of found cell, -1 otherwise
    @param query query chain
    @param queryFrontGap indicates that query starts with a gap
    @param target target chain
    @param scorer scorer object used for alignment
    @param score input alignment score
    */
    extern void nwFindScoreCpu(int *queryStart, int *targetStart, Chain *query,
                               int queryFrontGap, Chain *target, Scorer *scorer, int score);

    /*!
    @brief Needleman-Wunsch reconstruction implementation.

    If the score is provided function uses Ukkonen's banded optimization.
    QueryFrontGap and targetFrontGap arguments can't both be not equal to 0.
    QueryBackGap and targetBackGap arguments can't both be not equal to 0.
    For path format see ::Alignment.

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
    */
    extern void nwReconstructCpu(char **path, int *pathLen, int *outScore,
                                 Chain *query, int queryFrontGap, int queryBackGap, Chain *target,
                                 int targetFrontGap, int targetBackGap, Scorer *scorer, int score);

    /*!
    @brief Implementation of score finding function.

    Method uses Needleman-Wunsch algorithm. If the score is found and the indicies
    of the coresponding cell are on the border of the solving matrix, functions
    returns the coordinates of the cell with the provided score, (-1, -1) otherwise.

    @param queryStart output, if found query index of found cell, -1 otherwise
    @param targetStart output, if found target index of found cell, -1 otherwise
    @param query query chain
    @param target target chain
    @param scorer scorer object used for alignment
    @param score input alignment score
    */
    extern void ovFindScoreCpu(int *queryStart, int *targetStart, Chain *query,
                               Chain *target, Scorer *scorer, int score);

    /*!
    @brief Pairwise scoring function.

    Function provides only the alignment score without any other information.
    Scoring types are equivalent to aligning types.

    @param type scoring type, can be #SW_ALIGN, #NW_ALIGN, #HW_ALIGN or #OV_ALIGN
    @param query query chain
    @param target target chain
    @param scorer scorer object used for alignment
    */
    extern int scorePairCpu(int type, Chain *query, Chain *target, Scorer *scorer);

    extern void scoreDatabaseCpu(int *scores, int type, Chain *query,
                                 Chain **database, int databaseLen, Scorer *scorer);

    extern void scoreDatabasePartiallyCpu(int *scores, int type, Chain *query,
                                          Chain **database, int databaseLen, Scorer *scorer, int maxScore);

#ifdef __cplusplus
}
#endif
#endif // __SW_SHARP_CPU_MODULEH__
