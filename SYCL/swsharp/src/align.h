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

@brief Pairwise alignment oriented functions header.
*/

#ifndef __SW_SHARP_ALIGNH__
#define __SW_SHARP_ALIGNH__

#include "alignment.h"
#include "chain.h"
#include "scorer.h"
#include "thread.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /*!
    @brief Pairwise alignment function.

    Function aligns query and the target chain with the scorer object.
    If needed function utilzes provided CUDA cards.

    @param alignment output alignment object
    @param type aligning type, can be #SW_ALIGN, #NW_ALIGN, #HW_ALIGN or #OV_ALIGN
    @param query query chain
    @param target target chain
    @param scorer scorer object used for alignment
    @param cards cuda cards index array
    @param cardsLen cuda cards index array length
    @param thread thread on which the function will be executed, if NULL function is
        executed on the current thread
    */
    extern void alignPair(Alignment **alignment, int type, Chain *query,
                          Chain *target, Scorer *scorer, int *cards, int cardsLen, Thread *thread);

    /*!
    @brief Pairwise alignment function.

    Function aligns previously score query and the target chain with the scorer
    object. If the score isn't valid for the produced alignment an error will occur.
    Function is primaraly provided to get alignments after calling #scorePair
    function. If needed function utilzes provided CUDA cards.

    @param alignment output alignment object
    @param type aligning type, can be #SW_ALIGN, #NW_ALIGN, #HW_ALIGN or #OV_ALIGN
    @param query query chain
    @param target target chain
    @param scorer scorer object used for alignment
    @param score alignment score
    @param cards cuda cards index array
    @param cardsLen cuda cards index array length
    @param thread thread on which the function will be executed, if NULL function is
        executed on the current thread
    */
    extern void alignScoredPair(Alignment **alignment, int type, Chain *query,
                                Chain *target, Scorer *scorer, int score, int *cards, int cardsLen,
                                Thread *thread);

    /*!
    @brief Best scored pair alignment function.

    Function aligns queries and the target chain with the scorer object.
    Only the best scored pair is aligned and returned.
    If needed function utilzes provided CUDA cards.

    @param alignment output alignment object
    @param type aligning type, can be #SW_ALIGN, #NW_ALIGN, #HW_ALIGN or #OV_ALIGN
    @param queries query chains array
    @param queriesLen query chains array length
    @param target target chain
    @param scorer scorer object used for alignment
    @param cards cuda cards index array
    @param cardsLen cuda cards index array length
    @param thread thread on which the function will be executed, if NULL function is
        executed on the current thread
    */
    extern void alignBest(Alignment **alignment, int type, Chain **queries,
                          int queriesLen, Chain *target, Scorer *scorer, int *cards, int cardsLen,
                          Thread *thread);

    /*!
    @brief Pairwise scoring function.

    Function only returns the score of the alignment.
    If needed function utilzes provided CUDA cards.

    @param score output score
    @param type aligning type, can be #SW_ALIGN, #NW_ALIGN, #HW_ALIGN or #OV_ALIGN
    @param query query chain
    @param target target chain
    @param scorer scorer object used for alignment
    @param cards cuda cards index array
    @param cardsLen cuda cards index array length
    @param thread thread on which the function will be executed, if NULL function is
        executed on the current thread
    */
    extern void scorePair(int *score, int type, Chain *query, Chain *target,
                          Scorer *scorer, int *cards, int cardsLen, Thread *thread);

#ifdef __cplusplus
}
#endif
#endif // __SW_SHARP_ALIGNH__
