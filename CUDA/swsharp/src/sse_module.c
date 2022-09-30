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

#include <stdio.h>

#include "chain.h"
#include "constants.h"
#include "error.h"
#include "scorer.h"
#include "utils.h"

#include "swimd/Swimd.h"
#include "ssw/ssw.h"

#include "sse_module.h"

//******************************************************************************
// PUBLIC

//******************************************************************************

//******************************************************************************
// PRIVATE

static int sswWrapper(s_align **a, int type, Chain *query, Chain *target,
                      Scorer *scorer, int score, int flag);

static int sswDatabaseWrapper(int *scores, int type, Chain *query,
                              Chain **database, int databaseLen, Scorer *scorer);

static int swimdWrapper(int *scores, int type, Chain *query, Chain **database,
                        int databaseLen, Scorer *scorer, int solveChar);

//******************************************************************************

//******************************************************************************
// PUBLIC

extern int alignPairSse(Alignment **alignment, int type, Chain *query,
                        Chain *target, Scorer *scorer)
{
    return alignScoredPairSse(alignment, type, query, target, scorer, NO_SCORE);
}

extern int alignScoredPairSse(Alignment **alignment, int type, Chain *query,
                              Chain *target, Scorer *scorer, int score)
{

    int i, j;

    s_align *a = NULL;

    if (sswWrapper(&a, type, query, target, scorer, score, 1) != 0)
    {

        if (a != NULL)
        {
            align_destroy(a);
        }

        return -1;
    }

    uint32_t *cigar = a->cigar;
    int32_t cigarLen = a->cigarLen;

    int pathLen = 0;
    for (i = 0; i < cigarLen; ++i)
    {
        pathLen += cigar[i] >> 4;
    }

    char *path = (char *)malloc(pathLen * sizeof(char));
    for (i = 0, j = 0; i < cigarLen; ++i)
    {

        int len = cigar[i] >> 4;
        int val = cigar[i] & 0xF;

        char move;
        switch (val)
        {
        case 0:
            move = MOVE_DIAG;
            break;
        case 1:
            move = MOVE_UP;
            break;
        default:
            move = MOVE_LEFT;
            break;
        }

        memset(path + j, move, len * sizeof(char));
        j += len;
    }

    *alignment = alignmentCreate(query, a->read_begin1, a->read_end1, target,
                                 a->ref_begin1, a->ref_end1, a->score1, scorer, path, pathLen);

    /*
    printf("%d\n", a->score1);
    printf("%d %d %d %d\n", a->read_begin1, a->read_end1, a->ref_begin1, a->ref_end1);
    printf("%d\n", a->cigarLen);
    printf("%d\n", pathLen);
    */

    align_destroy(a);

    return 0;
}

extern int scorePairSse(int *score, int type, Chain *query, Chain *target,
                        Scorer *scorer)
{

    if (type != SW_ALIGN)
    {
        return -1;
    }

    s_align *a = NULL;

    if (sswWrapper(&a, type, query, target, scorer, NO_SCORE, 0) != 0)
    {

        if (a != NULL)
        {
            align_destroy(a);
        }

        return -1;
    }

    *score = a->score1;

    align_destroy(a);

    return 0;
}

extern int scoreDatabaseSse(int *scores, int type, Chain *query,
                            Chain **database, int databaseLen, Scorer *scorer)
{

    if (swimdWrapper(scores, type, query, database, databaseLen, scorer, 0) == 0)
    {
        return 0;
    }

    if (sswDatabaseWrapper(scores, type, query, database, databaseLen, scorer) == 0)
    {
        return 0;
    }

    return -1;
}

extern int scoreDatabasePartiallySse(int *scores, int type, Chain *query,
                                     Chain **database, int databaseLen, Scorer *scorer, int maxScore)
{

    if (swimdWrapper(scores, type, query, database, databaseLen, scorer, 1) == 0)
    {
        return 0;
    }

    return -1;
}

//******************************************************************************

//******************************************************************************
// PRIVATE

static int sswWrapper(s_align **a, int type, Chain *query, Chain *target,
                      Scorer *scorer, int score, int flag)
{

    const int sswMaxScore = (1 << 15) - 1;

    if (type != SW_ALIGN || score > sswMaxScore)
    {
        return -1;
    }

    if (score == NO_SCORE)
    {

        int queryLen = chainGetLength(query);
        int targetLen = chainGetLength(target);
        int maxScore = scorerGetMaxScore(scorer);

        int minLen = MIN(queryLen, targetLen);

        if (minLen * maxScore > sswMaxScore)
        {

            Chain *chain = queryLen < targetLen ? query : target;
            const char *codes = chainGetCodes(chain);

            int i;
            int score = 0;
            for (i = 0; i < minLen; ++i)
            {
                score += scorerScore(scorer, codes[i], codes[i]);
            }

            if (score > sswMaxScore)
            {
                return -1;
            }
        }
    }

    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);

    if (abs(gapOpen) > 127 || abs(gapExtend) > 127)
    {
        return -1;
    }

    const int32_t n = scorerGetMaxCode(scorer);
    int8_t *mat = (int8_t *)malloc(n * n * sizeof(int8_t));

    const int *table = scorerGetTable(scorer);

    int i;
    for (i = 0; i < n * n; ++i)
    {

        int val = table[i];

        // can't use ssw
        if (abs(val) > 127)
        {
            free(mat);
            return -1;
        }

        mat[i] = (int8_t)val;
    }

    const int8_t *read = (const int8_t *)chainGetCodes(query);
    const int32_t readLen = chainGetLength(query);

    s_profile *prof = ssw_init(read, readLen, mat, n, 2);

    const uint8_t weight_gapO = (const uint8_t)gapOpen;
    const uint8_t weight_gapE = (const uint8_t)gapExtend;

    const int8_t *ref = (const int8_t *)chainGetCodes(target);
    const int32_t refLen = chainGetLength(target);

    int8_t score_size;
    if (score == NO_SCORE)
    {
        score_size = 2;
    }
    else if (score < 255)
    {
        score_size = 0;
    }
    else
    {
        score_size = 1;
    }

    *a = ssw_align(prof, ref, refLen, weight_gapO, weight_gapE, flag,
                   0, 0, score_size);

    init_destroy(prof);
    free(mat);

    return 0;
}

static int sswDatabaseWrapper(int *scores, int type, Chain *query,
                              Chain **database, int databaseLen, Scorer *scorer)
{

    if (type != SW_ALIGN)
    {
        return -1;
    }

    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);

    if (abs(gapOpen) > 127 || abs(gapExtend) > 127)
    {
        return -1;
    }

    const int32_t n = scorerGetMaxCode(scorer);
    int8_t *mat = (int8_t *)malloc(n * n * sizeof(int8_t));

    const int *table = scorerGetTable(scorer);

    int i;
    for (i = 0; i < n * n; ++i)
    {

        int val = table[i];

        // can't use ssw
        if (abs(val) > 127)
        {
            free(mat);
            return -1;
        }

        mat[i] = (int8_t)val;
    }

    const int8_t *read = (const int8_t *)chainGetCodes(query);
    const int32_t readLen = chainGetLength(query);

    s_profile *prof = ssw_init(read, readLen, mat, n, 2);

    const uint8_t weight_gapO = (const uint8_t)gapOpen;
    const uint8_t weight_gapE = (const uint8_t)gapExtend;

    for (i = 0; i < databaseLen; ++i)
    {

        Chain *target = database[i];

        const int8_t *ref = (const int8_t *)chainGetCodes(target);
        const int32_t refLen = chainGetLength(target);

        s_align *a = ssw_align(prof, ref, refLen, weight_gapO, weight_gapE,
                               0, 0, 0, 2);

        scores[i] = a->score1;

        align_destroy(a);
    }

    init_destroy(prof);
    free(mat);

    return 0;
}

static int swimdWrapper(int *scores, int type, Chain *query, Chain **database,
                        int databaseLen, Scorer *scorer, int solveChar)
{

#if defined(__SSE4_1__) || defined(__AVX2__)

    int mode;
    switch (type)
    {
    case SW_ALIGN:
        mode = SWIMD_MODE_SW;
        break;
    case HW_ALIGN:
        mode = SWIMD_MODE_HW;
        break;
    case NW_ALIGN:
        mode = SWIMD_MODE_NW;
        break;
    case OV_ALIGN:
        mode = SWIMD_MODE_OV;
        break;
    default:
        return -1;
    }

    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);

    int *table = (int *)scorerGetTable(scorer);
    int maxCode = scorerGetMaxCode(scorer);

    unsigned char *queryPtr = (unsigned char *)chainGetCodes(query);
    int queryLen = chainGetLength(query);

    unsigned char **databasePtrs =
        (unsigned char **)malloc(databaseLen * sizeof(unsigned char *));

    int *databaseLens = (int *)malloc(databaseLen * sizeof(int));

    int i;
    for (i = 0; i < databaseLen; ++i)
    {
        databasePtrs[i] = (unsigned char *)chainGetCodes(database[i]);
        databaseLens[i] = chainGetLength(database[i]);
    }

    int status;
    if (solveChar && type == SW_ALIGN)
    {

        status = swimdSearchDatabaseCharSW(queryPtr, queryLen, databasePtrs,
                                           databaseLen, databaseLens, gapOpen, gapExtend, table, maxCode,
                                           scores);

        if (status != SWIMD_ERR_NO_SIMD_SUPPORT)
        {
            status = 0;
        }

        for (i = 0; i < databaseLen; ++i)
        {
            if (scores[i] == -1)
            {
                scores[i] = 128;
            }
        }
    }
    else
    {
        status = swimdSearchDatabase(queryPtr, queryLen, databasePtrs,
                                     databaseLen, databaseLens, gapOpen, gapExtend, table, maxCode,
                                     scores, mode, SWIMD_OVERFLOW_SIMPLE);
    }

    free(databasePtrs);
    free(databaseLens);

    return status;

#else
    return -1;
#endif
}

//******************************************************************************
