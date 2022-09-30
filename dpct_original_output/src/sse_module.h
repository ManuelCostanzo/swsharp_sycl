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

#ifndef __SSE_MODULE_H__
#define __SSE_MODULE_H__

#include "alignment.h"
#include "chain.h"
#include "scorer.h"

#ifdef __cplusplus 
extern "C" {
#endif

extern int alignPairSse(Alignment** alignment, int type, Chain* query, 
    Chain* target, Scorer* scorer);

extern int alignScoredPairSse(Alignment** alignment, int type, Chain* query, 
    Chain* target, Scorer* scorer, int score);

extern int scorePairSse(int* score, int type, Chain* query, Chain* target,
    Scorer* scorer);

extern int scoreDatabaseSse(int* scores, int type, Chain* query, 
    Chain** database, int databaseLen, Scorer* scorer);

extern int scoreDatabasePartiallySse(int* scores, int type, Chain* query, 
    Chain** database, int databaseLen, Scorer* scorer, int maxScore);

#ifdef __cplusplus 
}
#endif
#endif // __SSE_MODULE_H__
