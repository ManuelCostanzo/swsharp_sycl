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

#ifndef __EVALUEH__
#define __EVALUEH__

#include "chain.h"
#include "scorer.h"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct EValueParams EValueParams;

    extern EValueParams *createEValueParams(long long length, Scorer *scorer);

    extern void deleteEValueParams(EValueParams *eValueParams);

    extern void eValues(double *values, int *scores, Chain *query,
                        Chain **database, int databaseLen, int *cards, int cardsLen,
                        EValueParams *eValueParams);

#ifdef __cplusplus
}
#endif
#endif // __EVALUEH__
