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

#include <stdlib.h>
#include <string.h>

#include "error.h"
#include "utils.h"

#include "cuda_utils.h"

extern void cudaGetCards(int **cards, int *cardsLen)
{

#ifdef __CUDACC__
    cudaGetDeviceCount(cardsLen);

    *cards = (int *)malloc(*cardsLen * sizeof(int));

    for (int i = 0; i < *cardsLen; ++i)
    {
        (*cards)[i] = i;
    }
#else
    *cards = NULL;
    *cardsLen = 0;
#endif
}

extern int cudaCheckCards(int *cards, int cardsLen)
{

#ifdef __CUDACC__
    int maxDeviceId;
    cudaGetDeviceCount(&maxDeviceId);

    for (int i = 0; i < cardsLen; ++i)
    {
        if (cards[i] >= maxDeviceId)
        {
            return 0;
        }
    }

    return 1;
#else
    return cardsLen == 0;
#endif
}

extern size_t cudaMinimalGlobalMemory(int *cards, int cardsLen)
{

#ifdef __CUDACC__

    if (cards == NULL || cardsLen == 0)
    {
        return 0;
    }

    size_t minMem = (size_t)-1;
    for (int i = 0; i < cardsLen; ++i)
    {

        cudaDeviceProp cdp;
        cudaGetDeviceProperties(&cdp, i);

        minMem = MIN(minMem, cdp.totalGlobalMem);
    }

    return minMem;
#else
    return 0;
#endif
}

extern void maxWorkGroups(int card, int defaultBlocks, int defaultThreads, int cols, int *blocks, int *threads)
{

    cudaDeviceProp cdp;
    cudaGetDeviceProperties(&cdp, card);

    *threads = cdp.maxThreadsPerBlock;
    *blocks = cdp.multiProcessorCount;

    if (cols)
    {

        if (getenv("MAX_THREADS"))
        {
            *blocks = defaultBlocks;
            if (strcmp(getenv("MAX_THREADS"), "ORIGINAL") == 0)
                *threads = defaultThreads;
            else
                *threads = atoi(getenv("MAX_THREADS"));
            ASSERT(*threads * 2 <= cols, "too short gpu target chain");

            if (*threads * *blocks * 2 > cols)
            {
                *blocks = (int)(cols / (*threads * 2.));
                *blocks = *blocks <= 30 ? *blocks : *blocks - (*blocks % 30);
            }
        }
        else
        {

            *blocks = defaultBlocks;
            if (*threads * *blocks * 2 > cols)
            {
                *blocks = max((int)(cols / (*threads * 2.)), 1);
            }
        }
    }
    else
    {
        if (getenv("MAX_THREADS"))
        {
            *blocks = defaultBlocks;
            if (strcmp(getenv("MAX_THREADS"), "ORIGINAL") == 0)
                *threads = defaultThreads;
            else
                *threads = atoi(getenv("MAX_THREADS"));
        }
    }

    // printf("%s\n", cdp.namE);
    // printf("%d %d\n", *threads, *blocks);
}
