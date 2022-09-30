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

/*!
\mainpage

Simple library usage can be seen in the following simple.c file. This short
program aligns two nucleotide sequences in fasta format. The nucleotides paths
are read from the command line as the first two arguments. This example is for
the linux platform.

simple.c:

\code{.c}
#include "swsharp/swsharp.h"

int main(int argc, char* argv[]) {

    Chain* query = NULL;
    Chain* target = NULL;

    // read the query as the first command line argument
    readFastaChain(&query, argv[1]);

    // read the target as the first command line argument
    readFastaChain(&target, argv[2]);

    // use one CUDA card with index 0
    int cards[] = { 0 };
    int cardsLen = 1;

    // create a scorer object
    // match = 1
    // mismatch = -3
    // gap open = 5
    // gap extend = 2
    Scorer* scorer;
    scorerCreateScalar(&scorer, 1, -3, 5, 2);

    // do the pairwise alignment, use Smith-Waterman algorithm
    Alignment* alignment;
    alignPair(&alignment, SW_ALIGN, query, target, scorer, cards, cardsLen, NULL);

    // output the results in emboss stat-pair format
    outputAlignment(alignment, NULL, SW_OUT_STAT_PAIR);

    // clean the memory
    alignmentDelete(alignment);

    chainDelete(query);
    chainDelete(target);

    scorerDelete(scorer);

    return 0;
}
\endcode

This code can be compiled with:

\code
nvcc simple.c -I include/ -L lib/ -l swsharp -l pthread -o simple
\endcode

And the executable can be run with:

\code
./simple input1.fasta input2.fasta
\endcode
*/

/**
@file

@brief SW# project wrapper header.
*/

#ifndef __SW_SHARP_SWSHARPH__
#define __SW_SHARP_SWSHARPH__

#include "align.h"
#include "constants.h"
#include "cpu_module.h"
#include "cuda_utils.h"
#include "database.h"
#include "gpu_module.h"
#include "post_proc.h"
#include "pre_proc.h"
#include "reconstruct.h"
#include "threadpool.h"

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef __cplusplus
}
#endif
#endif // __SW_SHARP_SWSHARPH__
