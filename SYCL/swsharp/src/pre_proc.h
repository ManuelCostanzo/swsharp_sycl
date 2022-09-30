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

@brief Preprocessing utilities header.
*/

#ifndef __SW_SHARP_PRE_PROCESH__
#define __SW_SHARP_PRE_PROCESH__

#include <stdio.h>
#include "chain.h"
#include "scorer.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /*!
    @brief Creates chain complement.

    Function persumes the characters in the chain are nucleotides. Every 'A' is
    turned into 'T', every 'C' to 'G' and vice versa. Any other characters are left
    intact. Also the complement chain is in the reverse order of the original.

    @param chain chain object

    @return chain complement
    */
    extern Chain *createChainComplement(Chain *chain);

    /*!
    @brief Fasta chain reading function.

    Function reads file from the given path, which should be in the fasta format.
    More on fasta format <a>http://en.wikipedia.org/wiki/FASTA_format</a>. Function
    persumes only one chain is in the file. Function persumes that the fasta chain
    name is no more than 1000 characters long.

    @param chain output chain object
    @param path file path
    */
    extern void readFastaChain(Chain **chain, const char *path);

    /*!
    @brief Fasta chain reading function.

    Function works in the same way as the readFastaChain() but does not persume
    only one chain is in the file.

    @param chains output chain array object
    @param chainsLen output chain array length
    @param path file path
    */
    extern void readFastaChains(Chain ***chains, int *chainsLen, const char *path);

    extern void readFastaChainsPartInit(Chain ***chains, int *chainsLen,
                                        FILE **handle, int *serialized, const char *path);

    extern int readFastaChainsPart(Chain ***chains, int *chainsLen,
                                   FILE *handle, int serialized, const size_t maxBytes);

    extern int skipFastaChainsPart(Chain ***chains, int *chainsLen,
                                   FILE *handle, int serialized, const size_t skip);

    extern void statFastaChains(int *chains, long long *cells, const char *path);

    /*!
    @brief Fasta database serialization function.

    Function creates a file named path.swsharp which represents the serialized
    version of the chain database which was read from the given path. Since reading
    of the serialized version of the database is much faster than reading the
    original one, this function is used for caching the databases for future usage.

    @param path original Fasta chain database file path
    */
    extern void dumpFastaChains(char *path);

    /*!
    @brief Scalar scorer creation utility functions.

    Scorer is created with the max score equal to 26. Observing the scorerEncode()
    method it means this scorer only accepts alphabet characters and it must not be
    used with no other characters. This scorer is scalar, see scorerIsScalar().

    @param scorer output scorer object
    @param match match score for every two equal codes, defined as positive integer
    @param mismatch mismatch penalty for every two unequal codes, defined as
        negative integer
    @param gapOpen affine gap open penalty, defined as positive integer
    @param gapExtend affine gap extend penalty, defined as positive integer
    */
    extern void scorerCreateScalar(Scorer **scorer, int match, int mismatch,
                                   int gapOpen, int gapExtend);

    /*!
    @brief Nonscalar scorer creation utility functions.

    Scorer is created with the max score equal to 26. Observing the scorerEncode()
    method it means this scorer only accepts alphabet characters and it must not be
    used with no other characters. This function is used for creating the scorer
    with the most common standard similarity matrices.

    @param scorer output scorer object
    @param name name can be one of the "BLOSUM_62", "BLOSUM_45", "BLOSUM_50",
        "BLOSUM_80", "BLOSUM_90", "PAM_30", "PAM_70" or "PAM_250"
    @param gapOpen affine gap open penalty, defined as positive integer
    @param gapExtend affine gap extend penalty, defined as positive integer
    */
    extern void scorerCreateMatrix(Scorer **scorer, char *name, int gapOpen,
                                   int gapExtend);

#ifdef __cplusplus
}
#endif
#endif // __SW_SHARP_PRE_PROCESH__
