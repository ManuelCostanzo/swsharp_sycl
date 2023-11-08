
#ifdef SYCL_LANGUAGE_VERSION
#include <CL/sycl.hpp>
#ifdef HIP
namespace sycl = cl::sycl;
#endif
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

Contact SW#-SYCL authors by mcostanzo@lidi.info.unlp.edu.ar,
erucci@lidi.info.unlp.edu.ar
*/
/**
@file

@brief CUDA utility header.
*/

#ifndef __SW_SHARP_CUDA_UTILSH__
#define __SW_SHARP_CUDA_UTILSH__

#ifdef __cplusplus
extern "C" {
#endif

#define CUDA_SAFE_CALL(call)                                                   \
  do {                                                                         \
    int err = call;                                                            \
                                                                               \
  } while (0)

#define MAX3(x, y, w) sycl::max(x, sycl::max(y, w))
#define MAX4(x, y, w, z) sycl::max(sycl::max(x, y), sycl::max(w, z))

#define VEC2_ASSIGN(X, Y)                                                      \
  do {                                                                         \
    (X).x() = (Y).x();                                                         \
    (X).y() = (Y).y();                                                         \
  } while (0)

#define VEC3_ASSIGN(X, Y)                                                      \
  do {                                                                         \
    (X).x() = (Y).x();                                                         \
    (X).y() = (Y).y();                                                         \
    (X).z() = (Y).z();                                                         \
  } while (0)

#define VEC4_ASSIGN(X, Y)                                                      \
  do {                                                                         \
    (X).x() = (Y).x();                                                         \
    (X).y() = (Y).y();                                                         \
    (X).z() = (Y).z();                                                         \
    (X).w() = (Y).w();                                                         \
  } while (0)

extern std::unordered_map<int, sycl::queue> queues;

/*!
@brief Gets CUDA card index list.

@param cards output CUDA cards index array
@param cardsLen output CUDA cards index array length
*/
extern void cudaGetCards(int **cards, int *cardsLen);

/*!
@brief Checks CUDA card index list.

@param cards CUDA cards index array
@param cardsLen CUDA cards index array length

@return 1 if all input card indices are available, 0 otherwise
*/
extern int cudaCheckCards(int *cards, int cardsLen);

extern void loadQueues(int *cards, int cardsLen);

/*!
@brief Get minimal available CUDA global memory

@param cards CUDA cards index array
@param cardsLen CUDA cards index array length

@return global memory size of CUDA card with least memory
*/
extern size_t cudaMinimalGlobalMemory(int *cards, int cardsLen);

extern void maxWorkGroups(int card, int defaultBlocks, int defaultThreads,
                          int cols, int *blocks, int *threads);

#ifdef __cplusplus
}
#endif
#endif // __SW_SHARP_CUDA_UTILSH__
#endif