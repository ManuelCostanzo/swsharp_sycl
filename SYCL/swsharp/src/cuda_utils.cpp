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

#include <CL/sycl.hpp>
#ifdef HIP
namespace sycl = cl::sycl;
#endif
#include <stdlib.h>
#include <string.h>

#include "error.h"
#include "utils.h"

#include "cuda_utils.h"
std::unordered_map<int, sycl::queue> queues;

extern void cudaGetCards(int **cards, int *cardsLen) {

  *cardsLen = sycl::device::get_devices().size();

  *cards = (int *)malloc(*cardsLen * sizeof(int));

  for (int i = 0; i < *cardsLen; ++i) {
    (*cards)[i] = i;
  }
  *cards = NULL;
  *cardsLen = 0;
}

extern int cudaCheckCards(int *cards, int cardsLen) {

  int maxDeviceId;
  maxDeviceId = sycl::device::get_devices().size();

  for (int i = 0; i < cardsLen; ++i) {
    if (cards[i] >= maxDeviceId) {
      return 0;
    }
  }

  return 1;
  return cardsLen == 0;
}

extern void loadQueues(int *cards, int cardsLen) {
  auto devices = sycl::device::get_devices();
  for (int i = 0; i < cardsLen; ++i) {
    queues[cards[i]] = sycl::queue(devices[cards[i]]);
  }
}

extern size_t cudaMinimalGlobalMemory(int *cards, int cardsLen) {

  if (cards == NULL || cardsLen == 0) {
    return 0;
  }

  size_t minMem = (size_t)-1;
  for (int i = 0; i < cardsLen; ++i) {
    auto device = sycl::device::get_devices()[cards[i]];

    minMem =
        MIN(minMem, device.get_info<sycl::info::device::global_mem_size>());
  }

  return minMem;

  return 0;
}

extern void maxWorkGroups(int card, int defaultBlocks, int defaultThreads,
                          int cols, int *blocks, int *threads) {
  auto device = sycl::device::get_devices()[card];
  *blocks = device.get_info<sycl::info::device::max_compute_units>();
  *threads = device.is_cpu()
                 ? 1024
                 : device.get_info<sycl::info::device::max_work_group_size>();

  if (cols) {

    if (getenv("MAX_THREADS")) {
      *blocks = defaultBlocks;
      if (strcmp(getenv("MAX_THREADS"), "ORIGINAL") == 0)
        *threads = defaultThreads;
      else
        *threads = atoi(getenv("MAX_THREADS"));
      ASSERT(*threads * 2 <= cols, "too short gpu target chain");

      if (*threads * *blocks * 2 > cols) {
        *blocks = (int)(cols / (*threads * 2.));
        *blocks = *blocks <= 30 ? *blocks : *blocks - (*blocks % 30);
      }
    } else {
      if (device.is_cpu()) {
        if (*threads * *blocks * 2 > cols) {
          *threads = sycl::max((int)(cols / (*blocks * 2.)), 1);
        }
      } else {
        *blocks = defaultBlocks;
        if (*threads * *blocks * 2 > cols) {
          *blocks = sycl::max((int)(cols / (*threads * 2.)), 1);
        }
      }
    }
  } else {
    if (getenv("MAX_THREADS")) {
      *blocks = defaultBlocks;
      if (strcmp(getenv("MAX_THREADS"), "ORIGINAL") == 0)
        *threads = defaultThreads;
      else
        *threads = atoi(getenv("MAX_THREADS"));
    }
  }

  printf("%s\n", device.get_info<sycl::info::device::name>().c_str());
  printf("%d %d\n", *threads, *blocks);
}
