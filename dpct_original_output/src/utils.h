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

#ifndef __SW_SHARP_UTILSH__
#define __SW_SHARP_UTILSH__

#ifdef _WIN32
#include <windows.h>
#endif

#include <stdio.h>
#include <string.h>

#ifdef __cplusplus 
extern "C" {
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define SWAP(x, y) {\
        char tmp[sizeof(x)];\
        memcpy(tmp, &(x), sizeof(x));\
        memcpy(&(x), &(y), sizeof(x));\
        memcpy(&(y), tmp, sizeof(x));\
    }

#ifdef DEBUG
#define LOG(fmt, ...) \
    do { \
        printf("[LOG:%s:%d]: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);\
    } while(0)
    
extern void addTimer();
extern float stopTimer();

#ifdef TIMERS
#define TIMER_START(fmt, ...) \
    do { \
        printf("[TIMER_START:%s:%d]: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);\
        addTimer(); \
    } while(0)

#define TIMER_STOP \
do { \
    printf("[TIMER_STOP:%s:%d]: %.3f\n", __FILE__, __LINE__, stopTimer());\
} while(0)
#else 
#define TIMER_START(fmt, ...)
#define TIMER_STOP
#endif

#else 
#define TIMER_START(fmt, ...)
#define TIMER_STOP
#define LOG(fmt, ...)
#endif

extern FILE* fileSafeOpen(const char* path, const char* mode);
extern size_t fileLength(FILE* f);

extern void qselect(void* list, size_t n, size_t size, int k, 
    int (*cmp)(const void*, const void*));

extern void weightChunkArray(int* dstOff, int* dstLens, int* dstLen, int* src, 
    int srcLen, int chunks);

extern void chunkArray(int*** dst, int** dstLens, int* src, int srcLen,
    int chunks);

#ifdef __cplusplus 
}
#endif
#endif // __SW_SHARP_UTILSH__
