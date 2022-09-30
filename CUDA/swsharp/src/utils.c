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

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

#include "error.h"
#include "thread.h"

#include "utils.h"

#ifdef DEBUG

typedef struct Timer
{
    Thread thread;
    double time;
} Timer;

typedef struct TimerStack
{
    int top;
    int mutexInit;
    Mutex mutex;
    Timer stack[100];
} TimerStack;

static TimerStack timerStack = {0, 0};

static double getTime()
{
#ifdef _WIN32
    SYSTEMTIME time;
    GetSystemTime(&time);
    double s = 0;
    s += time.wDay * 86400.0;
    s += time.wHour * 3600.0;
    s += time.wMinute * 60.0;
    s += time.wSecond;
    s += time.wMilliseconds / 1000.0;
    return s;
#else
    struct timeval time;
    gettimeofday(&time, NULL);
    return time.tv_sec + time.tv_usec / 1e6;
#endif
}

extern void addTimer()
{

    Thread thread;
    threadCurrent(&thread);

    Timer timer = {thread, getTime()};

    if (!timerStack.mutexInit)
    {
        mutexCreate(&(timerStack.mutex));
        timerStack.mutexInit = 1;
    }

    mutexLock(&(timerStack.mutex));

    timerStack.stack[timerStack.top++] = timer;

    mutexUnlock(&(timerStack.mutex));
}

extern float stopTimer()
{

    Thread thread;
    threadCurrent(&thread);

    mutexLock(&(timerStack.mutex));

    int i = timerStack.top - 1;

    while (i >= 0)
    {

        if (timerStack.stack[i].thread == thread)
        {
            break;
        }

        i--;
    }

    double start = timerStack.stack[i].time;

    while (i < timerStack.top - 1)
    {
        timerStack.stack[i] = timerStack.stack[i + 1];
        i++;
    }

    timerStack.top--;

    mutexUnlock(&(timerStack.mutex));

    double end = getTime();

    return (float)(end - start);
}
#endif

extern FILE *fileSafeOpen(const char *path, const char *mode)
{
    FILE *f = fopen(path, mode);
    ASSERT(f != NULL, "cannot open file %s with mode %s", path, mode);
    return f;
}

extern size_t fileLength(FILE *f)
{

    fseek(f, 0L, SEEK_END);
    size_t len = ftell(f);
    fseek(f, 0L, SEEK_SET);

    return len;
}

static void swap(char *a, char *b, unsigned width)
{
    if (a != b)
    {
        while (width--)
        {
            char tmp = *a;
            *a++ = *b;
            *b++ = tmp;
        }
    }
}

extern void qselect(void *list_, size_t n, size_t size, int k,
                    int (*cmp)(const void *, const void *))
{

    char *list = (char *)list_;

    int low = 0;
    int high = n - 1;

    char *key = (char *)malloc(size);

    while (1)
    {
        if (high <= low + 1)
        {

            if (high == low + 1 && cmp(list + high * size, list + low * size) < 0)
            {
                swap(list + low * size, list + high * size, size);
            }

            free(key);
            return;
        }
        else
        {

            int mid = (low + high) / 2;

            swap(list + mid * size, list + (low + 1) * size, size);

            if (cmp(list + low * size, list + high * size) > 0)
            {
                swap(list + low * size, list + high * size, size);
            }

            if (cmp(list + (low + 1) * size, list + high * size) > 0)
            {
                swap(list + (low + 1) * size, list + high * size, size);
            }

            if (cmp(list + low * size, list + (low + 1) * size) > 0)
            {
                swap(list + low * size, list + (low + 1) * size, size);
            }

            int left = low + 1;
            int right = high;

            memcpy(key, list + (low + 1) * size, size);

            while (1)
            {

                do
                {
                    left++;
                } while (cmp(list + left * size, key) < 0);

                do
                {
                    right--;
                } while (cmp(list + right * size, key) > 0);

                if (right < left)
                {
                    break;
                }

                swap(list + left * size, list + right * size, size);
            }

            if (low + 1 != right)
            {
                memcpy(list + (low + 1) * size, list + right * size, size);
            }

            memcpy(list + right * size, key, size);

            if (right >= k)
            {
                high = right - 1;
            }

            if (right <= k)
            {
                low = left;
            }
        }
    }

    free(key);
}

extern void weightChunkArray(int *dstOff, int *dstLens, int *dstLen, int *src,
                             int srcLen, int chunks)
{

    ASSERT(chunks > 0, "invalid chunk data");

    int i, j;

    if (chunks >= srcLen)
    {

        for (i = 0; i < srcLen; ++i)
        {
            dstOff[i] = i;
            dstLens[i] = 1;
        }

        *dstLen = srcLen;

        return;
    }

    long long sum = 0;
    for (i = 0; i < srcLen; ++i)
    {
        sum += src[i];
    }

    long long chunk = sum / chunks;
    long long current = 0;

    dstOff[0] = 0;
    for (i = 0; i < chunks; ++i)
    {
        dstLens[i] = 0;
    }

    for (i = 0, j = 0; i < srcLen; ++i)
    {

        if ((current > chunk || srcLen - i == chunks - j) && j != chunks - 1)
        {

            current = 0;
            j++;

            dstOff[j] = i;
        }

        current += src[i];
        dstLens[j]++;
    }

    *dstLen = chunks;
}

extern void chunkArray(int ***dst, int **dstLens, int *src, int srcLen,
                       int chunks)
{

    ASSERT(chunks <= srcLen && chunks >= 1, "invalid chunk data");

    *dst = (int **)malloc(chunks * sizeof(int *));
    *dstLens = (int *)malloc(chunks * sizeof(int));

    memset(*dstLens, 0, chunks * sizeof(int));

    int i;

    int left = srcLen;
    i = 0;
    while (left > 0)
    {
        (*dstLens)[i]++;
        i = (i + 1) % chunks;
        left--;
    }

    int offset = 0;
    for (i = 0; i < chunks; ++i)
    {
        (*dst)[i] = src + offset;
        offset += (*dstLens)[i];
    }
}
