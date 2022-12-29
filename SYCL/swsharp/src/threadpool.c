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
#include <stdlib.h>

#include "thread.h"

#include "threadpool.h"

#define QUEUE_MAX_SIZE 100000

struct ThreadPoolTask
{
    Semaphore wait;
    void *(*routine)(void *);
    void *param;
};

typedef struct ThreadPoolQueue
{
    ThreadPoolTask **data;
    int length;
    int maxLength;
    int current;
    int last;
    int full;
    Semaphore mutex;
    Semaphore wait;   // notify when task over if queue full
    Semaphore submit; // notify when task submited
} ThreadPoolQueue;

typedef struct ThreadPool
{
    int terminated;
    Thread *threads;
    int threadsLen;
    ThreadPoolQueue queue;
} ThreadPool;

static ThreadPool *threadPool = NULL;

//******************************************************************************
// PUBLIC

extern int threadPoolInitialize(int n);

extern int withThreads();

extern void threadPoolTerminate();

extern ThreadPoolTask *threadPoolSubmit(void *(*routine)(void *), void *param);

extern ThreadPoolTask *threadPoolSubmitToFront(void *(*routine)(void *), void *param);

extern void threadPoolTaskDelete(ThreadPoolTask *task);

extern void threadPoolTaskWait(ThreadPoolTask *task);

//******************************************************************************

//******************************************************************************
// PRIVATE

static void *worker(void *param);

static ThreadPoolTask *sumbit(void *(*routine)(void *), void *param, int toFront);

//******************************************************************************

//******************************************************************************
// PUBLIC

extern int withThreads()
{
    return threadPool != NULL;
}

extern int
threadPoolInitialize(int n)
{

    if (threadPool != NULL)
    {
        return 0;
    }

    if (n < 1)
    {
        return -1;
    }

    Thread *threads = (Thread *)malloc(n * sizeof(Thread));

    int maxLength = QUEUE_MAX_SIZE;
    size_t queueSize = maxLength * sizeof(ThreadPoolTask);
    ThreadPoolTask **queue = (ThreadPoolTask **)malloc(queueSize);

    threadPool = (ThreadPool *)malloc(sizeof(ThreadPool));
    threadPool->terminated = 0;
    threadPool->threads = threads;
    threadPool->threadsLen = n;
    threadPool->queue.data = queue;
    threadPool->queue.length = 0;
    threadPool->queue.maxLength = maxLength;
    threadPool->queue.current = 0;
    threadPool->queue.last = 0;
    threadPool->queue.full = 0;
    semaphoreCreate(&(threadPool->queue.mutex), 1);
    semaphoreCreate(&(threadPool->queue.wait), 0);
    semaphoreCreate(&(threadPool->queue.submit), 0);

    int i;
    for (i = 0; i < n; ++i)
    {
        threadCreate(&(threads[i]), worker, NULL);
    }

    return 0;
}

extern void threadPoolTerminate()
{

    if (threadPool == NULL)
    {
        return;
    }

    int i;
    ThreadPoolQueue *queue = &(threadPool->queue);

    semaphoreWait(&(queue->mutex));

    threadPool->terminated = 1;

    // unlock all threads
    for (i = 0; i < threadPool->threadsLen; ++i)
    {
        semaphorePost(&(queue->submit));
    }

    // unlock waiting on full queue
    semaphorePost(&(queue->wait));

    semaphorePost(&(queue->mutex));

    // wait for threads to be killed
    for (i = 0; i < threadPool->threadsLen; ++i)
    {
        threadJoin(threadPool->threads[i]);
    }

    // release all waiting on tasks
    for (i = queue->current; i != queue->last; ++i)
    {
        if (i == queue->maxLength)
            i = 0;
        semaphorePost(&(queue->data[i]->wait));
    }

    semaphoreDelete(&(queue->wait));
    semaphoreDelete(&(queue->submit));
    semaphoreDelete(&(queue->mutex));

    free(queue->data);
    free(threadPool->threads);

    free(threadPool);
    threadPool = NULL;
}

extern ThreadPoolTask *threadPoolSubmit(void *(*routine)(void *), void *param)
{
    return sumbit(routine, param, 0);
}

extern ThreadPoolTask *threadPoolSubmitToFront(void *(*routine)(void *), void *param)
{
    return sumbit(routine, param, 1);
}

extern void threadPoolTaskDelete(ThreadPoolTask *task)
{

    if (task == NULL)
    {
        return;
    }

    semaphoreDelete(&(task->wait));
    free(task);
}

extern void threadPoolTaskWait(ThreadPoolTask *task)
{

    if (task == NULL)
    {
        return;
    }

    semaphoreWait(&(task->wait));
    semaphorePost(&(task->wait)); // unlock for double waiting
}

//******************************************************************************

//******************************************************************************
// PRIVATE

static ThreadPoolTask *sumbit(void *(*routine)(void *), void *param, int toFront)
{

    if (threadPool == NULL)
    {
        routine(param);
        return NULL;
    }

    ThreadPoolQueue *queue = &(threadPool->queue);

    semaphoreWait(&(queue->mutex));

    if (threadPool->terminated)
    {
        semaphorePost(&(queue->mutex));
        return NULL;
    }

    if (queue->current == (queue->last + 1) % queue->maxLength)
    {
        queue->full = 1;
        semaphorePost(&(queue->mutex));
        semaphoreWait(&(queue->wait));
        semaphoreWait(&(queue->mutex));
    }

    if (threadPool->terminated)
    {
        semaphorePost(&(queue->mutex));
        return NULL;
    }

    ThreadPoolTask *task = (ThreadPoolTask *)malloc(sizeof(ThreadPoolTask));
    task->routine = routine;
    task->param = param;
    semaphoreCreate(&(task->wait), 0);

    if (toFront)
    {
        queue->current = (queue->current - 1 + queue->maxLength) % queue->maxLength;
        queue->data[queue->current] = task;
    }
    else
    {
        queue->data[queue->last] = task;
        queue->last = (queue->last + 1) % queue->maxLength;
    }

    semaphorePost(&(queue->mutex));

    semaphorePost(&(queue->submit));

    return task;
}

static void *worker(void *param)
{

    ThreadPoolQueue *queue = &(threadPool->queue);

    while (1)
    {

        semaphoreWait(&(queue->submit));

        if (threadPool->terminated)
        {
            break;
        }

        semaphoreWait(&(queue->mutex));
        ThreadPoolTask *task = queue->data[queue->current];
        queue->current = (queue->current + 1) % queue->maxLength;
        semaphorePost(&(queue->mutex));

        if (threadPool->terminated)
        {
            semaphorePost(&(task->wait));
            break;
        }

        task->routine(task->param);
        semaphorePost(&(task->wait));

        semaphoreWait(&(queue->mutex));
        if (queue->full && queue->current == queue->last)
        {
            queue->full = 0;
            semaphorePost(&(queue->wait));
            break;
        }
        semaphorePost(&(queue->mutex));
    }

    return NULL;
}

//******************************************************************************
