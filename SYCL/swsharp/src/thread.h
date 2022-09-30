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

@brief Multiplatform threading header.
*/

#ifndef __SW_SHARP_THREADH__
#define __SW_SHARP_THREADH__

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <semaphore.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef _WIN32
  /*!
  @brief Mutex type.
  */
  typedef CRITICAL_SECTION Mutex;

  /*!
  @brief Semaphore type.
  */
  typedef HANDLE Semaphore;

  /*!
  @brief Thread type.
  */
  typedef HANDLE Thread;

#elif defined(__APPLE__)

/*!
@brief Mutex type.
*/
typedef pthread_mutex_t Mutex;

/*!
@brief Semaphore type.
*/
typedef struct AppleSemaphore *Semaphore;

/*!
@brief Thread type.
*/
typedef pthread_t Thread;

#else
/*!
@brief Mutex type.
*/
typedef pthread_mutex_t Mutex;

/*!
@brief Semaphore type.
*/
typedef sem_t Semaphore;

/*!
@brief Thread type.
*/
typedef pthread_t Thread;
#endif

  /*!
  @brief Mutex constructor.

  @param mutex output mutex object
  */
  extern void mutexCreate(Mutex *mutex);

  /*!
  @brief Mutex destructor.

  @param mutex mutex object
  */
  extern void mutexDelete(Mutex *mutex);

  /*!
  @brief Locks the mutex.

  @param mutex mutex object
  */
  extern void mutexLock(Mutex *mutex);

  /*!
  @brief Unlocks the mutex.

  @param mutex mutex object
  */
  extern void mutexUnlock(Mutex *mutex);

  /*!
  @brief Semaphore constructor.

  @param semaphore output semaphore object
  @param value initial semaphore value
  */
  extern void semaphoreCreate(Semaphore *semaphore, unsigned int value);

  /*!
  @brief Semaphore destructor.

  @param semaphore semaphore object
  */
  extern void semaphoreDelete(Semaphore *semaphore);

  /*!
  @brief Increases sempahore value.

  @param semaphore semaphore object
  */
  extern void semaphorePost(Semaphore *semaphore);

  /*!
  @brief Decreses sempahore value or waits.

  Current thread waits until semaphore value isn't greater than zero and then
  decreases the value and continues the execturion.
  */
  extern void semaphoreWait(Semaphore *semaphore);

  /*!
  @brief Cancels the current thread.

  @param thread thread object
  */
  extern void threadCancel(Thread thread);

  /*!
  @brief Thread constructor.

  @param thread thread output object
  @param *routine routine that the thread executes
  @param args arguments passed to the rutine function
  */
  extern void threadCreate(Thread *thread, void *(*routine)(void *), void *args);

  /*!
  @brief Thread getter.

  @param thread thread output object
  */
  extern void threadCurrent(Thread *thread);

  /*!
  @brief Waits for thread to finish.

  @param thread thread object
  */
  extern void threadJoin(Thread thread);

  /*!
  @brief Sleeps the current thread.

  @param ms time to force current thread to sleep in milliseconds
  */
  extern void threadSleep(unsigned int ms);
#ifdef __cplusplus
}
#endif
#endif // __SW_SHARP_THREADH__
