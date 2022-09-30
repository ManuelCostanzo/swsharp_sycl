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

#ifndef __SW_SHARP_ERRORH__
#define __SW_SHARP_ERRORH__

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus 
extern "C" {
#endif

#define ERROR(fmt, ...) ASSERT(0, fmt, ##__VA_ARGS__)

#define ASSERT(expr, fmt, ...)\
    do {\
        if (!(expr)) {\
            fprintf(stderr, "[ERROR:%s:%d]: " fmt "\n", __FILE__, __LINE__,\
                ##__VA_ARGS__);\
            exit(-1);\
        }\
    } while(0)
    
#define ASSERT_CALL(expr, call, fmt, ...)\
    do {\
        if (!(expr)) {\
            fprintf(stderr, "[ERROR:%s:%d]: " fmt "\n", __FILE__, __LINE__,\
                ##__VA_ARGS__);\
            call(); \
            exit(-1);\
        }\
    } while(0)

#define WARNING(expr, fmt, ...)\
    do {\
        if (expr) {\
            printf("[WARNING:%s:%d]: " fmt "\n", __FILE__, __LINE__,\
                                    ##__VA_ARGS__);\
        }\
    } while(0)
    
#ifdef __cplusplus 
}
#endif
#endif // __SW_SHARP_ERRORH__
