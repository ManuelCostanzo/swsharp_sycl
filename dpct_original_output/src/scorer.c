/*
swsharp - CUDA parallelized Smith Waterman with applying Hirschberg's and 
Ukkonen's algorithm and dynamic cell pruning.
Copyright (C) 2013 Matija Korpar, contributor Mile Å ikiÄ‡

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

#include <stdlib.h>
#include <string.h>

#include "error.h"
#include "utils.h"

#include "scorer.h"

struct Scorer {

    char* name;
    int nameLen;
    
    int gapOpen;
    int gapExtend;
    
    int* table;

    int maxCode;
    int maxScore;
    int scalar;
};

static const char CODER[] = {
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,   0,   1,   2,   3,   4, 
      5,   6,   7,   8,   9,  10,  11,  12,  13,  14, 
     15,  16,  17,  18,  19,  20,  21,  22,  23,  24, 
     25,  -1,  -1,  -1,  -1,  -1,  -1,   0,   1,   2, 
      3,   4,   5,   6,   7,   8,   9,  10,  11,  12, 
     13,  14,  15,  16,  17,  18,  19,  20,  21,  22, 
     23,  24,  25,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1
};

static const char DECODER[] = {
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
    'U', 'V', 'W', 'X', 'Y', 'Z',  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
     -1,  -1,  -1,  -1,  -1
};

//******************************************************************************
// PUBLIC

//******************************************************************************

//******************************************************************************
// PRIVATE

static int isScalar(Scorer* scorer);

static int maxScore(Scorer* scorer);

//******************************************************************************

//******************************************************************************
// PUBLIC

//------------------------------------------------------------------------------
// CONSTRUCTOR, DESTRUCTOR

extern Scorer* scorerCreate(const char* name, int* scores, char maxCode, 
    int gapOpen, int gapExtend) {

    ASSERT(maxCode > 0, "scorer table must have at least one element");
    ASSERT(gapOpen > 0, "gap open is defined as positive integer");
    ASSERT(gapExtend > 0, "gap extend is defined as positive integer");
    ASSERT(gapOpen >= gapExtend, "gap extend must be equal or less to gap open");
    
    int i;
    int j;
    for (i = 0; i < maxCode; ++i) {
        for (j = i + 1; j < maxCode; ++j) {
            int a = scores[i * maxCode + j];
            int b = scores[j * maxCode + i];
            ASSERT(a == b, "scorer table must be symmetrical");
        }
    }
    
    Scorer* scorer = (Scorer*) malloc(sizeof(struct Scorer));

    scorer->nameLen = strlen(name) + 1;
    scorer->name = (char*) malloc(scorer->nameLen * sizeof(char));
    scorer->name[scorer->nameLen - 1] = '\0';
    memcpy(scorer->name, name, (scorer->nameLen - 1) * sizeof(char));

    scorer->gapOpen = gapOpen;
    scorer->gapExtend = gapExtend;
    
    size_t tableSize = maxCode * maxCode * sizeof(int);    
    scorer->table = (int*) malloc(tableSize);
    memcpy(scorer->table, scores, tableSize);

    scorer->maxCode = maxCode;
    scorer->maxScore = maxScore(scorer);
    scorer->scalar = isScalar(scorer);
    
    return scorer;
}

extern void scorerDelete(Scorer* scorer) {
    free(scorer->name);
    free(scorer->table);
    free(scorer);
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// GETTERS

extern int scorerGetGapExtend(Scorer* scorer) {
    return scorer->gapExtend;
}

extern int scorerGetGapOpen(Scorer* scorer) {
    return scorer->gapOpen;
}

extern char scorerGetMaxCode(Scorer* scorer) {
    return scorer->maxCode;
}

extern int scorerGetMaxScore(Scorer* scorer) {
    return scorer->maxScore;
}

extern const char* scorerGetName(Scorer* scorer) {
    return scorer->name;
}

extern int scorerIsScalar(Scorer* scorer) {
    return scorer->scalar;
}

extern const int* scorerGetTable(Scorer* scorer) {
    return scorer->table;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// FUNCTIONS

extern int scorerScore(Scorer* scorer, char a, char b) {
    return scorer->table[(unsigned char) a * scorer->maxCode + (unsigned char) b];
}

extern Scorer* scorerDeserialize(char* bytes) {

    int ptr = 0;
    
    int nameLen;
    memcpy(&nameLen, bytes + ptr, sizeof(int));
    ptr += sizeof(int);
    
    char* name = (char*) malloc(nameLen);
    memcpy(name, bytes + ptr, nameLen);
    ptr += nameLen;
    
    int gapOpen;
    memcpy(&gapOpen, bytes + ptr, sizeof(int));
    ptr += sizeof(int);
    
    int gapExtend;
    memcpy(&gapExtend, bytes + ptr, sizeof(int));
    ptr += sizeof(int);
    
    char maxCode;
    memcpy(&maxCode, bytes + ptr, sizeof(char));
    ptr += sizeof(char);
    
    size_t tableSize = maxCode * maxCode * sizeof(int);
    int* table = (int*) malloc(tableSize);
    memcpy(table, bytes + ptr, tableSize);
    ptr += tableSize;
    
    int maxScore;
    memcpy(&maxScore, bytes + ptr, sizeof(int));
    ptr += sizeof(int);
    
    int scalar;
    memcpy(&scalar, bytes + ptr, sizeof(int));
    ptr += sizeof(int);
    
    Scorer* scorer = (Scorer*) malloc(sizeof(struct Scorer));
    
    scorer->nameLen = nameLen;
    scorer->name = name;
    scorer->gapOpen = gapOpen;
    scorer->gapExtend = gapExtend;
    scorer->maxCode = maxCode;
    scorer->maxScore = maxScore;
    scorer->scalar = scalar;
    scorer->table = table;
    
    return scorer;
}

extern void scorerSerialize(char** bytes, int* bytesLen, Scorer* scorer) {

    size_t tableSize = scorer->maxCode * scorer->maxCode * sizeof(int);
    
    *bytesLen = 0;
    *bytesLen += sizeof(int); // nameLen
    *bytesLen += scorer->nameLen; // name
    *bytesLen += sizeof(int); // gapOpen
    *bytesLen += sizeof(int); // gapExtend
    *bytesLen += sizeof(char); // maxCode
    *bytesLen += tableSize; // table
    *bytesLen += sizeof(int); // maxScore
    *bytesLen += sizeof(int); // scalar

    *bytes = (char*) malloc(*bytesLen);
        
    int ptr = 0;
    
    memcpy(*bytes + ptr, &scorer->nameLen, sizeof(int));
    ptr += sizeof(int);
    
    memcpy(*bytes + ptr, scorer->name, scorer->nameLen);
    ptr += scorer->nameLen;
    
    memcpy(*bytes + ptr, &scorer->gapOpen, sizeof(int));
    ptr += sizeof(int);
    
    memcpy(*bytes + ptr, &scorer->gapExtend, sizeof(int));
    ptr += sizeof(int);
    
    memcpy(*bytes + ptr, &scorer->maxCode, sizeof(char));
    ptr += sizeof(char);
    
    memcpy(*bytes + ptr, scorer->table, tableSize);
    ptr += tableSize;
    
    memcpy(*bytes + ptr, &scorer->maxScore, sizeof(int));
    ptr += sizeof(int);
    
    memcpy(*bytes + ptr, &scorer->scalar, sizeof(int));
    ptr += sizeof(int);
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// STATIC

extern char scorerDecode(char c) {
    return DECODER[(unsigned char) c];
}

extern char scorerEncode(char c) {
    return CODER[(unsigned char) c];
}

//------------------------------------------------------------------------------
//******************************************************************************

//******************************************************************************
// PRIVATE

static int isScalar(Scorer* scorer) {
    
    int x, i, j;

    int scorerLen = scorer->maxCode;

    x = scorer->table[0];
    for (i = 1; i < scorerLen; ++i) {
        if (scorer->table[i * scorerLen + i] != x) {
            return 0;
        }
    }
    
    x = scorer->table[1];
    for (i = 0; i < scorerLen; ++i) {
        for (j = 0; j < scorerLen; ++j) {
            if (i != j && scorer->table[i * scorerLen + j] != x) {
                return 0;
            }
        }
    }
    
    return 1;
}

static int maxScore(Scorer* scorer) {
    
    int i, j;
    
    int scorerLen = scorer->maxCode;
    
    int max = scorer->table[0];
    for (i = 0; i < scorerLen; ++i) {
        for (j = 0; j < scorerLen; ++j) {
            max = MAX(max, scorer->table[i * scorerLen + j]);
        }
    }
    
    return max;
}

//******************************************************************************
