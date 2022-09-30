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

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chain.h"
#include "constants.h"
#include "error.h"
#include "scorer.h"
#include "utils.h"

#include "pre_proc.h"

#define SCORERS_LEN (sizeof(scorers) / sizeof(ScorerEntry))

typedef struct ScorerEntry {
    const char* name;
    int (*table)[26 * 26];
} ScorerEntry;

// to register a scorer just add his name and corresponding table to this array
static ScorerEntry scorers[] = {
    { "BLOSUM_62", &BLOSUM_62_TABLE }, // default one
    { "BLOSUM_45", &BLOSUM_45_TABLE },
    { "BLOSUM_50", &BLOSUM_50_TABLE },
    { "BLOSUM_80", &BLOSUM_80_TABLE },
    { "BLOSUM_90", &BLOSUM_90_TABLE },
    { "PAM_30", &PAM_30_TABLE },
    { "PAM_70", &PAM_70_TABLE },
    { "PAM_250", &PAM_250_TABLE },
    { "EDNA_FULL", &EDNA_FULL_TABLE }
};

//******************************************************************************
// PUBLIC

//******************************************************************************

//******************************************************************************
// PRIVATE

static char* fastaChainsSerializedPath(const char* path);

static int readFastaChainsPartNormal(Chain*** chains, int* chainsLen,
    FILE* handle, const size_t maxBytes);

static int readFastaChainsPartSerialized(Chain*** chains, int* chainsLen,
    FILE* handle, const size_t maxBytes);

static int skipFastaChainsPartNormal(Chain*** chains, int* chainsLen,
    FILE* handle, const size_t skip);

static int skipFastaChainsPartSerialized(Chain*** chains, int* chainsLen,
    FILE* handle, const size_t skip);

//******************************************************************************

//******************************************************************************
// PUBLIC

//------------------------------------------------------------------------------
// CHAIN UTILS

extern Chain* createChainComplement(Chain* chain) {

    int length = chainGetLength(chain);
    char* string = (char*) malloc(length * sizeof(char));
    
    int i;
    for (i = 0; i < length; ++i) {
    
        char chr = chainGetChar(chain, i);
        
        switch(chr) {
            case 'A':
                chr = 'T';
                break;
            case 'T':
                chr = 'A';
                break;
            case 'C':
                chr = 'G';
                break;     
            case 'G':
                chr = 'C';
                break;       
        }
        
        string[length - 1 - i] = chr;
    }

    const char prefix[] = "complement: ";
    
    const char* name = chainGetName(chain);
    int nameLen = strlen(name);

    int newNameLen = nameLen + sizeof(prefix);
    char* newName = (char*) malloc(newNameLen);

    sprintf(newName, "%s%s", prefix, name);

    Chain* complement = chainCreate(newName, newNameLen - 1, string, length);

    free(newName);
    free(string);
    
    return complement;
}

extern void readFastaChain(Chain** chain, const char* path) {

    FILE* f = fileSafeOpen(path, "r");
    
    char* str = (char*) malloc(fileLength(f) * sizeof(char));
    int strLen = 0;
    
    int nameSize = 4096;
    char* name = (char*) malloc(nameSize * sizeof(char));
    int nameLen = 0;
    
    char buffer[4096];
    int isName = 1;
    
    while (!feof(f)) {
        
        int read = fread(buffer, sizeof(char), 4096, f);
        
        int i;
        for (i = 0; i < read; ++i) {
            
            char c = buffer[i];
            
            if (isName) {
                if (c == '\n') {
                    name[nameLen] = 0;
                    isName = 0;
                } else if (!(nameLen == 0 && (c == '>' || isspace(c)))) {
                    if (c != '\r') {

                        if (nameLen == nameSize) {
                            nameSize *= 2;
                            name = (char*) realloc(name, nameSize * sizeof(char));
                        }

                        name[nameLen++] = c;
                    }
                }
            } else {
                str[strLen++] = c;
            }
        }
    }
    
    *chain = chainCreate(name, nameLen, str, strLen);

    free(str);
    free(name);
    
    fclose(f);
}

extern void readFastaChains(Chain*** chains, int* chainsLen, const char* path) {

    FILE* handle;
    int serialized;

    readFastaChainsPartInit(chains, chainsLen, &handle, &serialized, path);
    readFastaChainsPart(chains, chainsLen, handle, serialized, 0);

    fclose(handle);
}

extern void readFastaChainsPartInit(Chain*** chains, int* chainsLen, 
    FILE** handle, int* serialized, const char* path_) {

    char* path = fastaChainsSerializedPath(path_);

    *handle = fopen(path, "r");

    if (*handle == NULL) {
        *handle = fileSafeOpen(path_, "r");
        *serialized = 0;
    } else {
        *serialized = 1;
        WARNING(1, "Reading serilized database %s.", path);
    }

    *chains = NULL;
    *chainsLen = 0;

    free(path);
}

extern int readFastaChainsPart(Chain*** chains, int* chainsLen,
    FILE* handle, int serialized, const size_t maxBytes) {

    TIMER_START("Reading database (serialized %d)", serialized);

    int status;

    if (serialized) {
        status = readFastaChainsPartSerialized(chains, chainsLen, handle, maxBytes);
    } else {
        status = readFastaChainsPartNormal(chains, chainsLen, handle, maxBytes);
    }

    TIMER_STOP;

    return status;
}

extern int skipFastaChainsPart(Chain*** chains, int* chainsLen,
    FILE* handle, int serialized, const size_t skip) {

    int status;

    if (serialized) {
        status = skipFastaChainsPartSerialized(chains, chainsLen, handle, skip);
    } else {
        status = skipFastaChainsPartNormal(chains, chainsLen, handle, skip);
    }

    return status;
}

extern void statFastaChains(int* chains_, long long* cells_, const char* path_) {

    char* path = fastaChainsSerializedPath(path_);

    FILE* file = fopen(path, "r");

    int chains = 0;
    long long cells = 0;

    if (file == NULL) {

        file = fileSafeOpen(path_, "r");

        char buffer[1024 * 1024];
        int isName = 1;

        while (!feof(file)) {
            
            int read = fread(buffer, sizeof(char), 1024 * 1024, file);

            int i;
            for (i = 0; i < read; ++i) {
                switch (buffer[i]) {
                case '>':
                    if (!isName) {
                        chains++;
                    }
                    isName = 1;
                    break;
                case '\r':
                    break;
                case '\n':
                    isName = 0;
                    break;
                default:
                    if (!isName) {
                        cells++;
                    }
                    break;
                }
            }
        }

        if (!isName) {
            chains++;
        }

    } else {

        WARNING(1, "Reading serilized database %s.", path);

        ASSERT(fread(&chains, sizeof(int), 1, file) == 1, "io error");
        ASSERT(fread(&cells, sizeof(long long), 1, file) == 1, "io error");
    }

    fclose(file);
    free(path);

    if (chains_ != NULL) *chains_ = chains;
    if (cells_ != NULL) *cells_ = cells;
}

extern void dumpFastaChains(char* path_) {

    static const size_t readChunk = 200 * 1024 * 1024; // 200MB

    char* path = fastaChainsSerializedPath(path_);
    FILE* file = fopen(path, "r");

    if (file != NULL) {
        WARNING(1, "File %s exists, chains are not dumped.", path);
    } else {

        Chain** chains;
        int chainsStart = 0;
        int chainsLen;

        FILE* handle;
        int serialized;

        readFastaChainsPartInit(&chains, &chainsLen, &handle, &serialized, path_);

        int length;
        long long cells;
        statFastaChains(&length, &cells, path_);

        file = fileSafeOpen(path, "w");

        fwrite(&length, sizeof(int), 1, file);
        fwrite(&cells, sizeof(long long), 1, file);

        LOG("Dumping chains to: %s", path);

        while (1) {

            int status = readFastaChainsPart(&chains, &chainsLen, handle, 
                serialized, readChunk);

            int chainIdx;
            for (chainIdx = chainsStart; chainIdx < chainsLen; ++chainIdx) {
            
                Chain* chain = chains[chainIdx];
                
                char* buffer;
                int bufferLen;
                chainSerialize(&buffer, &bufferLen, chain);
                
                fwrite(&bufferLen, sizeof(int), 1, file);
                fwrite(buffer, sizeof(char), bufferLen, file);
                
                free(buffer);
                chainDelete(chain);
            }

            if (status == 0) {
                break;
            }

            chainsStart = chainsLen;
        }

        free(chains);
        fclose(handle);
    }

    free(path);
    fclose(file);
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// SCORES UTILS

extern void scorerCreateScalar(Scorer** scorer, int match, int mismatch, 
    int gapOpen, int gapExtend) {
    
    int scores[26 * 26];
    
    int i, j;
    for (i = 0; i < 26; ++i) {
        for (j = 0; j < 26; ++j) {
            scores[i * 26 + j] = i == j ? match : mismatch;
        }
    }
    
    char name[100];
    sprintf(name, "match/mismatch +%d/%d", match, mismatch);
    
    *scorer = scorerCreate(name, scores, 26, gapOpen, gapExtend);
}

extern void scorerCreateMatrix(Scorer** scorer, char* name, int gapOpen, 
    int gapExtend) {
    
    int index = -1;
  
    int i;
    for (i = 0; i < SCORERS_LEN; ++i) {
        if (strcmp(name, scorers[i].name) == 0) {
            index = i;
            break;
        }
    }
    
    ASSERT(index != -1, "unknown table %s", name);
    
    ScorerEntry* entry = &(scorers[index]);
    *scorer = scorerCreate(entry->name, *(entry->table), 26, gapOpen, gapExtend);
}

//------------------------------------------------------------------------------
//******************************************************************************

//******************************************************************************
// PRIVATE

//------------------------------------------------------------------------------
// DATABASE UTILS

static char* fastaChainsSerializedPath(const char* path_) {

    static const char ext[] = ".swsharp";

    char* path = (char*) malloc(strlen(path_) + sizeof(ext) + 1);
    sprintf(path, "%s%s", path_, ext);

    return path;
}

static int readFastaChainsPartNormal(Chain*** chains, int* chainsLen,
    FILE* handle, const size_t maxBytes) {

    static const int chainsStep = 100000;

    size_t chainsSize;

    if (*chains == NULL) {
        chainsSize = chainsStep;
        *chains = (Chain**) malloc(chainsSize * sizeof(Chain*));
        *chainsLen = 0;
    } else {
        chainsSize = *chainsLen + chainsStep;
        *chains = (Chain**) realloc(*chains, chainsSize * sizeof(Chain*));
    }

    size_t strSize = 65000;
    char* str = (char*) malloc(strSize * sizeof(char));
    int strLen = 0;

    size_t nameSize = 4096;
    char* name = (char*) malloc(nameSize * sizeof(char));
    int nameLen = 0;
    
    char buffer[1024 * 1024];
    int isName = 1;

    size_t bytesRead = 0;
    long int bytesOver = 0;
    int status = 0;

    int isEnd = feof(handle);

    while (!isEnd) {
        
        int read = fread(buffer, sizeof(char), 1024 * 1024, handle);
        isEnd = feof(handle);

        bytesRead += read;

        if (maxBytes != 0 && bytesRead > maxBytes) {
            fseek(handle, -(bytesOver + read), SEEK_CUR);
            status = 1;
            break;
        }

        int i;
        for (i = 0; i < read; ++i) {
            
            char c = buffer[i];

            if (!isName && (c == '>' || (isEnd && i == read - 1))) {

                bytesOver = 0;

                isName = 1;
                
                Chain* chain = chainCreate(name, nameLen, str, strLen);
                
                if (*chainsLen + 1 == chainsSize) {
                    chainsSize += chainsStep;
                    *chains = (Chain**) realloc(*chains, chainsSize * sizeof(Chain*));
                }

                (*chains)[(*chainsLen)++] = chain;
                      
                nameLen = 0;
                strLen = 0;
            }
            
            if (isName) {
                if (c == '\n') {
                    name[nameLen] = 0;
                    isName = 0;
                } else if (!(nameLen == 0 && (c == '>' || isspace(c)))) {
                    if (c != '\r') {

                        if (nameLen == nameSize) {
                            nameSize *= 2;
                            name = (char*) realloc(name, nameSize * sizeof(char));
                        }

                        name[nameLen++] = c;
                    }              
                }
            } else {
                if (strLen == strSize) {
                    strSize *= 2;
                    str = (char*) realloc(str, strSize * sizeof(char));
                }
                str[strLen++] = c;
            }

            bytesOver++;
        }
    }

    free(str);
    free(name);

    return status;
}

static int readFastaChainsPartSerialized(Chain*** chains, int* chainsLen,
    FILE* handle, const size_t maxBytes) {

    if (feof(handle)) {
        return 0;
    }

    if (*chains == NULL) {

        int length;
        long long cells;

        ASSERT(fread(&length, sizeof(int), 1, handle) == 1, "io error");
        ASSERT(fread(&cells, sizeof(long long), 1, handle) == 1, "io error");

        *chains = (Chain**) malloc(length * sizeof(Chain*));
        *chainsLen = 0;
    }

    int bufferSize = 65000;
    char* buffer = (char*) malloc(bufferSize);
    
    int chainSize;

    size_t bytesRead = 0;
    int status = 0;

    while (!feof(handle)) {

        if (fread(&chainSize, sizeof(int), 1, handle) != 1) {
            break;
        }

        bytesRead += chainSize;

        if (maxBytes != 0 && bytesRead > maxBytes) {
            fseek(handle, -sizeof(int), SEEK_CUR);
            status = 1;
            break;
        }

        if (chainSize > bufferSize) {
            bufferSize = 2 * chainSize;
            buffer = (char*) realloc(buffer, bufferSize);
        }
        
        ASSERT(fread(buffer, 1, chainSize, handle) == chainSize, "io error");

        Chain* chain = chainDeserialize(buffer);
        (*chains)[(*chainsLen)++] = chain;
    }
    
    free(buffer);

    return status;
}

static int skipFastaChainsPartNormal(Chain*** chains, int* chainsLen,
    FILE* handle, const size_t skip) {

    size_t chainsSize;

    if (*chains == NULL) {
        chainsSize = skip;
        *chains = (Chain**) malloc(chainsSize * sizeof(Chain*));
        *chainsLen = 0;
    } else {
        chainsSize = *chainsLen + skip;
        *chains = (Chain**) realloc(*chains, chainsSize * sizeof(Chain*));
    }

    char buffer[1024 * 1024];
    int isName = 1;
    int chainsRead = 0;

    size_t bytesRead = 0;
    long int bytesOver = 0;
    int status = 0;

    int isEnd = feof(handle);

    while (!isEnd) {
        
        if (chainsRead >= skip) {
            fseek(handle, -bytesOver, SEEK_CUR);
            status = 1;
            break;
        }

        int read = fread(buffer, sizeof(char), 1024 * 1024, handle);
        isEnd = feof(handle);

        bytesRead += read;

        int i;
        for (i = 0; i < read; ++i) {
            
            char c = buffer[i];

            if (!isName && (c == '>' || (isEnd && i == read - 1))) {

                bytesOver = 0;
                chainsRead++;

                if (chainsRead >= skip) {
                    bytesOver += read - i;
                    break;
                }

                isName = 1;
            }
            
            if (isName) {
                if (c == '\n') {
                    isName = 0;
                }
            }

            bytesOver++;
        }
    }

    int chainIdx;
    for (chainIdx = *chainsLen; chainIdx < *chainsLen + chainsRead; ++chainIdx) {
        (*chains)[chainIdx] = NULL;
    }

    *chainsLen += chainsRead;

    return status;
}

static int skipFastaChainsPartSerialized(Chain*** chains, int* chainsLen,
    FILE* handle, const size_t skip) {

    if (feof(handle)) {
        return 0;
    }

    if (*chains == NULL) {

        int length;
        long long cells;

        ASSERT(fread(&length, sizeof(int), 1, handle) == 1, "io error");
        ASSERT(fread(&cells, sizeof(long long), 1, handle) == 1, "io error");

        *chains = (Chain**) malloc(length * sizeof(Chain*));
        *chainsLen = 0;
    }

    int chainSize;

    size_t read = 0;
    int status = 0;

    while (!feof(handle)) {

        if (fread(&chainSize, sizeof(int), 1, handle) != 1) {
            break;
        }

        read++;

        if (read > skip) {
            fseek(handle, -sizeof(int), SEEK_CUR);
            status = 1;
            read--;
            break;
        }
        
        ASSERT(fseek(handle, chainSize, SEEK_CUR) == 0, "io error");
    }
    
    int chainIdx;
    for (chainIdx = *chainsLen; chainIdx < *chainsLen + read; ++chainIdx) {
        (*chains)[chainIdx] = NULL;
    }

    *chainsLen += read;

    return status;
}

//------------------------------------------------------------------------------
//******************************************************************************
