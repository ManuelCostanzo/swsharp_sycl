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

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "alignment.h"
#include "chain.h"
#include "constants.h"
#include "error.h"
#include "scorer.h"
#include "sse_module.h"
#include "utils.h"

#include "cpu_module.h"

typedef struct Move {
    char move;
    int vGaps;
    int hGaps;
} Move;

typedef struct HBus {
    int scr;
    int aff;
} HBus;

//******************************************************************************
// PUBLIC

extern void alignPairCpu(Alignment** alignment, int type, Chain* query, 
    Chain* target, Scorer* scorer);
    
extern void alignScoredPairCpu(Alignment** alignment, int type, Chain* query, 
    Chain* target, Scorer* scorer, int score);

extern void nwFindScoreCpu(int* queryStart, int* targetStart, Chain* query, 
    int queryFrontGap, Chain* target, Scorer* scorer, int score);
    
extern void nwReconstructCpu(char** path, int* pathLen, int* outScore, 
    Chain* query, int queryFrontGap, int queryBackGap, Chain* target, 
    int targetFrontGap, int targetBackGap, Scorer* scorer, int score);

extern void ovFindScoreCpu(int* queryStart, int* targetStart, Chain* query, 
    Chain* target, Scorer* scorer, int score);

extern int scorePairCpu(int type, Chain* query, Chain* target, Scorer* scorer);

extern void scoreDatabaseCpu(int* scores, int type, Chain* query, 
    Chain** database, int databaseLen, Scorer* scorer);

extern void scoreDatabasePartiallyCpu(int* scores, int type, Chain* query, 
    Chain** database, int databaseLen, Scorer* scorer, int maxScore);

//******************************************************************************

//******************************************************************************
// PRIVATE

static void hwAlign(Alignment** alignment, Chain* query, Chain* target, 
    Scorer* scorer, int score);
    
static int hwScore(Chain* query, Chain* target, Scorer* scorer);

static void nwAlign(Alignment** alignment, Chain* query, Chain* target, 
    Scorer* scorer, int score);
    
static int nwScore(Chain* query, Chain* target, Scorer* scorer);

static void ovAlign(Alignment** alignment, Chain* query, Chain* target, 
    Scorer* scorer, int score);

static int ovScore(Chain* query, Chain* target, Scorer* scorer);

static void swAlign(Alignment** alignment, Chain* query, Chain* target, 
    Scorer* scorer, int score);

static int swScore(Chain* query, Chain* target, Scorer* scorer);

//******************************************************************************

//******************************************************************************
// PUBLIC

extern void alignPairCpu(Alignment** alignment, int type, Chain* query, 
    Chain* target, Scorer* scorer) {
    alignScoredPairCpu(alignment, type, query, target, scorer, NO_SCORE);
}

extern void alignScoredPairCpu(Alignment** alignment, int type, Chain* query, 
    Chain* target, Scorer* scorer, int score) {
    
    void (*function) (Alignment**, Chain*, Chain*, Scorer*, int);

    if (alignScoredPairSse(alignment, type, query, target, scorer, score) == 0) {
        return;
    }

    switch (type) {
    case HW_ALIGN: 
        function = hwAlign;
        break;
    case NW_ALIGN: 
        function = nwAlign;
        break;
    case SW_ALIGN: 
        function = swAlign;
        break;
    case OV_ALIGN: 
        function = ovAlign;
        break;
    default:
        ERROR("invalid align type");
    }
    
    function(alignment, query, target, scorer, score);
    
    int outScore = alignmentGetScore(*alignment);
    ASSERT(score == NO_SCORE || score == outScore, "invalid alignment input score %s %s",
            chainGetName(query), chainGetName(target));
}

extern int scorePairCpu(int type, Chain* query, Chain* target, Scorer* scorer) {

    int (*function) (Chain*, Chain*, Scorer*);

    if (type != HW_ALIGN) {
        if (chainGetLength(query) < chainGetLength(target)) {
            SWAP(query, target);
        }
    }

    int score;
    if (scorePairSse(&score, type, query, target, scorer) == 0) {
        return score;
    }

    switch (type) {
    case HW_ALIGN: 
        function = hwScore;
        break;
    case NW_ALIGN: 
        function = nwScore;
        break;
    case SW_ALIGN: 
        function = swScore;
        break;
    case OV_ALIGN: 
        function = ovScore;
        break;
    default:
        ERROR("invalid align type");
    }
    
    return function(query, target, scorer);
}

extern void scoreDatabaseCpu(int* scores, int type, Chain* query, 
    Chain** database, int databaseLen, Scorer* scorer) {

    // if sse is available return
    if (scoreDatabaseSse(scores, type, query, database, databaseLen, scorer) == 0) {
        return;
    }

    int databaseIdx;
    for (databaseIdx = 0; databaseIdx < databaseLen; ++ databaseIdx) {
        Chain* target = database[databaseIdx];
        scores[databaseIdx] = scorePairCpu(type, query, target, scorer);
    }
}


extern void scoreDatabasePartiallyCpu(int* scores, int type, Chain* query, 
    Chain** database, int databaseLen, Scorer* scorer, int maxScore) {

    if (maxScore < 0 && type == SW_ALIGN) {
        memset(scores, 0, databaseLen * sizeof(int));
        return;
    }

    if (maxScore == INT_MAX) {
        scoreDatabaseCpu(scores, type, query, database, databaseLen, scorer);
        return;
    }

    int status = scoreDatabasePartiallySse(scores, type, query, database, 
        databaseLen, scorer, maxScore);

    // if sse is available return
    if (status == 0) {
        return;
    }

    int databaseIdx;
    for (databaseIdx = 0; databaseIdx < databaseLen; ++databaseIdx) {
        Chain* target = database[databaseIdx];
        scores[databaseIdx] = MIN(maxScore, scorePairCpu(type, query, target, scorer));
    }
}

//------------------------------------------------------------------------------
// NW MODULES

extern void nwReconstructCpu(char** path, int* pathLen, int* outScore, 
    Chain* query, int queryFrontGap, int queryBackGap, Chain* target, 
    int targetFrontGap, int targetBackGap, Scorer* scorer, int score) {

    ASSERT(!(queryFrontGap && targetFrontGap), "cannot start with both gaps");
    ASSERT(!(queryBackGap && targetBackGap), "cannot end with both gaps");
    
    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);

    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    
    if (rows == 0 || cols == 0) {
    
        *pathLen = rows + cols;

        char move = rows == 0 ? MOVE_LEFT : MOVE_UP;
        
        *path = (char*) malloc(*pathLen * sizeof(char));
        memset(*path, move, *pathLen * sizeof(char));
        
        if (outScore != NULL) {
            *outScore = -gapOpen - gapExtend * MAX(0, *pathLen - 1);
        }
        
        return;
    }
    
    int maxScore = scorerGetMaxScore(scorer);
    int minMatch = maxScore ? score / maxScore : 0;
    int t = MAX(rows, cols) - minMatch;
    int p = (t - abs(rows - cols)) / 2 + 1;
    
    if (score < 0) {
        p = MAX(rows, cols);
    }

    // perfect match, chains are equal
    if (t == 0) {
    
        *pathLen = rows;
        
        *path = (char*) malloc(*pathLen * sizeof(char));
        memset(*path, MOVE_DIAG, *pathLen * sizeof(char));
        
        if (outScore != NULL) {
            *outScore = score;
        }
        
        return;
    }
    
    int width = MIN(2 * p + abs(rows - cols) + 1, cols);

    HBus* hBus = (HBus*) malloc((width + 1) * sizeof(HBus));
        
    int movesLen = width * rows;
    Move* moves = (Move*) malloc(movesLen * sizeof(Move));
    
    int offL = cols >= rows ? p : p + rows - cols;
    int offR = cols >= rows ? p + cols - rows : p;
    
    int row;
    int col; 

    const char* const rowCodes = chainGetCodes(query);
    const char* const colCodes = chainGetCodes(target);

    const int* const scorerTable = scorerGetTable(scorer);
    int scorerMaxCode = scorerGetMaxCode(scorer);

    for (col = 0; col < width; ++col) {
        hBus[col].scr = -gapOpen - col * gapExtend;
        hBus[col].aff = SCORE_MIN;
    }
    
    hBus[width].scr = SCORE_MIN;
    hBus[width].aff = SCORE_MIN;
    
    for (row = 0; row < rows; ++row) {
    
        int start = MAX(0, row - offL);
        int end = MIN(cols, row + offR + 1); 
        
        int gap = -gapOpen - row * gapExtend;
        
        int iScr = start == 0 ? gap: SCORE_MIN;
        int iAff = SCORE_MIN;
        
        int diag = start == 0 ? (gap + gapExtend) * (row > 0) : hBus[0].scr;
        
        for (col = 0; col < end - start; ++col) {
        
            int up = col + (start != 0);
            int moveIdx = row * width + col;

            // MATCHING
            int mch = scorerTable[rowCodes[row] * scorerMaxCode + colCodes[col + start]] + diag;
            // MATCHING END
            
            // INSERT                
            int ins = MAX(iScr - gapOpen, iAff - gapExtend); 
            
            if (ins == iAff - gapExtend) {
                moves[moveIdx].hGaps = moves[moveIdx - 1].hGaps + 1;
            } else {
                moves[moveIdx].hGaps = 0;
            }
            // INSERT END

            // DELETE 
            int del = MAX(hBus[up].scr - gapOpen, hBus[up].aff - gapExtend); 
           
            if (del == hBus[up].aff - gapExtend) {
                int up = moveIdx - width + (start != 0);
                moves[moveIdx].vGaps = moves[up].vGaps + 1;
            } else {
                moves[moveIdx].vGaps = 0;
            } 
            // DELETE END
            
            int scr = MAX(mch, MAX(ins, del));
            
            if (row == 0 && queryFrontGap) {
                scr = del;
                moves[moveIdx].move = MOVE_UP;
            } else if (col + start == 0 && targetFrontGap) {
                scr = ins;
                moves[moveIdx].move = MOVE_LEFT;
            } else if (row == rows - 1 && queryBackGap) {
                scr = del;
                moves[moveIdx].move = MOVE_UP;
            } else if (col + start == cols - 1 && targetBackGap) {
                scr = ins;
                moves[moveIdx].move = MOVE_LEFT;
            } else if (del == scr) {
                moves[moveIdx].move = MOVE_UP;
            } else if (ins == scr) {
                moves[moveIdx].move = MOVE_LEFT;
            } else {
                moves[moveIdx].move = MOVE_DIAG;
            }
            
            // UPDATE BUSES  
            iScr = scr;
            iAff = ins;
            
            diag = hBus[up].scr;
            
            hBus[col].scr = scr;
            hBus[col].aff = del;
            // UPDATE BUSES END
        }
    }
    
    row = rows - 1;
    col = cols - 1 - MAX(0, row - offL);
    
    if (score != NO_SCORE) {
        ASSERT(hBus[col].scr == score, "invalid nw block align %s %s",
            chainGetName(query), chainGetName(target));
    }
    
    // save the score if needed
    if (outScore != NULL) {
        *outScore = hBus[col].scr;
    }
    
    int pathEnd = rows + cols + 1;
    int pathIdx = pathEnd;
    
    *path = (char*) malloc(pathEnd * sizeof(char));
    
    do {

        int movesIdx = row * width + col;
        char move = moves[movesIdx].move;
               
        (*path)[--pathIdx] = move;
        
        if (move == MOVE_DIAG) {
            col -= (row - offL) <= 0;
            row--;
        } else if (move == MOVE_LEFT) {

            int gaps = moves[movesIdx].hGaps;
            
            pathIdx -= gaps;
            memset((*path) + pathIdx, MOVE_LEFT, gaps);
            
            col -= gaps + 1;

        } else if (move == MOVE_UP) {

            int gaps = moves[movesIdx].vGaps;
            
            pathIdx -= gaps;
            memset((*path) + pathIdx, MOVE_UP, gaps);

            col += MAX(0, row - offL) - MAX(0, row - gaps - 1 - offL);
            row -= gaps + 1;
        }
        
        // vertical gaps till the end
        if (row == -1 && col >= 0) {
        
            pathIdx -= col + 1;
            memset((*path) + pathIdx, MOVE_LEFT, col + 1);
            
            break;
        }
        
        // horizontal gaps till the end
        if (col == -1 && row >= 0) {
        
            pathIdx -= row + 1;
            memset((*path) + pathIdx, MOVE_UP, row + 1);
            
            break;
        }
        
    } while (row >= 0 || col >= 0);

    (*pathLen) = pathEnd - pathIdx;
    
    // shift data to begining of the array
    int shiftIdx;
    for (shiftIdx = 0; shiftIdx < *pathLen; ++shiftIdx) {
        (*path)[shiftIdx] = (*path)[pathIdx + shiftIdx];
    }   
    
    free(hBus);
    free(moves);
}

extern void nwFindScoreCpu(int* queryStart, int* targetStart, Chain* query, 
    int queryFrontGap, Chain* target, Scorer* scorer, int score) {
    
    *queryStart = -1;
    *targetStart = -1;
    
    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);
    int gapDiff = gapOpen - gapExtend;

    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    
    HBus* hBus = (HBus*) malloc(cols * sizeof(HBus));
        
    int row;
    int col; 

    const char* const rowCodes = chainGetCodes(query);
    const char* const colCodes = chainGetCodes(target);

    const int* const scorerTable = scorerGetTable(scorer);
    int scorerMaxCode = scorerGetMaxCode(scorer);

    for (col = 0; col < cols; ++col) {
        hBus[col].scr = -gapOpen - col * gapExtend;
        hBus[col].aff = SCORE_MIN;
    }
    
    for (row = 0; row < rows; ++row) {
    
        int iScr = -gapOpen - row * gapExtend + queryFrontGap * gapDiff;
        int iAff = SCORE_MIN;
        
        int diag = (-gapOpen - (row - 1) * gapExtend + queryFrontGap * gapDiff) * (row > 0);
                
        for (col = 0; col < cols; ++col) {
        
            // MATCHING
            int mch = scorerTable[rowCodes[row] * scorerMaxCode + colCodes[col]] + diag;
            // MATCHING END
            
            // INSERT                
            int ins = MAX(iScr - gapOpen, iAff - gapExtend); 
            // INSERT END

            // DELETE 
            int del = MAX(hBus[col].scr - gapOpen, hBus[col].aff - gapExtend); 
            // DELETE END
            
            int scr = MAX(mch, MAX(ins, del));
            
            if (scr == score) {
                *queryStart = row;
                *targetStart = col;
                free(hBus);
                return;
            }
                       
            // UPDATE BUSES  
            iScr = scr;
            iAff = ins;
            
            diag = hBus[col].scr;
            
            hBus[col].scr = scr;
            hBus[col].aff = del;
            // UPDATE BUSES END
        }
    }
    
    free(hBus);
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// OV MODULES

extern void ovFindScoreCpu(int* queryStart, int* targetStart, Chain* query, 
    Chain* target, Scorer* scorer, int score) {
    
    *queryStart = -1;
    *targetStart = -1;
    
    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);

    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    
    HBus* hBus = (HBus*) malloc(cols * sizeof(HBus));
        
    int row;
    int col; 

    const char* const rowCodes = chainGetCodes(query);
    const char* const colCodes = chainGetCodes(target);

    const int* const scorerTable = scorerGetTable(scorer);
    int scorerMaxCode = scorerGetMaxCode(scorer);

    for (col = 0; col < cols; ++col) {
        hBus[col].scr = -gapOpen - col * gapExtend;
        hBus[col].aff = SCORE_MIN;
    }
    
    for (row = 0; row < rows; ++row) {
    
        int iScr = -gapOpen - row * gapExtend;
        int iAff = SCORE_MIN;
        
        int diag = (-gapOpen - (row - 1) * gapExtend) * (row > 0);
                
        for (col = 0; col < cols; ++col) {
        
            // MATCHING
            int mch = scorerTable[rowCodes[row] * scorerMaxCode + colCodes[col]] + diag;
            // MATCHING END
            
            // INSERT                
            int ins = MAX(iScr - gapOpen, iAff - gapExtend); 
            // INSERT END

            // DELETE 
            int del = MAX(hBus[col].scr - gapOpen, hBus[col].aff - gapExtend); 
            // DELETE END
            
            int scr = MAX(mch, MAX(ins, del));
            
            if (scr == score && (row == rows - 1 || col == cols - 1)) {
                *queryStart = row;
                *targetStart = col;
                free(hBus);
                return;
            }
                       
            // UPDATE BUSES  
            iScr = scr;
            iAff = ins;
            
            diag = hBus[col].scr;
            
            hBus[col].scr = scr;
            hBus[col].aff = del;
            // UPDATE BUSES END
        }
    }
    
    free(hBus);
}
//------------------------------------------------------------------------------
//******************************************************************************

//******************************************************************************
// PRIVATE

//------------------------------------------------------------------------------
// HW MODULES

static void hwAlign(Alignment** alignment, Chain* query, Chain* target, 
    Scorer* scorer, int score) {
    
    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);

    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    
    HBus* hBus = (HBus*) malloc(cols * sizeof(HBus));
        
    int movesLen = cols * rows;
    Move* moves = (Move*) malloc(movesLen * sizeof(Move));
    
    int row;
    int col; 

    const char* const rowCodes = chainGetCodes(query);
    const char* const colCodes = chainGetCodes(target);

    const int* const scorerTable = scorerGetTable(scorer);
    int scorerMaxCode = scorerGetMaxCode(scorer);

    for (col = 0; col < cols; ++col) {
        hBus[col].scr = 0;
        hBus[col].aff = SCORE_MIN;
    }
    
    int outScore = SCORE_MIN;
    int endRow = 0;
    int endCol = 0;
    
    for (row = 0; row < rows; ++row) {
    
        int iScr = -gapOpen - row * gapExtend;
        int iAff = SCORE_MIN;
        
        int diag = (-gapOpen - (row - 1) * gapExtend) * (row > 0);
        
        for (col = 0; col < cols; ++col) {
        
            int moveIdx = row * cols + col;

            // MATCHING
            int mch = scorerTable[rowCodes[row] * scorerMaxCode + colCodes[col]] + diag;
            // MATCHING END
            
            // INSERT                
            int ins = MAX(iScr - gapOpen, iAff - gapExtend); 
            
            if (ins == iAff - gapExtend) {
                moves[moveIdx].hGaps = moves[moveIdx - 1].hGaps + 1;
            } else {
                moves[moveIdx].hGaps = 0;
            }
            // INSERT END

            // DELETE 
            int del = MAX(hBus[col].scr - gapOpen, hBus[col].aff - gapExtend); 
           
            if (del == hBus[col].aff - gapExtend) {
                moves[moveIdx].vGaps = moves[moveIdx - cols].vGaps + 1;
            } else {
                moves[moveIdx].vGaps = 0;
            } 
            // DELETE END
            
            int scr = MAX(mch, MAX(ins, del));
            
            if (del == scr) {
                moves[moveIdx].move = MOVE_UP;
            } else if (ins == scr) {
                moves[moveIdx].move = MOVE_LEFT;
            } else {
                moves[moveIdx].move = MOVE_DIAG;
            }
            
            if (scr > outScore && row == rows - 1) {
                outScore = scr;
                endRow = row;
                endCol = col;
            }
           
            // UPDATE BUSES  
            iScr = scr;
            iAff = ins;
            
            diag = hBus[col].scr;
            
            hBus[col].scr = scr;
            hBus[col].aff = del;
            // UPDATE BUSES END
        }
    }
    
    row = endRow;
    col = endCol;
    
    int pathEnd = endRow + endCol + 1;
    int pathIdx = pathEnd;
    
    char* path = (char*) malloc(pathEnd * sizeof(char));
    
    while (row >= 0 && col >= 0) {
        
        int movesIdx = row * cols + col;
        char move = moves[movesIdx].move;
               
        path[--pathIdx] = move;
        
        if (move == MOVE_DIAG) {
            col--;
            row--;
        } else if (move == MOVE_LEFT) {

            int gaps = moves[movesIdx].hGaps;
            
            pathIdx -= gaps;
            memset(path + pathIdx, MOVE_LEFT, gaps);
            
            col -= gaps + 1;

        } else if (move == MOVE_UP) {

            int gaps = moves[movesIdx].vGaps;
            
            pathIdx -= gaps;
            memset(path + pathIdx, MOVE_UP, gaps);

            row -= gaps + 1;
            
        }
        
        // horizontal gaps till the end
        if (col == -1 && row >= 0) {
        
            pathIdx -= row + 1;
            memset(path + pathIdx, MOVE_UP, row + 1);

            col = 0;
            row = 0;
            
            break;
        }
    }
    
    // don't count last move
    if (row == -1) {
        row++;
        col++;
    }
        
    int pathLen = pathEnd - pathIdx;
    
    // shift data to begining of the array
    int shiftIdx;
    for (shiftIdx = 0; shiftIdx < pathLen; ++shiftIdx) {
        path[shiftIdx] = path[pathIdx + shiftIdx];
    }
    
    free(hBus);
    free(moves);
    
    *alignment = alignmentCreate(query, row, endRow, target, col, endCol, 
        outScore, scorer, path, pathLen);
}

static int hwScore(Chain* query, Chain* target, Scorer* scorer) {

    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);

    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    
    int max = SCORE_MIN;
    
    HBus* hBus = (HBus*) malloc(cols * sizeof(HBus));
        
    int row;
    int col; 

    const char* const rowCodes = chainGetCodes(query);
    const char* const colCodes = chainGetCodes(target);

    const int* const scorerTable = scorerGetTable(scorer);
    int scorerMaxCode = scorerGetMaxCode(scorer);

    for (col = 0; col < cols; ++col) {
        hBus[col].scr = 0;
        hBus[col].aff = SCORE_MIN;
    }
    
    for (row = 0; row < rows; ++row) {
    
        int iScr = -gapOpen - row * gapExtend;
        int iAff = SCORE_MIN;
        
        int diag = (-gapOpen - (row - 1) * gapExtend) * (row > 0);
        
        for (col = 0; col < cols; ++col) {
        
            // MATCHING
            int mch = scorerTable[rowCodes[row] * scorerMaxCode + colCodes[col]] + diag;
            // MATCHING END
            
            // INSERT                
            int ins = MAX(iScr - gapOpen, iAff - gapExtend); 
            // INSERT END

            // DELETE 
            int del = MAX(hBus[col].scr - gapOpen, hBus[col].aff - gapExtend); 
            // DELETE END
            
            int scr = MAX(mch, MAX(ins, del));
           
            if (row == rows - 1) {
                max = MAX(max, scr);
            }
            
            // UPDATE BUSES  
            iScr = scr;
            iAff = ins;
            
            diag = hBus[col].scr;
            
            hBus[col].scr = scr;
            hBus[col].aff = del;
            // UPDATE BUSES END
        }
    }
    
    free(hBus);
    
    return max;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// NW MODULES

static void nwAlign(Alignment** alignment, Chain* query, Chain* target, 
    Scorer* scorer, int score) {
    
    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    
    char* path;
    int pathLen;
    int outScore;
    
    nwReconstructCpu(&path, &pathLen, &outScore, query, 0, 0, target, 0, 0, 
        scorer, score);
    
    *alignment = alignmentCreate(query, 0, rows - 1, target, 0, cols - 1, 
        outScore, scorer, path, pathLen);
}

static int nwScore(Chain* query, Chain* target, Scorer* scorer) {

    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);

    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    
    HBus* hBus = (HBus*) malloc(cols * sizeof(HBus));
        
    int row;
    int col; 

    const char* const rowCodes = chainGetCodes(query);
    const char* const colCodes = chainGetCodes(target);

    const int* const scorerTable = scorerGetTable(scorer);
    int scorerMaxCode = scorerGetMaxCode(scorer);

    for (col = 0; col < cols; ++col) {
        hBus[col].scr = -gapOpen - col * gapExtend;
        hBus[col].aff = SCORE_MIN;
    }
    
    for (row = 0; row < rows; ++row) {
    
        int iScr = -gapOpen - row * gapExtend;
        int iAff = SCORE_MIN;
        
        int diag = (-gapOpen - (row - 1) * gapExtend) * (row > 0);
        
        for (col = 0; col < cols; ++col) {
        
            // MATCHING
            int mch = scorerTable[rowCodes[row] * scorerMaxCode + colCodes[col]] + diag;
            // MATCHING END
            
            // INSERT                
            int ins = MAX(iScr - gapOpen, iAff - gapExtend); 
            // INSERT END

            // DELETE 
            int del = MAX(hBus[col].scr - gapOpen, hBus[col].aff - gapExtend); 
            // DELETE END
            
            int scr = MAX(mch, MAX(ins, del));
           
            // UPDATE BUSES  
            iScr = scr;
            iAff = ins;
            
            diag = hBus[col].scr;
            
            hBus[col].scr = scr;
            hBus[col].aff = del;
            // UPDATE BUSES END
        }
    }
    
    int score = hBus[cols - 1].scr;
    
    free(hBus);
    
    return score;
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// OV MODULES

static void ovAlign(Alignment** alignment, Chain* query, Chain* target, 
    Scorer* scorer, int score) {
    
    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);

    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    
    HBus* hBus = (HBus*) malloc(cols * sizeof(HBus));
        
    int movesLen = cols * rows;
    Move* moves = (Move*) malloc(movesLen * sizeof(Move));
    
    int row;
    int col; 

    const char* const rowCodes = chainGetCodes(query);
    const char* const colCodes = chainGetCodes(target);

    const int* const scorerTable = scorerGetTable(scorer);
    int scorerMaxCode = scorerGetMaxCode(scorer);

    for (col = 0; col < cols; ++col) {
        hBus[col].scr = 0;
        hBus[col].aff = SCORE_MIN;
    }
    
    int outScore = SCORE_MIN;
    int endRow = 0;
    int endCol = 0;
    
    for (row = 0; row < rows; ++row) {
    
        int iScr = 0;
        int iAff = SCORE_MIN;
        
        int diag = 0;
        
        for (col = 0; col < cols; ++col) {
        
            int moveIdx = row * cols + col;

            // MATCHING
            int mch = scorerTable[rowCodes[row] * scorerMaxCode + colCodes[col]] + diag;
            // MATCHING END
            
            // INSERT                
            int ins = MAX(iScr - gapOpen, iAff - gapExtend); 
            
            if (ins == iAff - gapExtend) {
                moves[moveIdx].hGaps = moves[moveIdx - 1].hGaps + 1;
            } else {
                moves[moveIdx].hGaps = 0;
            }
            // INSERT END

            // DELETE 
            int del = MAX(hBus[col].scr - gapOpen, hBus[col].aff - gapExtend); 
           
            if (del == hBus[col].aff - gapExtend) {
                moves[moveIdx].vGaps = moves[moveIdx - cols].vGaps + 1;
            } else {
                moves[moveIdx].vGaps = 0;
            } 
            // DELETE END
            
            int scr = MAX(mch, MAX(ins, del));
            
            if (del == scr) {
                moves[moveIdx].move = MOVE_UP;
            } else if (ins == scr) {
                moves[moveIdx].move = MOVE_LEFT;
            } else {
                moves[moveIdx].move = MOVE_DIAG;
            }
            
            if (scr > outScore && (row == rows - 1 || col == cols - 1)) {
                outScore = scr;
                endRow = row;
                endCol = col;
            }
           
            // UPDATE BUSES  
            iScr = scr;
            iAff = ins;
            
            diag = hBus[col].scr;
            
            hBus[col].scr = scr;
            hBus[col].aff = del;
            // UPDATE BUSES END
        }
    }

    row = endRow;
    col = endCol;
    
    int pathEnd = endRow + endCol + 1;
    int pathIdx = pathEnd;
    
    char* path = (char*) malloc(pathEnd * sizeof(char));
    
    while (row >= 0 && col >= 0) {
        
        int movesIdx = row * cols + col;
        char move = moves[movesIdx].move;
               
        path[--pathIdx] = move;
        
        if (move == MOVE_DIAG) {
            col--;
            row--;
        } else if (move == MOVE_LEFT) {

            int gaps = moves[movesIdx].hGaps;
            
            pathIdx -= gaps;
            memset(path + pathIdx, MOVE_LEFT, gaps);
            
            col -= gaps + 1;

        } else if (move == MOVE_UP) {

            int gaps = moves[movesIdx].vGaps;
            
            pathIdx -= gaps;
            memset(path + pathIdx, MOVE_UP, gaps);

            row -= gaps + 1;
            
        }
    }
    
    // don't count last move (diagonal) out of the matrix
    if (row == -1 || col == -1) {
        row++;
        col++;
    }

    if (pathEnd - pathIdx <= 0) {
        *alignment = alignmentCreate(query, 0, 0, target, 0, 0, outScore, scorer, NULL, 0);
    } else {

        int pathLen = pathEnd - pathIdx;
        
        // shift data to begining of the array
        int shiftIdx;
        for (shiftIdx = 0; shiftIdx < pathLen; ++shiftIdx) {
            path[shiftIdx] = path[pathIdx + shiftIdx];
        }

        *alignment = alignmentCreate(query, row, endRow, target, col, endCol, 
            outScore, scorer, path, pathLen);
    }

    free(hBus);
    free(moves);
}

static int ovScore(Chain* query, Chain* target, Scorer* scorer) {

    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);

    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    
    int max = SCORE_MIN;
    
    HBus* hBus = (HBus*) malloc(cols * sizeof(HBus));
        
    int row;
    int col; 

    const char* const rowCodes = chainGetCodes(query);
    const char* const colCodes = chainGetCodes(target);

    const int* const scorerTable = scorerGetTable(scorer);
    int scorerMaxCode = scorerGetMaxCode(scorer);

    for (col = 0; col < cols; ++col) {
        hBus[col].scr = 0;
        hBus[col].aff = SCORE_MIN;
    }
    
    for (row = 0; row < rows; ++row) {
    
        int iScr = 0;
        int iAff = SCORE_MIN;
        
        int diag = 0;
        
        for (col = 0; col < cols; ++col) {
        
            // MATCHING
            int mch = scorerTable[rowCodes[row] * scorerMaxCode + colCodes[col]] + diag;
            // MATCHING END
            
            // INSERT                
            int ins = MAX(iScr - gapOpen, iAff - gapExtend); 
            // INSERT END

            // DELETE 
            int del = MAX(hBus[col].scr - gapOpen, hBus[col].aff - gapExtend); 
            // DELETE END
            
            int scr = MAX(mch, MAX(ins, del));
           
            if (row == rows - 1 || col == cols - 1) {
                max = MAX(max, scr);
            }
            
            // UPDATE BUSES  
            iScr = scr;
            iAff = ins;
            
            diag = hBus[col].scr;
            
            hBus[col].scr = scr;
            hBus[col].aff = del;
            // UPDATE BUSES END
        }
    }
    
    free(hBus);
    
    return max;
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// SW MODULES

static void swAlign(Alignment** alignment, Chain* query, Chain* target, 
    Scorer* scorer, int score) {
    
    if (scorerGetMaxScore(scorer) <= 0) {
        *alignment = alignmentCreate(query, 0, 0, target, 0, 0, 0, scorer, NULL, 0);
        return;
    }

    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);

    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    
    HBus* hBus = (HBus*) malloc(cols * sizeof(HBus));

    int movesLen = cols * rows;
    Move* moves = (Move*) malloc(movesLen * sizeof(Move));
    
    int row;
    int col; 
    
    for (col = 0; col < cols; ++col) {
        hBus[col].scr = 0;
        hBus[col].aff = SCORE_MIN;
    }

    const char* const rowCodes = chainGetCodes(query);
    const char* const colCodes = chainGetCodes(target);

    const int* const scorerTable = scorerGetTable(scorer);
    int scorerMaxCode = scorerGetMaxCode(scorer);

    int outScore = 0;
    int endRow = 0;
    int endCol = 0;
    
    int pruning = 1;
    int pruneLow = 0;
    int pruneHigh = cols;
    int pruneFactor = scorerGetMaxScore(scorer);
    
    int bestScore = MAX(outScore, score);

    for (row = 0; row < rows; ++row) {
    
        int iScr = 0;
        int iAff = SCORE_MIN;
        
        int diag = 0;
        
        if (pruning && row > rows / 2) {
        
            for (col = pruneLow; rows - row <= cols - col; ++col) {
            
                int scr = hBus[col].scr; 
                if (col > 0) {
                    scr = MAX(scr, hBus[col - 1].scr);
                }
                
                if (scr + (rows - row) * pruneFactor < bestScore) {
                    pruneLow = col;
                } else {
                    break;
                }
            }

            if (pruneLow != 0) {
                diag = hBus[pruneLow - 1].scr;
            }
        }
        
        if (pruning) {
        
            pruneHigh = cols;
            for (col = cols - 1; cols - col <= rows - row; --col) {
            
                int scr = hBus[col].scr; 
                if (col > 0) {
                    scr = MAX(scr, hBus[col - 1].scr);
                }
                
                if (scr + (cols - col) * pruneFactor < bestScore) {
                    pruneHigh = col;
                } else {
                    break;
                }
            }
            
            if (pruneHigh < cols) {
                hBus[pruneHigh].scr = 0;
                hBus[pruneHigh].aff = SCORE_MIN;
            }
        }
        
        for (col = pruneLow; col < pruneHigh; ++col) {
        
            int moveIdx = row * cols + col;

            // MATCHING
            int mch = scorerTable[rowCodes[row] * scorerMaxCode + colCodes[col]] + diag;
            // MATCHING END
            
            // INSERT                
            int ins = MAX(iScr - gapOpen, iAff - gapExtend); 
            
            if (ins == iAff - gapExtend) {
                moves[moveIdx].hGaps = moves[moveIdx - 1].hGaps + 1;
            } else {
                moves[moveIdx].hGaps = 0;
            }
            // INSERT END

            // DELETE 
            int del = MAX(hBus[col].scr - gapOpen, hBus[col].aff - gapExtend); 
           
            if (del == hBus[col].aff - gapExtend) {
                moves[moveIdx].vGaps = moves[moveIdx - cols].vGaps + 1;
            } else {
                moves[moveIdx].vGaps = 0;
            } 
            // DELETE END
            
            int scr = MAX(MAX(0, mch), MAX(ins, del));
            
            if (del == scr) {
                moves[moveIdx].move = MOVE_UP;
            } else if (ins == scr) {
                moves[moveIdx].move = MOVE_LEFT;
            } else if (mch == scr) {
                moves[moveIdx].move = MOVE_DIAG;
            } else {
                moves[moveIdx].move = MOVE_STOP;
            }
            
            if (scr > outScore) {
                outScore = scr;
                endRow = row;
                endCol = col;
                bestScore = MAX(bestScore, scr);
            }
           
            // UPDATE BUSES  
            iScr = scr;
            iAff = ins;
            
            diag = hBus[col].scr;
            
            hBus[col].scr = scr;
            hBus[col].aff = del;
            // UPDATE BUSES END
        }
    }
    
    if ((endRow == 0 || endCol == 0) && bestScore == 0) {

        free(hBus);
        free(moves);

        *alignment = alignmentCreate(query, 0, 0, target, 0, 0, 0, scorer, NULL, 0);

        return;
    }

    row = endRow;
    col = endCol;
    
    int pathEnd = endRow + endCol + 1;
    int pathIdx = pathEnd;
    
    char* path = (char*) malloc(pathEnd * sizeof(char));
    
    while (row >= 0 && col >= 0) {
        
        int movesIdx = row * cols + col;
        char move = moves[movesIdx].move;
               
        path[--pathIdx] = move;
        
        if (move == MOVE_DIAG) {
            col--;
            row--;
        } else if (move == MOVE_LEFT) {

            int gaps = moves[movesIdx].hGaps;
            
            pathIdx -= gaps;
            memset(path + pathIdx, MOVE_LEFT, gaps);
            
            col -= gaps + 1;

        } else if (move == MOVE_UP) {

            int gaps = moves[movesIdx].vGaps;
            
            pathIdx -= gaps;
            memset(path + pathIdx, MOVE_UP, gaps);

            row -= gaps + 1;
            
        } else {
            // don't count the stop move, it came from the diagonal
            row++;
            col++;
            pathIdx++;
            break;
        }
    }
    
    // don't count last move (diagonal) out of the matrix
    if (row == -1 || col == -1) {
        row++;
        col++;
    }
        
    int pathLen = pathEnd - pathIdx;
    
    // shift data to begining of the array
    int shiftIdx;
    for (shiftIdx = 0; shiftIdx < pathLen; ++shiftIdx) {
        path[shiftIdx] = path[pathIdx + shiftIdx];
    }
    
    free(hBus);
    free(moves);
    
    *alignment = alignmentCreate(query, row, endRow, target, col, endCol, 
        outScore, scorer, path, pathLen);
}

static int swScore(Chain* query, Chain* target, Scorer* scorer) {

    if (scorerGetMaxScore(scorer) <= 0) {
        return 0;
    }

    int gapOpen = scorerGetGapOpen(scorer);
    int gapExtend = scorerGetGapExtend(scorer);

    int rows = chainGetLength(query);
    int cols = chainGetLength(target);
    
    int max = 0;
    
    HBus* hBus = (HBus*) malloc(cols * sizeof(HBus));
    memset(hBus, 0, cols * sizeof(HBus));

    int row;
    int col; 

    const char* const rowCodes = chainGetCodes(query);
    const char* const colCodes = chainGetCodes(target);

    const int* const scorerTable = scorerGetTable(scorer);
    int scorerMaxCode = scorerGetMaxCode(scorer);

    int pruning = 1;
    int pruneLow = 0;
    int pruneHigh = cols;
    int pruneFactor = scorerGetMaxScore(scorer);
    
    for (row = 0; row < rows; ++row) {
    
        int iScr = 0;
        int iAff = SCORE_MIN;
        
        int diag = 0;
        
        if (pruning && row > rows / 2) {
        
            for (col = pruneLow; rows - row <= cols - col; ++col) {
            
                int scr = hBus[col].scr; 
                if (col > 0) {
                    scr = MAX(scr, hBus[col - 1].scr);
                }
                
                if (scr + (rows - row) * pruneFactor < max) {
                    pruneLow = col;
                } else {
                    break;
                }
            }

            if (pruneLow != 0) {
                diag = hBus[pruneLow - 1].scr;
            }
        }
        
        if (pruning) {
        
            pruneHigh = cols;
            for (col = cols - 1; cols - col <= rows - row; --col) {
            
                int scr = hBus[col].scr; 
                if (col > 0) {
                    scr = MAX(scr, hBus[col - 1].scr);
                }
                
                if (scr + (cols - col) * pruneFactor < max) {
                    pruneHigh = col;
                } else {
                    break;
                }
            }
            
            if (pruneHigh < cols) {
                hBus[pruneHigh].scr = 0;
                hBus[pruneHigh].aff = SCORE_MIN;
            }
        }
        
        for (col = pruneLow; col < pruneHigh; ++col) {
        
            // MATCHING
            int mch = scorerTable[rowCodes[row] * scorerMaxCode + colCodes[col]] + diag;
            // MATCHING END
            
            // INSERT                
            int ins = MAX(iScr - gapOpen, iAff - gapExtend); 
            // INSERT END

            // DELETE 
            int del = MAX(hBus[col].scr - gapOpen, hBus[col].aff - gapExtend); 
            // DELETE END
            
            int scr = MAX(MAX(0, mch), MAX(ins, del));
           
            max = MAX(max, scr);
            
            // UPDATE BUSES  
            iScr = scr;
            iAff = ins;
            
            diag = hBus[col].scr;
            
            hBus[col].scr = scr;
            hBus[col].aff = del;
            // UPDATE BUSES END
        }
    }
    
    free(hBus);
    
    return max;
}
//------------------------------------------------------------------------------
//******************************************************************************
