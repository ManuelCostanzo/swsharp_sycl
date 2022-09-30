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
/**
@file

@brief Postprocessing utilities header.
*/

#ifndef __SW_SHARP_POST_PROCESH__
#define __SW_SHARP_POST_PROCESH__

#include "alignment.h"
#include "chain.h"
#include "db_alignment.h"
#include "scorer.h"

#ifdef __cplusplus 
extern "C" {
#endif

/*!
@brief Pairwise alignment pair output format.

Format is equal to the emboss pair output without the header. Emboss output 
example: <a>http://emboss.sourceforge.net/docs/themes/alnformats/align.pair</a>
*/
#define SW_OUT_PAIR         0

/*!
@brief Pairwise alignment plotting output format.

This output format is used as the input file for plotting with gnuplot.
*/
#define SW_OUT_PLOT         1

/*!
@brief Pairwise alignment stat output format.

Format is equal to the emboss pair output header. Emboss output example:
<a>http://emboss.sourceforge.net/docs/themes/alnformats/align.pair</a>
*/
#define SW_OUT_STAT         2

/*!
@brief Pairwise alignment stat-pair output format.

Format is equal to the emboss pair output. Emboss output example:
<a>http://emboss.sourceforge.net/docs/themes/alnformats/align.pair</a>
*/
#define SW_OUT_STAT_PAIR    3

/*!
@brief Pairwise alignment binary output format.

This output can be read with the readAlignment().
*/
#define SW_OUT_DUMP         4

/*!
@brief Database alignment blast m1-like output format.

Blast m1 output example: 
<a>http://www.compbio.ox.ac.uk/analysis_tools/BLAST/BLAST_blastall/blastall_examples.shtml#m_0</a>
*/
#define SW_OUT_DB_BLASTM0   0

/*!
@brief Database alignment blast m8-like output format.

Blast m8 output example: 
<a>http://www.compbio.ox.ac.uk/analysis_tools/BLAST/BLAST_blastall/blastall_examples.shtml#m_8</a>
*/
#define SW_OUT_DB_BLASTM8   1

/*!
@brief Database alignment blast m9-like output format.

Blast m9 output example: 
<a>http://www.compbio.ox.ac.uk/analysis_tools/BLAST/BLAST_blastall/blastall_examples.shtml#m_9</a>
*/
#define SW_OUT_DB_BLASTM9   2

/*!
@brief Database alignment light output format.

Format outputs only the scores and the names of the output alignments.
*/
#define SW_OUT_DB_LIGHT     3

/*!
@brief Checks if the alignment is correct.

@param alignment alignment object

@return 1 if the alignment is correct, 0 otherwise
*/
extern int checkAlignment(Alignment* alignment);

/*!
@brief Reads the alignment object from the binary file.

The binary file must previously be created with the outputAlignment() function
with the #SW_OUT_DUMP type.

@param path input binary file path

@return alignment object
*/
extern Alignment* readAlignment(char* path);

/*!
@brief Pairwise alignment output function.

@param alignment alignment object
@param path output file path, if NULL output goes to standard output
@param type output format type, can be #SW_OUT_PAIR, #SW_OUT_PLOT, #SW_OUT_STAT,
    #SW_OUT_STAT_PAIR or #SW_OUT_DUMP
*/
extern void outputAlignment(Alignment* alignment, char* path, int type);

/*!
@brief Score only output function.

@param score alignment score
@param query query chain
@param target target chain
@param scorer scorer object used for alignment
@param path output file path, if NULL output goes to standard output
*/
extern void outputScore(int score, Chain* query, Chain* target, Scorer* scorer, 
    char* path);

/*!
@brief Database alignment output function.

@param dbAlignments database alignments array
@param dbAlignmentsLen database alignments array length
@param path output file path, if NULL output goes to standard output
@param type output format type, can be #SW_OUT_DB_BLASTM0 , #SW_OUT_DB_BLASTM8,
    #SW_OUT_DB_BLASTM9 or #SW_OUT_DB_LIGHT
*/
extern void outputDatabase(DbAlignment** dbAlignments, int dbAlignmentsLen, 
    char* path, int type);

/*!
@brief Shotgun database alignment output function.

@param dbAlignments database alignments array of arrays
@param dbAlignmentsLens database alignments arrays lengths
@param dbAlignmentsLen database alignments array of arrays length
@param path output file path, if NULL output goes to standard output
@param type output format type, can be #SW_OUT_DB_BLASTM0 , #SW_OUT_DB_BLASTM8,
    #SW_OUT_DB_BLASTM9 or #SW_OUT_DB_LIGHT
*/
extern void outputShotgunDatabase(DbAlignment*** dbAlignments, 
    int* dbAlignmentsLens, int dbAlignmentsLen, char* path, int type);
    
/*!
@brief Chain array delete utility.

Function calls chainDelete() function on every chain and then deletes the array.

@param chains chain array object
@param chainsLen chain array length
*/
extern void deleteFastaChains(Chain** chains, int chainsLen);

/*!
@brief Database delete utility.

Function calls dbAlignmentDelete() function on every dbAlignment object in 
dbAlignments array and then deletes the array.

@param dbAlignments database alignments array
@param dbAlignmentsLen database alignments array length
*/
extern void deleteDatabase(DbAlignment** dbAlignments, int dbAlignmentsLen);

/*!
@brief Shotgun database delete utility.

Function calls dbAlignmentDelete() function on every dbAlignment object in 
dbAlignments, deletes dbAlignments structure and then deletes dbAlignmentsLens 
array.

@param dbAlignments database alignments array of arrays
@param dbAlignmentsLens database alignments arrays lengths
@param dbAlignmentsLen database alignments array of arrays length
*/
extern void deleteShotgunDatabase(DbAlignment*** dbAlignments, 
    int* dbAlignmentsLens, int dbAlignmentsLen);

extern void dbAlignmentsMerge(DbAlignment*** dbAlignmentsDst, 
    int* dbAlignmentsDstLens, DbAlignment*** dbAlignmentsSrc, 
    int* dbAlignmentsSrcLens, int dbAlignmentsLen, int maxAlignments);

#ifdef __cplusplus 
}
#endif
#endif // __SW_SHARP_POST_PROCESH__
